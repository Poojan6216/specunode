"""The run loop. One code path, whether or not anything is speculating.

Section 2.2 is emphatic that a sequential run is *a chain of single-branch retirements* rather
than a second implementation, and this module is where that promise is kept. There is no
``if policy.speculation`` in the loop body. Sequential is the case where the drafters offer no
candidate, so exactly one branch exists per decision point, it is confirmed by the model's own
output, and it retires. Phase 3 adds candidates; it does not add a path.

Three orderings in here are load-bearing, and each has a concrete failure behind it.

**The canonical branch is minted before the turn, not after it.** A turn journaled with no
owning branch has no ``branch_id``, which is a required field on ``model_request``; worse, the
branch that later retires would have an empty ``prompts_sent``, the Hard Rule 13 rebuild would
loop over nothing, and the ledger would still stamp the run as context-checked. A stamp for a
property nobody checked is worse than no stamp, so the count is journaled and the stamp reads
``unchecked`` at zero.

**The confirming entry is durable before anything drains.** That is Hard Rule 3 stated as an
ordering, and the drain re-asserts it rather than trusting this module, because task 1.6 plants
exactly the bug of doing it the other way round.

**The drain runs on this task, never on a branch's.** A node body that writes and then reads
the result -- ``ack = await charge_card(...)`` then ``send_receipt(ack["charge_id"])`` -- parks
on a future only the drain completes. The repair that suggests itself when a node is waiting is
to drain inline from the branch task, which dispatches before the branch is confirmed; the
store buffer refuses a caller that is not this task, so that repair is unavailable rather than
merely discouraged.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import EffectOutcome, StoreBuffer
from specunode.canonical import JsonValue, chash
from specunode.core.branch import Branch, BranchStatus, ReadRecord, StepCursor
from specunode.core.decision import Decision, ToolCall, decision_key, decision_payload, is_barrier
from specunode.core.effects import EffectClass, ToolRegistry
from specunode.core.graph import END, GraphAdapter, NodeRef, Parallel, RunSession, session_scope
from specunode.core.hazards import Hazard, analyse, keys_touched
from specunode.core.model import (
    CallScope,
    JournaledModel,
    ModelClient,
    ModelResponse,
    RequestEnvelope,
    ToolUseComplete,
    TurnComplete,
    TurnResults,
    call_scope,
)
from specunode.core.policy import Budget, Policy
from specunode.core.state import (
    CommittedState,
    Reducer,
    _touched_keys,
    patch_payload,
    resolve_reducers,
)
from specunode.drafters.base import DraftContext, Drafter
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.journal.ledger import Ledger, build_ledger
from specunode.journal.replay import OpenGroup, RecordedTurns, recover
from specunode.verify.gate import resolve_decision
from specunode.verify.witness import ReadValidation, validate_reads

__all__ = ["BranchOutcome", "RunResult", "Scheduler", "SchedulerError", "SpeculativeTurn"]


class SchedulerError(RuntimeError):
    """The run cannot continue."""


class BranchOutcome(Enum):
    """How a branch's task came to rest."""

    DONE = "done"
    #: Waiting on a future only the runtime can complete -- a staged write's result. Not an
    #: error and not a stall: the branch is mid-node and the drain is what unblocks it.
    PARKED = "parked"
    STALLED = "stalled"
    FAULTED = "faulted"


@dataclass
class RunResult:
    run_id: str
    ok: bool
    state: dict[str, JsonValue]
    ledger: Ledger
    steps: int = 0
    error: str | None = None


class ParallelWriteConflict(SchedulerError):
    """Two nodes declared independent both wrote one state key, and no reducer says how."""

    def __init__(self, key: str, first: str, second: str) -> None:
        super().__init__(
            f"parallel nodes {first} and {second} both write state key {key!r}; declare a "
            "reducer for it, or run them one after another"
        )
        self.key = key


def _not_retired(node_name: str, branch: Branch) -> str:
    """Why a node that reached retirement did not retire, in words that send the operator to
    the right place.

    Three failures arrive here and they are different facts. A node refused at retirement --
    a witnessed read gone stale -- sent nothing; its writes were discarded unsent. A node that
    raised after its writes went out has effects in the world. And a write that would not
    dispatch is a fault in the tool, not in the node. Reporting the first as the second sends
    the operator looking for an effect that never happened.
    """
    if branch.status is BranchStatus.SQUASHED:
        return (
            f"node {node_name} was refused at retirement and nothing it staged was sent: "
            f"{branch.reason or 'no reason recorded'}"
        )
    if branch.reason:
        return f"node {node_name} failed after its effects were dispatched: {branch.reason}"
    return f"effects from node {node_name} did not all dispatch"


def _cursor_payload(cursor: StepCursor) -> JsonValue:
    """A program position as the journal records it, and as recovery reads it back."""
    return {
        "step_index": cursor.step_index,
        "visits": [[name, count] for name, count in cursor.visits],
    }


@dataclass(frozen=True)
class _GroupOutcome:
    ok: bool
    error: str | None
    committed: CommittedState
    cursor: StepCursor
    retired: int


def _first_conflict(
    lanes: Sequence[tuple[NodeRef, str]],
    branches: Sequence[Branch],
    reducers: Mapping[str, Reducer],
    claimed: Mapping[str, str] | None = None,
) -> ParallelWriteConflict | None:
    """The first state key two lanes both wrote without a reducer, in declared order.

    ``claimed`` is the keys lanes already retired have committed. A key in a parked lane's
    delta is one its body has already written, so a clash found mid-body is a real one, even
    if the body later puts the old value back. What a mid-body check cannot see is what the
    body writes after it resumes.
    """
    claimed = dict(claimed or {})
    for (_node, node_id), branch in zip(lanes, branches, strict=True):
        for key in sorted(_touched_keys(branch.state.delta())):
            owner = claimed.get(key)
            if owner is not None and key not in reducers:
                return ParallelWriteConflict(key, owner, node_id)
            claimed.setdefault(key, node_id)
    return None


@dataclass
class _Counters:
    branches_forked: int = 0
    branches_retired: int = 0
    branches_squashed: int = 0
    branches_stalled: int = 0
    effects_staged: int = 0
    effects_dispatched: int = 0
    effects_dead_lettered: int = 0
    effects_discarded: int = 0
    speculative_reads_upstream: int = 0
    #: Why the gate closed, if it did. None on a run where it never did.
    speculation_disabled_reason: str | None = None
    #: E3's tallies. Zero on a run with no speculative reads, which is honest; they were zero
    #: on *every* run while ``validate_reads`` had no caller, which was not.
    reads_validated: int = 0
    reads_stale: int = 0
    context_divergences: int = 0
    wasted_tokens: int = 0
    stalls_by_hazard: dict[str, int] = field(default_factory=dict)


class BranchTools:
    """The tool port. A READ runs; anything that mutates waits in the store buffer.

    This is the only route from a node body to the world that the runtime can see, which is
    also why a speculative branch does not run node bodies by default: anything a node does
    outside this port is invisible to the fake world, and therefore invisible to the leak test.
    """

    def __init__(self, scheduler: Scheduler, branch: Branch, node_id: str) -> None:
        self._scheduler = scheduler
        self._branch = branch
        self._node_id = node_id

    async def call(
        self,
        name: str,
        args: Mapping[str, JsonValue],
        *,
        step: int | None = None,
        pending_writes: Sequence[frozenset[str]] = (),
    ) -> JsonValue:
        """Issue one tool call. ``step`` names the program position instead of taking the next.

        A node body calling this directly takes the next position, which is right: the runtime
        has no other notion of where it is. A model-emitted block passes its *ordinal*, because
        the position a call occupies is where the model put it, not when the runtime got round
        to running it -- and early issue and speculation both run calls out of that order.
        """
        scheduler = self._scheduler
        branch = self._branch
        spec = scheduler.registry.get(name)
        call = ToolCall(name=name, args=dict(args))
        step = branch.advance_step() if step is None else branch.reserve_step(step)

        hazard = analyse(
            branch,
            call,
            spec,
            scheduler.policy,
            # The buffer's staged writes, PLUS writes this turn has already emitted but not
            # yet staged. A turn stages its writes only after the stream ends -- deliberately,
            # so a staged effect never precedes a durable turn -- while reads are issued as
            # their blocks parse. So for the one shape READ_AFTER_STAGED_WRITE exists to catch,
            # a read that follows a write *inside a single model turn*, the buffer was empty at
            # the moment the read was analysed and the hazard could not fire. The model was
            # handed a pre-write value with no stall and no note in the ledger.
            staged_keys=(*scheduler.buffer.staged_keys(branch), *pending_writes),
        )
        if hazard is not None:
            scheduler.record_stall(step, hazard)

        call_id = new_ulid()
        await scheduler.journal.append_async(
            scheduler.run_id,
            "tool_request",
            {
                "v": 1,
                "step": step,
                "branch_id": branch.id,
                "lineage": list(branch.lineage),
                "node_id": self._node_id,
                "call_id": call_id,
                "tool": name,
                "args": dict(args),
                "args_hash": chash(dict(args)),
                "effect": spec.effect.value,
                "idempotent": spec.idempotent,
                "mode": "executed" if spec.effect is EffectClass.READ else "staged",
                "speculative": branch.status is BranchStatus.SPECULATIVE,
                # The program position this call occupies, which is what Rule 13 would
                # assemble results by. It used to journal ``len(branch.read_set)`` -- the
                # number of reads completed so far -- so every staged write recorded 0, the
                # counter never reset per turn, and several calls in one turn shared an
                # ordinal. A field documented as "the ordering key Rule 13 assembles results
                # by -- never completion order" was recording something close to completion
                # order, for reads only.
                "program_order": step,
            },
        )

        if spec.effect is EffectClass.READ:
            return await scheduler.execute_read(branch, call, spec, call_id, step, self._node_id)

        # ``step``, not the cursor: this call's program position is the one reserved above.
        effect = await scheduler.buffer.stage(branch, call, spec, node_id=self._node_id, step=step)
        scheduler.counters.effects_staged += 1
        ack = scheduler.buffer.ack_for(effect.id)
        # Parking, not blocking: the scheduler is told the branch is waiting on a future only
        # the drain completes, so it can proceed to confirm and drain rather than deadlock.
        scheduler.mark_parked(branch)
        return await ack


@dataclass
class Scheduler:
    """Drives a graph, one decision point at a time."""

    graph: GraphAdapter
    registry: ToolRegistry
    journal: Journal
    buffer: StoreBuffer
    dispatcher: Dispatcher
    target: ModelClient
    policy: Policy = field(default_factory=Policy)
    drafters: Sequence[object] = ()
    #: The tier-1 (or tier-2) predictor, if one is configured. Tier 0 needs no object: it is
    #: the stream itself, and it is always on.
    predictor: Drafter | None = None
    reducers: Mapping[str, str] = field(default_factory=dict)

    run_id: str = ""
    counters: _Counters = field(default_factory=_Counters)
    _cursor: StepCursor = field(default_factory=StepCursor)
    _committed: CommittedState = field(default_factory=CommittedState)
    _reducers: Mapping[str, Reducer] = field(default_factory=dict)
    #: A Parallel group a crash interrupted, which a resume finishes before routing on.
    _open_group: OpenGroup | None = None
    #: Node bodies still running, and the branch each runs on. A run that ends by an exception
    #: nothing catches -- cancelled from outside, by a timeout or a shutdown -- ends them too.
    _node_tasks: dict[asyncio.Task[Decision], Branch] = field(default_factory=dict)
    _steps: int = 0
    #: Signalled by the tool port when a branch parks on a staged write's result. An Event
    #: rather than a flag because the scheduler has to wait for the *next* park, not merely
    #: observe that one happened: a node released by a drain may stage again, and spinning on
    #: a flag cannot tell the two apart.
    _park_events: dict[str, asyncio.Event] = field(default_factory=dict)
    _stalls: list[tuple[int, Hazard]] = field(default_factory=list)
    _retire_seq: int = 0
    #: Turns run with early issue, kept so a test can assert on their timings.
    _turns: list[SpeculativeTurn] = field(default_factory=list)
    _budget: Budget | None = field(default=None, repr=False)

    # -- bookkeeping the ports call back into ------------------------------------------------

    @property
    def budget(self) -> Budget:
        """What speculation has spent, and what it may still spend (Hard Rule 10)."""
        if self._budget is None:
            self._budget = Budget(policy=self.policy)
        return self._budget

    def _park_event(self, branch_id: str) -> asyncio.Event:
        event = self._park_events.get(branch_id)
        if event is None:
            event = asyncio.Event()
            self._park_events[branch_id] = event
        return event

    def mark_parked(self, branch: Branch) -> None:
        """Signal this branch's own quiesce loop. Deliberately not resolved through adoption.

        Resolving through adoption looked right and cancelled another fix. An adopted child
        keeps running, and when it stages it calls this -- so the *parent's* event was set while
        the parent's node was still streaming its turn. The parent woke, reported PARKED, and
        drained a buffer holding the adopted effect and nothing else, because the writes from
        blocks the model emitted *earlier* are staged after the stream ends. The predicted call
        then reached the world before the call that preceded it, which is Hard Rule 9 failing on
        ordering, intermittently, so it reads as flakiness.

        The parent is woken where it actually needs the ack: the results loop signals before it
        awaits an adopted slot, by which time everything emitted earlier is staged and the drain
        walks them in program order. An effect staged late by an adopted child is still placed
        on the parent's list by :meth:`StoreBuffer.stage`, so it is picked up by that drain.
        """
        self._park_event(branch.id).set()

    def record_stall(self, step: int, hazard: Hazard) -> None:
        self._stalls.append((step, hazard))
        name = hazard.name
        self.counters.stalls_by_hazard[name] = self.counters.stalls_by_hazard.get(name, 0) + 1

    async def execute_read(
        self,
        branch: Branch,
        call: ToolCall,
        spec: object,
        call_id: str,
        step: int,
        node_id: str,
    ) -> JsonValue:
        """Run a read for real, and record enough to re-check it at retirement."""
        started = time.monotonic()
        tool = self.registry.get(call.name)
        # Either kind of "not authorised by a durable decision yet": a branch that is still a
        # guess, or a read issued for a turn whose output is not journaled. Attack 7.2 counts
        # both, because both reached upstream without a durable decision behind them.
        #
        # "A guess" is ``predicted is not None``, not ``status is SPECULATIVE``. Every branch is
        # SPECULATIVE until it is confirmed, the canonical one included, so the status test
        # counted an ordinary read on the ordinary path -- one the model had asked for, in a
        # turn already journaled -- as a read that reached upstream unauthorised. A plain
        # sequential run with speculation and early issue both switched off reported one, and
        # the ledger, the limitations page and attack 7.2's number all repeated it.
        # Two questions, and they had been sharing one answer.
        #
        # ``unretired`` is "could this read still go stale before the effects it informed reach
        # the world?" -- true for every read a branch makes before it retires, which is what
        # lattice rule E3 re-checks at retirement. Staleness is about elapsed time, not about
        # authorisation, so narrowing this would quietly shrink a safety property.
        #
        # ``unauthorised`` is "did this reach upstream with no durable decision behind it?" --
        # a guess, or a read issued for a turn that is not journaled yet. That is what attack
        # 7.2 counts, what the read budget charges, and what the ledger publishes. Every branch
        # is SPECULATIVE until it is confirmed, the canonical one included, so asking the
        # status counted an ordinary read on the ordinary path as unauthorised: a plain
        # sequential run with speculation and early issue both off reported one.
        unretired = branch.status is not BranchStatus.RETIRED
        unauthorised = branch.predicted is not None or branch.unjournaled_reads > 0
        scope = CallScope(
            run_id=self.run_id,
            branch_id=branch.id,
            lineage=branch.lineage,
            step=step,
            node_id=node_id,
            speculative=unauthorised,
        )
        token = call_scope.set(scope)
        try:
            value = await tool.fn(**dict(call.args))
        finally:
            call_scope.reset(token)

        witness: JsonValue = None
        if isinstance(value, Mapping) and tool.witness:
            witness = value.get("witness")
        branch.read_set.append(
            ReadRecord(
                tool=call.name,
                args=dict(call.args),
                args_hash=chash(dict(call.args)),
                result_hash=chash(value),
                witness=witness,
                at_step=step,
                issued_while_speculative=unretired,
            )
        )
        if unauthorised:
            self.counters.speculative_reads_upstream += 1
        if branch.predicted is not None:
            # The budget charges only reads made on a forked guess -- a branch that exists
            # because something predicted a decision, which is the test ``record_prompt`` uses
            # too. (Every branch is SPECULATIVE until it is confirmed, the canonical one
            # included, so status cannot tell a guess from the path the model really took.)
            # ``max_speculative_reads`` bounds what wrong guesses may cost. The wider tally
            # above also counts a read
            # issued early for a turn that is not yet durable (tier 0, on the canonical
            # branch), which attack 7.2 reports but which is not a guess -- charging those
            # closed the gate on runs that had no predictor at all.
            self.budget.record_speculative_read()
        await self.journal.append_async(
            self.run_id,
            "tool_result",
            {
                "v": 1,
                "step": step,
                "branch_id": branch.id,
                "call_id": call_id,
                "tool": call.name,
                "ok": True,
                "value": value,
                "value_hash": chash(value),
                "witness": witness,
                "source": "upstream",
                "duration_ms": int((time.monotonic() - started) * 1000),
                "speculative": unauthorised,
                "reached_upstream": True,
            },
        )
        return value

    # -- the loop ------------------------------------------------------------------------------

    async def run(self, run_id: str, inputs: JsonValue) -> RunResult:
        """Drive the graph to completion.

        Sequential mode is the case where no drafter offers a candidate, so each decision
        point has exactly one branch: the canonical one. It is confirmed by the model's own
        output rather than resolved against a prediction, and then retired. Phase 3 adds
        siblings beside it; the retirement sequence below does not change.
        """
        self.run_id = run_id
        # One source of truth for the run id. Letting the buffer carry its own lets the two
        # disagree, and the failure is silent and confident: effects are journaled under one
        # run while the ledger is built from another, so the run finishes ok with an empty
        # ledger and a world that was nonetheless changed.
        self.buffer.run_id = run_id
        self.buffer.scheduler_task = asyncio.current_task()
        self._committed = CommittedState.initial(inputs if isinstance(inputs, Mapping) else {})
        self._cursor = StepCursor()
        self._reducers = resolve_reducers(dict(self.reducers))
        if isinstance(self.target, JournaledModel):
            # A fresh run has been told nothing yet; a previous resume's turns are not its own.
            self.target.serve_recorded(None)
        await self._journal_run_started(inputs)

        if self.graph.capabilities().drives_itself:
            return await self._run_driven(run_id, inputs)

        return await self._drive_from_state(run_id)

    async def _drive_from_state(self, run_id: str) -> RunResult:
        """The scheduler-driven loop, from whatever committed state it was handed."""
        committed = self._committed
        cursor = self._cursor
        reducers = self._reducers
        ok, error, steps = True, None, 0
        open_group, self._open_group = self._open_group, None
        try:
            if open_group is not None:
                # Finish the group the crash interrupted before asking the router anything: it
                # was one decision, made on the state before any of its lanes committed.
                group = await self._resume_group(open_group, committed, reducers)
                committed, cursor = group.committed, group.cursor
                self._committed, self._cursor = committed, cursor
                steps += group.retired
                if not group.ok:
                    ok, error = False, group.error
            while ok:
                node = self.graph.next(committed.to_dict())
                if node is END or isinstance(node, type(END)):
                    break
                if isinstance(node, Parallel):
                    group = await self._run_group(node, cursor, committed, reducers)
                    committed, cursor = group.committed, group.cursor
                    self._committed, self._cursor = committed, cursor
                    steps += group.retired
                    if not group.ok:
                        ok, error = False, group.error
                        break
                    continue
                assert isinstance(node, NodeRef)
                cursor, node_id = cursor.visit(node.structural_id)

                branch = self._fork_canonical(cursor, node_id)
                await self._journal_fork(branch, node_id)

                outcome, decision = await self._run_node(node, branch, node_id, committed)
                if outcome is BranchOutcome.FAULTED:
                    # Close the branch's lifecycle first. This loop journaled the fork and then
                    # broke straight out, so a node that raised here left a ``branch_forked``
                    # with no ``branch_resolved`` -- a branch that ended in none of the three
                    # ways branch.py says every branch ends, and invisible to every reader that
                    # selects on a resolution. The adapter-driven path has always done this.
                    await self._journal_faulted(branch, node_id)
                    # The reason the node failed is the whole of the diagnostic value here.
                    # A replay that refuses because the prompt changed reports the step index
                    # and a field-level diff, and reducing that to "node act failed" throws
                    # away the one thing the operator needs to answer "what changed?".
                    ok = False
                    error = f"node {node.name} failed: {branch.reason or 'no reason recorded'}"
                    break

                drained, updated = await self._retire(branch, node_id, committed, reducers)
                if not drained:
                    ok = False
                    error = _not_retired(node.name, branch)
                if updated is not None:
                    committed = updated
                cursor = branch.cursor
                self._committed, self._cursor = committed, cursor
                steps += 1
                if not ok:
                    break
                if is_barrier(decision) and self.graph.capabilities().drives_itself:
                    break
        except Exception as exc:  # a run fault, journaled rather than swallowed
            ok, error = False, f"{type(exc).__name__}: {exc}"
        except BaseException:
            self._stop_node_tasks()
            raise

        await self._close_gate_if_spent(steps)
        await self._journal_run_finished(ok, error, steps, committed)
        return RunResult(
            run_id=run_id,
            ok=ok,
            state=committed.to_dict(),
            ledger=build_ledger(self.journal, run_id),
            steps=steps,
            error=error,
        )

    async def resume(self, run_id: str) -> RunResult:
        """Continue a run that was interrupted, without re-sending what already went out.

        Nothing is replayed and nothing is re-decided: committed state is rebuilt from the
        deltas of branches the journal records as RETIRED, the step counter continues above
        the highest position those branches consumed, and the graph is driven on from there.

        The dedupe table is what makes it safe rather than merely possible. An effect that was
        acked before the crash is claimed and skipped; one whose request demonstrably never
        left the dead process is re-sent; and one that may or may not have taken effect is
        dead-lettered unless the tool declared a repeat harmless. That last case is the
        two-generals boundary, and guessing at it is how a card gets charged twice.

        Only the retired chain contributes. A branch that was confirmed but never retired had
        its drain in flight when the process died; its state delta is not applied and its
        cursor is not adopted, because resuming from it would dispatch effects that were never
        context-checked or witness-validated.

        Its node runs again, and asks the model what it asked before. The answer is served
        from the journal when it is there and the question is identical (``RecordedTurns``),
        so the node decides again what it decided the first time -- which is what lets the
        dedupe table recognise any effect of that decision that already went out.
        """
        recovery = recover(self.journal, run_id)
        if not recovery.exists:
            # A run id the journal has never seen. Driving the graph from empty state here
            # dispatches every write the workload contains, under brand-new idempotency keys
            # that the dedupe table cannot match against anything -- so a typo in a run id
            # sends real writes and calls it a resume.
            raise SchedulerError(
                f"run {run_id!r} has no entries in this journal, so there is nothing to "
                "resume. Check the run id with `specunode runs`; resuming an unknown run "
                "would start a fresh one and dispatch its writes."
            )
        self.run_id = run_id
        self.buffer.run_id = run_id
        self.buffer.scheduler_task = asyncio.current_task()
        recorded = RecordedTurns(self.journal, run_id)
        if isinstance(self.target, JournaledModel):
            self.target.serve_recorded(recorded)
        self._committed = CommittedState(recovery.state)
        self._cursor = recovery.cursor
        self._open_group = recovery.open_group
        self._reducers = resolve_reducers(dict(self.reducers))
        self._steps = 0

        await self.journal.append_async(
            run_id,
            "run_started",
            {
                "v": 1,
                "mode": "resume",
                "resumed_from_offset": recovery.last_offset,
                "config_hash": chash({"reducers": dict(self.reducers)}),
                "registry_hash": chash(sorted(self.registry.names())),
                "policy": self._policy_payload(),
                "graph": {"adapter": self.graph.capabilities().framework},
                "target": {"provider": "configured", "model": "configured"},
                "recovered": {
                    "retired_branches": len(recovery.retired_branches),
                    "confirmed_not_retired": list(recovery.confirmed_not_retired),
                    "unresolved_dispatches": len(recovery.unresolved_dispatches),
                    "step_index": recovery.step_index,
                    # Turns the journal holds; a resumed node asking one again is served it.
                    "recorded_turns": recorded.recorded,
                    "served": isinstance(self.target, JournaledModel),
                },
            },
        )
        return await self._drive_from_state(run_id)

    async def _run_driven(self, run_id: str, inputs: JsonValue) -> RunResult:
        """The framework owns the loop; the runtime is reached from inside each node.

        LangGraph decides its own next node inside Pregel, and driving it node by node was
        tried and rejected: a node body that calls ``get_config()``, ``interrupt()`` or
        ``get_stream_writer()`` raises the moment it runs outside a runnable context, and
        re-deriving routing, reducers and map-reduce outside Pregel would break the one thing
        task 2.3 checks -- that a wrapped graph reaches the same final state as an unwrapped one.
        """
        ok, error = True, None
        session = RunSession(
            run_id=run_id,
            call_tool=self._orphan_tool_call,
            decide=self._decide,
            run_in_node=self._run_in_node,
            model=self.target,
            state=self._committed.fork(),
        )
        final: JsonValue = None
        try:
            with session_scope(session):
                final = await self.graph.drive(session, inputs)
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {exc}"
        except BaseException:
            self._stop_node_tasks()
            raise

        await self._close_gate_if_spent(self._steps)
        await self._journal_run_finished(ok, error, self._steps, self._committed)
        state = dict(final) if isinstance(final, Mapping) else self._committed.to_dict()
        return RunResult(
            run_id=run_id,
            ok=ok,
            state=state,
            ledger=build_ledger(self.journal, run_id),
            steps=self._steps,
            error=error,
        )

    async def _orphan_tool_call(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        """A routed tool called outside any node scope.

        Refused rather than executed. A call the runtime cannot attribute to a branch cannot be
        staged, cannot be retired and cannot appear in the ledger, so letting it through would
        put an effect in the world that no decision authorised.
        """
        raise SchedulerError(
            f"{name} was called outside a node. The runtime attributes every effect to the "
            "branch that issued it, and a call it cannot attribute cannot be made safe."
        )

    async def _run_in_node(
        self, node_name: str, body: Callable[[], Awaitable[JsonValue]]
    ) -> JsonValue:
        """Run one framework node under a branch of its own, and retire it."""
        self._cursor, node_id = self._cursor.visit(node_name)
        branch = self._fork_canonical(self._cursor, node_id)
        await self._journal_fork(branch, node_id)
        branch.state = self._committed.fork()

        tools = BranchTools(self, branch, node_id)
        session = RunSession(
            run_id=self.run_id,
            call_tool=tools.call,
            call_turn=self._turn_runner(branch, node_id),
            decide=self._decide,
            run_in_node=self._run_in_node,
            model=self.target,
            state=branch.state,
        )
        scope = CallScope(
            run_id=self.run_id,
            branch_id=branch.id,
            lineage=branch.lineage,
            step=branch.cursor.step_index,
            node_id=node_id,
            record_prompt=branch.record_prompt,
            # Whether this request is being sent on a guess. Always False before, which made
            # every ``model_request`` entry claim it was authorised work -- and a field that
            # never varies reads as a check while recording nothing. "On a guess" is the same
            # test ``Branch.record_prompt`` uses: this branch exists because something
            # predicted a decision, not merely that it has yet to be confirmed.
            speculative=branch.predicted is not None,
        )

        async def run() -> JsonValue:
            token = call_scope.set(scope)
            with session_scope(session):
                try:
                    return await body()
                finally:
                    call_scope.reset(token)

        task: asyncio.Task[JsonValue] = asyncio.create_task(run())
        branch.task = task
        outcome = await self._quiesce(branch, task)  # type: ignore[arg-type]
        if outcome is BranchOutcome.FAULTED:
            # The scheduler-driven loop returns here too. This path used to throw the outcome
            # away and retire regardless, so a node that raised had its branch squashed by
            # ``_quiesce`` and then confirmed by ``_retire`` -- and its staged write dispatched
            # on the strength of that forged status. ``Branch.confirm`` now refuses as well;
            # both halves are kept, because one of them is the guard and the other is not
            # asking it a question it should never be asked.
            await self._journal_faulted(branch, node_id)
            raise SchedulerError(f"node {node_id} failed: {branch.reason or 'no reason recorded'}")
        # State on this path belongs to the framework's checkpointer, so no delta is journaled
        # and none is passed here.
        #
        # The verdict is checked, not discarded. ``_retire`` returns False when it refuses --
        # today that is lattice rule E3 finding a witnessed read went stale -- and the
        # scheduler-driven loop has always consumed that flag. This path threw it away, so on
        # the LangGraph adapter a branch whose read had gone stale was squashed, its writes
        # discarded, and the run then walked on to the next node and dispatched *its* writes,
        # returning ok=True with a clean ledger. The twin of this bug, ``_quiesce``'s outcome
        # being discarded, was fixed three lines above and this one was left.
        drained, _state = await self._retire(branch, node_id)
        if not drained:
            raise SchedulerError(
                f"node {node_id} did not retire: {branch.reason or 'no reason recorded'}"
            )
        self._cursor = branch.cursor
        self._steps += 1
        # LangGraph owns reducers and channel semantics, and a second copy in the journal would
        # be a second answer to what the run's state is. docs/replay.md says so.
        return task.result()

    # -- the pieces -----------------------------------------------------------------------------

    async def _run_group(
        self,
        group: Parallel,
        cursor: StepCursor,
        committed: CommittedState,
        reducers: Mapping[str, Reducer],
    ) -> _GroupOutcome:
        """Run independent nodes side by side, and retire them one at a time in declared order.

        Every lane forks from the same committed state and the same program position, and its
        node id -- ``name#visit``, minted here in declared order -- is what keeps two lanes'
        idempotency keys apart. So the keys, the positions and the journal order are the same
        whether the bodies overlap or not (``Policy.parallel_nodes``). Retiring in declared
        order is what makes their effects reach the world in a reproducible order, however the
        scheduler interleaved them.

        The group is one routing decision, and it is journaled whole before any lane forks. A
        crash in the middle must not re-decide it: resumed, the router would see the state some
        lanes had already committed and could name something else, or the same lanes under new
        visit counts -- new keys, and effects that already went out going out again. A resume
        finishes the journaled group instead (:meth:`_resume_group`).

        A lane parked on a staged write stays parked until its turn to retire, so bodies
        overlap up to each one's first write and no further. And nodes named together must be
        independent: a lane that read what an earlier lane then changed is caught by the
        retirement re-check if the read was witnessed, and refused -- not re-run.
        """
        lanes: list[tuple[NodeRef, str]] = []
        visited = cursor
        for node in group.nodes:
            visited, node_id = visited.visit(node.structural_id)
            lanes.append((node, node_id))
        group_id = new_ulid()
        await self.journal.append_async(
            self.run_id,
            "group_forked",
            {
                "v": 1,
                "group_id": group_id,
                "lanes": [
                    {"name": node.name, "path": list(node.path), "node_id": node_id}
                    for node, node_id in lanes
                ],
                "fork_cursor": _cursor_payload(visited),
            },
        )
        return await self._run_lanes(group_id, lanes, visited, committed, committed, reducers)

    async def _resume_group(
        self, group: OpenGroup, committed: CommittedState, reducers: Mapping[str, Reducer]
    ) -> _GroupOutcome:
        """Finish a group a crash interrupted, as the group it was.

        Only its unretired lanes run, each under its own node id, from the position and the
        committed state the whole group forked from -- which is what they saw the first time,
        and what their keys were derived from. A lane whose writes went out before the crash
        derives the same keys again, and the dedupe table claims them rather than sending them
        twice. The lanes that retired are not re-run; the state keys they wrote stay claimed.
        """
        lanes = [
            (NodeRef(name=name, path=path), node_id)
            for name, path, node_id in group.lanes
            if node_id not in group.retired
        ]
        return await self._run_lanes(
            group.group_id,
            lanes,
            group.fork_cursor,
            CommittedState(dict(group.base_state)),
            committed,
            reducers,
            prior_step=group.max_step,
            claimed=dict(group.claimed),
        )

    async def _run_lanes(
        self,
        group_id: str,
        lanes: Sequence[tuple[NodeRef, str]],
        fork_cursor: StepCursor,
        base: CommittedState,
        committed: CommittedState,
        reducers: Mapping[str, Reducer],
        *,
        prior_step: int | None = None,
        claimed: dict[str, str] | None = None,
    ) -> _GroupOutcome:
        """Fork ``lanes`` from ``base`` at ``fork_cursor``, run them, retire them in order."""
        branches = [self._fork_canonical(fork_cursor, node_id) for _node, node_id in lanes]
        for (_node, node_id), branch in zip(lanes, branches, strict=True):
            await self._journal_fork(branch, node_id, group_id=group_id)

        async def start(index: int) -> tuple[BranchOutcome, Decision]:
            node, node_id = lanes[index]
            return await self._run_node(node, branches[index], node_id, base)

        if self.policy.parallel_nodes:
            outcomes = list(await asyncio.gather(*(start(i) for i in range(len(lanes)))))
        else:
            outcomes = [await start(i) for i in range(len(lanes))]

        def cursor_after() -> StepCursor:
            # Past every position any lane consumed, with every lane's visit counted.
            steps = [b.cursor.step_index for b in branches]
            if prior_step is not None:
                steps.append(prior_step)
            return replace(fork_cursor, step_index=max(steps))

        faulted = next(
            (i for i, (outcome, _) in enumerate(outcomes) if outcome is BranchOutcome.FAULTED),
            None,
        )
        if faulted is not None:
            name = lanes[faulted][0].name
            error = f"node {name} failed: {branches[faulted].reason or 'no reason recorded'}"
            await self._abandon_lanes(lanes, branches, error)
            return _GroupOutcome(False, error, committed, cursor_after(), 0)

        # Every lane is at rest -- finished, or parked on a staged write only its retirement
        # releases -- and nothing any of them staged has left. A clash in what they have written
        # so far is refused now, with nothing sent. A lane parked on its own write can still
        # write state after that write returns, which no check can see before it goes out; that
        # is caught at the lane's commit, and the lanes after it are abandoned unsent.
        claimed = dict(claimed or {})
        conflict = _first_conflict(lanes, branches, reducers, claimed)
        if conflict is not None:
            await self._abandon_lanes(lanes, branches, str(conflict))
            return _GroupOutcome(False, str(conflict), committed, cursor_after(), 0)

        # Every lane's witnessed reads are re-checked now, together, rather than one lane at a
        # time as each retires: lanes retire in order, so the re-checks used to queue behind one
        # another, one tool latency per lane, and a fan-out of read-only lanes paid for all of
        # them in series. A lane is at rest, so the reads checked here are the ones its
        # retirement would check. Only one thing can make a verdict out of date before the lane
        # retires: an effect an earlier lane in this group sends -- a lane that read what a
        # sibling then changed -- and a lane after one of those is checked again at its own
        # retirement, as before.
        checked = list(
            await asyncio.gather(*(validate_reads(branch, self.registry) for branch in branches))
        )

        for index, ((node, node_id), branch) in enumerate(zip(lanes, branches, strict=True)):
            rest = (lanes[index + 1 :], branches[index + 1 :])
            # Again before this lane's writes leave: a lane retired above may have committed,
            # after its own write returned, a key this one had already written.
            conflict = _first_conflict([lanes[index]], [branch], reducers, claimed)
            if conflict is not None:
                await self._abandon_lanes(lanes[index:], branches[index:], str(conflict))
                return _GroupOutcome(False, str(conflict), committed, cursor_after(), index)
            # The last lane to retire journals the group's cursor, not its own: it is where the
            # run goes on from, and the one a resume after the next node's crash must restore.
            # Its own would put that node at a lower step, under a key the dedupe table has
            # never seen, and send what it had already sent.
            last = index == len(lanes) - 1
            sent_before = sum(self.buffer.delivered(earlier.id) for earlier in branches[:index])
            try:
                drained, updated = await self._retire(
                    branch,
                    node_id,
                    committed,
                    reducers,
                    claimed=claimed,
                    cursor_after=cursor_after if last else None,
                    validation=checked[index] if sent_before == 0 else None,
                )
            except Exception as exc:
                # Raised after the lane was confirmed, so after its writes may have gone out: a
                # state clash or a reducer refusing at commit, or a write that dead-lettered.
                # Catching only the clash let the others escape with the lanes after this one
                # still parked and their forks never resolved.
                error = await self._close_failed_lane(node, node_id, branch, exc)
                await self._abandon_lanes(*rest, error)
                return _GroupOutcome(False, error, committed, cursor_after(), index)
            if updated is not None:
                committed = updated
            if not drained:
                error = _not_retired(node.name, branch)
                await self._abandon_lanes(*rest, error)
                return _GroupOutcome(False, error, committed, cursor_after(), index + 1)
        return _GroupOutcome(True, None, committed, cursor_after(), len(lanes))

    async def _close_failed_lane(
        self, node: NodeRef, node_id: str, branch: Branch, exc: Exception
    ) -> str:
        """A lane whose retirement raised: stop it, close its lifecycle, say what went out.

        It was confirmed, so it is not abandoned -- what it sent was authorised -- but it did
        not retire either. Without a resolution it read as confirmed forever, which recovery
        takes to mean a drain that was in flight when the process died. And the operator is
        told how many of its effects are in the world, because a failure that does not say so
        sends them looking for an effect that never happened, or away from one that did.
        """
        task = branch.task
        if isinstance(task, asyncio.Task) and not task.done():
            self.buffer.close(branch)
            task.cancel()
            await asyncio.wait({task})
        if isinstance(task, asyncio.Task) and task.done() and not task.cancelled():
            task.exception()
        cause = str(exc) if isinstance(exc, SchedulerError) else f"{type(exc).__name__}: {exc}"
        sent = self.buffer.delivered(branch.id)
        reason = f"{cause} ({sent} of its effects had already been dispatched)" if sent else cause
        await self.journal.append_async(
            self.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": branch.id,
                "step": branch.fork_step,
                "status": "faulted",
                "reason": reason,
                "node_id": node_id,
            },
        )
        return f"node {node.name} did not retire: {reason}"

    async def _abandon_lanes(
        self, lanes: Sequence[tuple[NodeRef, str]], branches: Sequence[Branch], reason: str
    ) -> None:
        """Close lanes that will not retire: revoke them, stop them, journal it.

        None of them has retired, so none of their writes has left -- which is the whole of
        what makes abandoning them safe. Each still ends in exactly one of the ways a branch
        ends, in the durable record, so no reader finds a fork without a resolution.

        Revoke first, cancel second, as a squashed speculation is. A lane parked on its own
        write whose body has a ``finally`` that writes -- give a lease back -- used to be
        cancelled while its buffer was still open: the ``finally`` staged a write nothing
        would drain, parked on its ack, and the wait for it never returned. A revoked branch's
        next write is refused at once. The wait does not swallow a cancellation of the run
        itself either; it used to, so a run cancelled from outside finished as if it had not
        been.
        """
        owns: list[str | None] = []
        for branch in branches:
            owns.append(branch.reason)
            if branch.status is not BranchStatus.SQUASHED:
                branch.squash(f"abandoned: {reason}")
            discarded = await self.buffer.discard_and_journal(branch, f"abandoned: {reason}")
            self.counters.effects_discarded += discarded
        running = [
            branch.task
            for branch in branches
            if isinstance(branch.task, asyncio.Task) and not branch.task.done()
        ]
        for task in running:
            task.cancel()
        if running:
            await asyncio.wait(running)
        for task in running:
            if not task.cancelled():
                task.exception()  # retrieved: it is reported below, not by the event loop
        for (_node, node_id), branch, own in zip(lanes, branches, owns, strict=True):
            await self.journal.append_async(
                self.run_id,
                "branch_resolved",
                {
                    "v": 1,
                    "branch_id": branch.id,
                    "step": branch.fork_step,
                    "status": "faulted" if own else "squashed",
                    "reason": own or f"abandoned: {reason}",
                    "node_id": node_id,
                },
            )

    def _fork_canonical(self, cursor: StepCursor, node_id: str) -> Branch:
        branch = Branch(
            id=new_ulid(),
            fork_step=cursor.step_index,
            predicted=None,
            status=BranchStatus.SPECULATIVE,
            cursor=cursor,
            node_id=node_id,
        )
        self.counters.branches_forked += 1
        return branch

    async def _journal_fork(
        self, branch: Branch, node_id: str, *, group_id: str | None = None
    ) -> None:
        await self.journal.append_async(
            self.run_id,
            "branch_forked",
            {
                "v": 1,
                **({"group_id": group_id} if group_id is not None else {}),
                "branch_id": branch.id,
                "parent_id": branch.parent_id,
                "lineage": list(branch.lineage),
                "fork_step": branch.fork_step,
                "node_id": node_id,
                "depth": branch.depth,
                # The canonical branch predicts nothing: it is confirmed by the model's own
                # output rather than resolved against a guess, and these three stay null for
                # it. A branch forked *on a prediction* records what was predicted, so the
                # journal can tell the two apart -- without this, every fork looks canonical
                # on replay and "how much did this run actually speculate" is unanswerable
                # from the durable record, which is the only record Hard Rule 12 permits an
                # answer to come from.
                "predicted": decision_payload(branch.predicted) if branch.predicted else None,
                "predicted_hash": decision_key(branch.predicted) if branch.predicted else "",
                "tier": branch.tier,
            },
        )

    async def _run_node(
        self, node: NodeRef, branch: Branch, node_id: str, committed: CommittedState
    ) -> tuple[BranchOutcome, Decision]:
        """Run a node body to rest: finished, or parked on a value only the drain supplies."""
        tools = BranchTools(self, branch, node_id)
        branch.state = committed.fork()
        session = RunSession(
            run_id=self.run_id,
            call_tool=tools.call,
            call_turn=self._turn_runner(branch, node_id),
            decide=self._decide,
            model=self.target,
            state=branch.state,
        )
        scope = CallScope(
            run_id=self.run_id,
            branch_id=branch.id,
            lineage=branch.lineage,
            step=branch.cursor.step_index,
            node_id=node_id,
            record_prompt=branch.record_prompt,
            # Whether this request is being sent on a guess. Always False before, which made
            # every ``model_request`` entry claim it was authorised work -- and a field that
            # never varies reads as a check while recording nothing. "On a guess" is the same
            # test ``Branch.record_prompt`` uses: this branch exists because something
            # predicted a decision, not merely that it has yet to be confirmed.
            speculative=branch.predicted is not None,
        )

        async def body() -> Decision:
            token = call_scope.set(scope)
            # The session goes into a ContextVar too, so a tool wrapped by `routed` reaches
            # the runtime from inside a framework node whose signature has nowhere to pass one.
            with session_scope(session):
                try:
                    return await self.graph.run_node(node, session)
                finally:
                    call_scope.reset(token)

        task: asyncio.Task[Decision] = asyncio.create_task(body())
        branch.task = task
        self._node_tasks[task] = branch
        task.add_done_callback(self._forget_node_task)
        outcome = await self._quiesce(branch, task)
        if outcome is BranchOutcome.FAULTED:
            return outcome, ToolCall("", {})
        if outcome is BranchOutcome.DONE:
            return outcome, task.result()
        # PARKED: the decision is not available until the drain releases the node body.
        return outcome, ToolCall("", {})

    def _forget_node_task(self, task: asyncio.Task[Decision]) -> None:
        self._node_tasks.pop(task, None)

    def _stop_node_tasks(self) -> None:
        """End every node body a run left running when it ended by an uncaught exception.

        Its caller cancelled it -- a timeout, a shutdown -- and its node bodies went on without
        it: reading upstream and staging writes for a run that was over. Each body's branch is
        closed first, so a ``finally`` that writes is refused rather than parked on an ack no
        drain will ever complete, and then it is cancelled. Nothing is journaled: a run ended
        this way is resumed, as a crashed one is, and what is durable is what it had written.
        """
        for task, branch in list(self._node_tasks.items()):
            if task.done():
                continue
            self.buffer.close(branch)
            task.cancel()

    async def _quiesce(self, branch: Branch, task: asyncio.Task[Decision]) -> BranchOutcome:
        """Wait until the branch's task finishes, or parks on a value only the drain supplies.

        Never simply awaits the task. A node parked on a staged write's result is waiting for
        something this scheduler has not done yet, so awaiting it is the deadlock this design
        exists to prevent -- and it bites on the first write of the first sequential run,
        before any speculation is involved.
        """
        waiter = asyncio.ensure_future(self._park_event(branch.id).wait())
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()

        if task.done():
            if task.cancelled():
                return BranchOutcome.FAULTED
            exc = task.exception()
            if exc is not None:
                # A branch that has already been confirmed is NOT squashed here, and the
                # reason is Hard Rule 3 read in the other direction. ``_retire`` confirms, then
                # drains, then calls this again to let the node run on -- so by the second call
                # the branch may hold effects that are already in the world. Squashing it then
                # marks a branch that changed the world as one that never retired, which is the
                # invariant the leak test asserts; the next drain refuses with ``BranchClosed``
                # and the node's real exception is replaced by a fabricated one, so the operator
                # is told the wrong thing about a run that already had side effects.
                #
                # The failure is still a failure: FAULTED is returned either way, ``_retire``
                # stops, and the run reports the node's own error. Only the status lie is
                # removed. This is the mirror image of the ``_timed_read`` status race -- that
                # one could promote a squashed branch, this one demotes a confirmed one.
                if branch.status is not BranchStatus.CONFIRMED:
                    branch.squash(f"{type(exc).__name__}: {exc}")
                else:
                    branch.reason = f"{type(exc).__name__}: {exc}"
                return BranchOutcome.FAULTED
            return BranchOutcome.DONE
        return BranchOutcome.PARKED

    async def _journal_faulted(self, branch: Branch, node_id: str) -> None:
        """Close a faulted branch's lifecycle in the durable record."""
        await self.journal.append_async(
            self.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": branch.id,
                "step": branch.fork_step,
                "status": "faulted",
                "reason": branch.reason or f"node {node_id} raised",
            },
        )

    def _verify_context(self, branch: Branch) -> int:
        """Hard Rule 13 at retirement. Returns how many prompts were actually re-checked.

        This used to be ``branch.context_verified = True`` with a hard-coded zero, which made
        the store buffer's Rule 13 gate unreachable -- both drain call sites are below it. The
        ledger stamp stayed honest (it reads ``unchecked`` at zero checks), but a branch that
        *had* sent a request while guessing would have been marked verified without anything
        looking at it, and its writes would have drained on the strength of that stamp.

        **A branch that sent no speculative request has nothing to rebuild.** Saying so is not
        the same as claiming a check, and it is the case every run takes today: a speculative
        child runs a single tool call and never opens a turn of its own.

        **A branch that did send one is refused.** Not rebuilt-and-compared, because that
        comparison is not implementable correctly here yet: ``fold_context`` reconstructs the
        *message list*, while the recorded ``request_hash`` covers the whole projected envelope
        -- system blocks, tool declarations and all -- so hashing one against the other would
        never match, and the repair that suggests itself (rebuild from the branch's own message
        list) compares that list to itself and passes every time. That is by far the most
        likely way to ship Rule 13 dead.

        So this fails closed. The drain refuses, the run stops, and nothing downstream of a
        request nobody verified reaches the world. A refusal on a path no shipped configuration
        reaches costs nothing today and is the correct answer the day one does.
        """
        if not branch.speculative_prompts:
            branch.context_verified = True
            return 0

        branch.context_verified = False
        branch.reason = (
            f"branch {branch.id} sent {branch.speculative_prompts} request(s) while "
            "speculating, and the retirement-time rebuild is not implemented. Refusing rather "
            "than stamping a check nobody performed (Hard Rule 13)."
        )
        self.counters.context_divergences += 1
        return 0

    async def _retire(
        self,
        branch: Branch,
        node_id: str,
        committed: CommittedState | None = None,
        reducers: Mapping[str, Reducer] | None = None,
        claimed: dict[str, str] | None = None,
        cursor_after: Callable[[], StepCursor] | None = None,
        validation: ReadValidation | None = None,
    ) -> tuple[bool, CommittedState | None]:
        """R5 through R9: validate reads, confirm, make it durable, drain, then retire."""
        # E3, before anything is confirmed. ``validate_reads`` is the retirement-time witness
        # re-check the spec calls "the ordering that carries most of the integrity", and it had
        # no caller anywhere in ``src/`` -- only a unit test and an attack script, both of which
        # hand-build a Branch and call it directly. So a branch whose witnessed speculative
        # reads had gone stale retired anyway and drained its writes; ``policy.on_stale_read``
        # was dead config that was nonetheless journaled into ``run_started`` and rendered as
        # though it applied; and the ledger printed "reads validated at retirement: 0/0 fresh"
        # on every run, which is the difference between a receipt and a reassurance.
        if validation is None:
            validation = await validate_reads(branch, self.registry)
        if validation.verdicts:
            await self.journal.append_async(
                self.run_id,
                "read_validated",
                {
                    "v": 1,
                    "branch_id": branch.id,
                    "step": branch.cursor.step_index,
                    "fresh": validation.fresh,
                    "stale": validation.stale,
                    "unwitnessed": validation.unwitnessed,
                    "unreadable": validation.unreadable,
                    "total": len(validation.verdicts),
                    "probes": validation.probes,
                },
            )
            self.counters.reads_validated += validation.fresh
            self.counters.reads_stale += validation.stale
        refuse = validation.stale and self.policy.on_stale_read == "squash"
        # A read whose re-check *failed* is a different fact from one that went stale, and it
        # gets its own policy. It is never silently treated as fresh -- the count is journaled
        # and rendered either way -- but refusing on it by default pre-empts the drain, whose
        # failure handling dead-letters the effect by name and is strictly more informative
        # when the upstream is simply unreachable.
        refuse = refuse or (validation.unreadable and self.policy.on_unverifiable_read == "squash")
        if refuse:
            # The branch computed its arguments from a value the world may no longer agree
            # with. Its writes are not authorised by anything that has been confirmed, so it
            # does not retire and the store buffer discards what it staged, unsent.
            detail = (
                f"{validation.stale} stale, {validation.unreadable} unreadable"
                if validation.unreadable
                else f"{validation.stale} witnessed read(s) went stale"
            )
            branch.squash(f"{detail} before retirement")
            discarded = await self.buffer.discard_and_journal(branch, "stale_read")
            self.counters.effects_discarded += discarded
            await self.journal.append_async(
                self.run_id,
                "branch_resolved",
                {
                    "v": 1,
                    "branch_id": branch.id,
                    "step": branch.fork_step,
                    "status": "squashed",
                    "reason": branch.reason or "stale read",
                },
            )
            return False, committed

        branch.confirm()
        checked = self._verify_context(branch)
        confirmed_offset = await self.journal.append_async(
            self.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": branch.id,
                "step": branch.fork_step,
                "status": "confirmed",
                "context_verified": branch.context_verified,
                "context_checks": [checked, len(branch.prompts_sent)],
            },
        )
        # Drain, let the node make progress, drain again. A node released by one drain can
        # stage the next write from the value it just received, so a single pass would leave
        # that effect staged and never dispatched -- an authorised write dropped silently.
        task = branch.task
        ok = True
        undrained: tuple[str, ...] = ()
        while True:
            report = await self.buffer.drain(
                branch,
                self.dispatcher,
                confirmed_offset=confirmed_offset,
                authorised_by_offset=confirmed_offset,
            )
            ok = ok and report.ok
            undrained = report.undrained
            self.counters.effects_dispatched += report.count(EffectOutcome.DISPATCHED)
            self.counters.effects_dead_lettered += report.count(EffectOutcome.DEAD_LETTER)
            if not isinstance(task, asyncio.Task) or task.done() or not report.ok:
                break
            self._park_event(branch.id).clear()
            outcome = await self._quiesce(branch, task)
            if outcome is not BranchOutcome.PARKED:
                # The node finished. Anything it staged on the way out is picked up by one
                # more pass, which then finds nothing new and stops.
                final = await self.buffer.drain(
                    branch,
                    self.dispatcher,
                    confirmed_offset=confirmed_offset,
                    authorised_by_offset=confirmed_offset,
                )
                ok = ok and final.ok
                undrained = final.undrained
                self.counters.effects_dispatched += final.count(EffectOutcome.DISPATCHED)
                self.counters.effects_dead_lettered += final.count(EffectOutcome.DEAD_LETTER)
                break

        if isinstance(task, asyncio.Task) and not task.done():
            # Refuse its next write before cancelling it, and wait for it to stop: a node left
            # running here outlived the run, and one whose ``finally`` writes would park on an
            # ack nothing completes.
            self.buffer.close(branch)
            task.cancel()
            await asyncio.wait({task})
            if not task.cancelled():
                task.exception()
            if not ok:
                raise SchedulerError(
                    f"a write from node {node_id} was dead-lettered, so the run halts here "
                    "for a human; a resume retries it under the same key"
                )
            raise SchedulerError(
                f"node {node_id} did not finish after its writes were dispatched; it is "
                "waiting on something the runtime never completes"
            )
        if isinstance(task, asyncio.Task) and task.done() and task.exception() is not None:
            # The node failed after its writes were authorised and dispatched. The branch is
            # not squashed -- its effects were legitimate and are in the world -- but it never
            # reaches RETIRED either, so without this its lifecycle would simply stop in the
            # durable record. That is the same open-ended state a process death after dispatch
            # leaves, and ``Recovery`` already treats it as evidence rather than input; saying
            # so explicitly is what lets an auditor tell "confirmed, dispatched, then the node
            # failed" apart from "we have no idea what happened here".
            await self.journal.append_async(
                self.run_id,
                "branch_resolved",
                {
                    "v": 1,
                    "branch_id": branch.id,
                    "step": branch.fork_step,
                    "status": "faulted",
                    "reason": branch.reason or "node raised after dispatch",
                },
            )
            return False, committed
        if undrained:
            raise SchedulerError(
                f"branch {branch.id} staged {len(undrained)} effect(s) that were never "
                "dispatched; an authorised write cannot be silently dropped"
            )
        # The state delta is made durable BEFORE the entry that says this branch retired.
        # The other order loses a crash window with teeth: if the process dies between them,
        # the branch reads as retired while its state change is gone, so a resume re-runs the
        # node -- from a *different* program position, deriving different idempotency keys, and
        # dispatching effects that already went out. It looks like a resume bug and is an
        # ordering bug.
        new_committed = committed
        if committed is not None:
            new_committed = await self._commit(branch, committed, reducers or {}, claimed)

        branch.retire()
        self._retire_seq += 1
        self.counters.branches_retired += 1
        await self.journal.append_async(
            self.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": branch.id,
                "step": branch.fork_step,
                "status": "retired",
                "retire_seq": self._retire_seq,
                # The exact program position this retirement committed. A resume restores it
                # verbatim rather than inferring it from the highest step it can see: inferring
                # lands the resumed run at a different position, so every key it derives differs
                # from the pre-crash one, the dedupe table misses, and the effects go out twice.
                "cursor_after": _cursor_payload(
                    cursor_after() if cursor_after is not None else branch.cursor
                ),
            },
        )
        return ok, new_committed

    async def _commit(
        self,
        branch: Branch,
        committed: CommittedState,
        reducers: Mapping[str, Reducer],
        claimed: dict[str, str] | None = None,
    ) -> CommittedState:
        """Apply the retiring branch's delta, and journal what it did to committed state."""
        patch = branch.state.delta()
        if not patch:
            return committed
        touched: set[str] | None = None
        if claimed is not None:
            # A node in a Parallel group. Its delta was taken against the state the whole group
            # forked from, so applying it on top of a sibling's would silently overwrite any key
            # they both wrote -- the later one in declared order winning, with no one told. A
            # declared reducer is the developer saying how to combine them; without one, refuse.
            touched = _touched_keys(patch)
            for key in sorted(touched):
                owner = claimed.get(key)
                if owner is not None and owner != branch.node_id and key not in reducers:
                    raise ParallelWriteConflict(key, owner, branch.node_id)
            for key in touched:
                claimed.setdefault(key, branch.node_id)
            result = committed.commit_values(
                {key: branch.state[key] for key in touched if key in branch.state},
                [key for key in touched if key not in branch.state],
                reducers=reducers,
            )
        else:
            result = committed.commit(patch, reducers=reducers)
        await self.journal.append_async(
            self.run_id,
            "state_delta_applied",
            {
                "v": 1,
                "branch_id": branch.id,
                "step": branch.fork_step,
                "patch": patch_payload(result.patch),
                "patch_hash": result.patch_hash,
                "base_state_hash": result.base_state_hash,
                "result_state_hash": result.result_state_hash,
                "reducers_applied": [
                    {"key": key, "reducer": name} for key, name in result.reducers_applied
                ],
                # What a lane in a group wrote, which a resume needs to keep claimed: the patch
                # above is the effective change, and a write of the value already committed
                # changes nothing while still being a write.
                **({"touched_keys": sorted(touched)} if touched is not None else {}),
            },
        )
        return result.state

    def _turn_runner(
        self, branch: Branch, node_id: str
    ) -> Callable[[object], Awaitable[Sequence[JsonValue]]]:
        """A node's handle on tier-0 early issue."""

        async def call_turn(envelope: object) -> Sequence[JsonValue]:
            if not isinstance(envelope, RequestEnvelope):
                raise SchedulerError("call_turn needs a RequestEnvelope")
            turn = SpeculativeTurn(self, branch, node_id)
            self._turns.append(turn)
            return await turn.run(envelope)

        return call_turn

    async def _decide(self, decision: Decision) -> Decision:
        """A node reporting its decision. In sequential mode it is simply itself."""
        return decision

    # -- journal bookends --------------------------------------------------------------------------

    async def _journal_alpha_observed(self, step: int, branch_id: str | None = None) -> None:
        """One ``policy_event`` per resolution, carrying the window's current alpha.

        The ledger reads alpha from ``policy_event`` payloads and from nowhere else, and none
        was ever written -- so every ledger ever produced printed ``alpha: n/a`` while the
        runtime was measuring it the whole time. One small entry per resolution is bounded by
        the speculation budget itself, and it gives the durable record the trajectory rather
        than a single number nobody can place in time.
        """
        window = self.budget.window
        await self.journal.append_async(
            self.run_id,
            "policy_event",
            {
                "v": 1,
                "event": "alpha_observed",
                "reason": "a prediction resolved",
                # The guess's own fork step and id, not the parent's cursor -- early reads
                # advance that concurrently, so the two entries for one resolution carried
                # different step numbers and could not be joined.
                "step": step,
                "branch_id": branch_id,
                "alpha": window.alpha,
                # The rate the gate consults is None until the window is full, on purpose --
                # disabling on three data points is a worse error than guessing three more
                # times. The *receipt* still has to say what was measured, so the counts
                # travel too and the ledger renders them when the rate is not yet judged.
                "hits": window.hits,
                "samples": window.samples,
                "window": window.size,
                # Per tier as well -- the tiers the configured predictor actually offered --
                # so the receipt can say which predictor the misses belong to. The gate never
                # reads this.
                "by_tier": {
                    str(tier): {"hits": hits, "samples": samples}
                    for tier, (hits, samples) in window.graded_by_tier().items()
                },
                "wasted_tokens": self.budget.wasted_tokens,
                "speculative_reads_used": self.budget.speculative_reads_used,
            },
        )

    async def _journal_speculation_disabled(self, step: int) -> None:
        """Record, once, that the gate closed and why. Idempotent across calls."""
        if self.budget.speculation_disabled:
            return
        reason = self.budget.should_disable() or "unknown"
        self.budget.disable(reason)
        self.counters.speculation_disabled_reason = reason
        await self.journal.append_async(
            self.run_id,
            "policy_event",
            {
                "v": 1,
                "event": "speculation_disabled",
                "reason": reason,
                "step": step,
                "alpha": self.budget.window.alpha,
                "hits": self.budget.window.hits,
                "samples": self.budget.window.samples,
                "window": self.budget.window.size,
                "by_tier": {
                    str(tier): {"hits": hits, "samples": samples}
                    for tier, (hits, samples) in self.budget.window.graded_by_tier().items()
                },
                "wasted_tokens": self.budget.wasted_tokens,
                "speculative_reads_used": self.budget.speculative_reads_used,
                "inflight_branches": self.budget.inflight_branches,
            },
        )

    def _policy_payload(self) -> dict[str, JsonValue]:
        """The whole policy, for whichever ``run_started`` is being written.

        One helper rather than two literals: ``resume`` journaled ``{"speculation": ...}`` alone,
        so a resumed run's ledger read its alpha window as absent and rendered "-", and the
        budgets that governed the run were missing from the only durable record of them.
        """
        return {
            "speculation": self.policy.speculation,
            "early_issue": self.policy.early_issue,
            "max_inflight_branches": self.policy.max_inflight_branches,
            "max_speculation_depth": self.policy.max_speculation_depth,
            "max_wasted_tokens": self.policy.max_wasted_tokens,
            "max_speculative_reads": self.policy.max_speculative_reads,
            # The ledger reads its window size from here; it read -1 and rendered "-".
            "alpha_window": self.policy.alpha_window,
            "alpha_floor": self.policy.alpha_floor,
            "speculate_writes": self.policy.speculate_writes,
            "stage_irreversible": self.policy.stage_irreversible,
            "on_stale_read": self.policy.on_stale_read,
            "on_unverifiable_read": self.policy.on_unverifiable_read,
        }

    async def _journal_run_started(self, inputs: JsonValue) -> None:
        capabilities = self.graph.capabilities()
        if self.policy.speculation and self.policy.alpha_floor is None:
            # The ledger already knows how to render this event; nothing ever emitted it.
            await self.journal.append_async(
                self.run_id,
                "policy_event",
                {
                    "v": 1,
                    "event": "alpha_floor_unmeasured",
                    "reason": (
                        "no break-even alpha has been measured for this workload, so the "
                        "alpha gate is inactive; the other budgets still apply"
                    ),
                    "step": 0,
                },
            )
        await self.journal.append_async(
            self.run_id,
            "run_started",
            {
                "v": 1,
                "mode": "run",
                "config_hash": chash({"reducers": dict(self.reducers)}),
                "registry_hash": chash(sorted(self.registry.names())),
                "policy": self._policy_payload(),
                "graph": {
                    "adapter": capabilities.framework,
                    "nodes": [n.structural_id for n in self.graph.nodes()],
                },
                "target": {"provider": "configured", "model": "configured"},
                "inputs": inputs,
            },
        )

    async def _close_gate_if_spent(self, step: int) -> None:
        """Journal a closure that nothing will get round to announcing.

        ``_journal_speculation_disabled`` ran only from ``_speculate_next``, so a budget spent
        by the last resolution of a run -- or by a read issued after the final block -- left
        ``may_speculate()`` False with no ``policy_event`` and no reason in the counters. The
        run stopped speculating and the durable record did not say why.
        """
        if self.budget.speculation_disabled or self.budget.should_disable() is None:
            return
        await self._journal_speculation_disabled(step)

    async def _journal_run_finished(
        self, ok: bool, error: str | None, steps: int, committed: CommittedState
    ) -> None:
        await self.journal.append_async(
            self.run_id,
            "run_finished",
            {
                "v": 1,
                "ok": ok,
                "status": "completed" if ok else "failed",
                "error": None if error is None else {"type": "RunFault", "message": error},
                "steps": steps,
                "final_state_hash": committed.hash,
                "counters": {
                    "branches_forked": self.counters.branches_forked,
                    "branches_retired": self.counters.branches_retired,
                    "branches_squashed": self.counters.branches_squashed,
                    "branches_stalled": self.counters.branches_stalled,
                    "effects_staged": self.counters.effects_staged,
                    "effects_dispatched": self.counters.effects_dispatched,
                    "effects_dead_lettered": self.counters.effects_dead_lettered,
                    "effects_discarded": self.counters.effects_discarded,
                    "speculative_reads_upstream": self.counters.speculative_reads_upstream,
                    "context_divergences": self.counters.context_divergences,
                    "wasted_tokens": self.counters.wasted_tokens,
                    "stalls_by_hazard": dict(self.counters.stalls_by_hazard),
                    # Why the gate closed, or None. The policy_event says it at the moment it
                    # happened; the summary says it where a reader looks first.
                    "speculation_disabled_reason": self.counters.speculation_disabled_reason,
                },
            },
        )


class SpeculativeTurn:
    """One target turn, with its tool calls issued as the stream emits them.

    This is tier-0 early issue, and it is the part of the design that produces wall-clock
    savings without predicting anything. The model streams its answer; a ``tool_use`` block
    becomes complete some time before the turn does; the runtime issues that call immediately
    rather than waiting for the end of the turn. On a turn that emits a read followed by two
    writes, the read can be finished before the model has stopped talking.

    Nothing is guessed, so nothing can be wrong -- these are calls the target has already
    emitted. What is speculated is *time*: the turn is not yet durable when the read goes out,
    so the read is marked speculative, counted in the ledger's upstream-read total, and
    re-validated at retirement if it carries a witness.

    Writes are staged, not run, exactly as anywhere else. They dispatch when the branch retires,
    which cannot happen before the turn's ``model_response`` entry is on disk.

    The pattern is Claude Code's streaming tool executor, credited as theirs.
    """

    def __init__(self, scheduler: Scheduler, branch: Branch, node_id: str) -> None:
        self._scheduler = scheduler
        self._branch = branch
        self._node_id = node_id
        self.decisions: list[ToolCall] = []
        #: When each block finished parsing and when its call finished, for task 3.1's
        #: timestamp assertion that a read completes before the stream ends.
        self.issued_at: list[float] = []
        self.completed_at: list[float] = []
        self.stream_ended_at: float = 0.0
        self.reads_issued_early = 0
        #: Speculations that the model's real decision confirmed, so their work was kept.
        self.adopted = 0
        self.confirmed = 0
        self.squashed = 0
        self.stalled = 0
        self._base = 0
        #: What the WRITE blocks emitted so far in this turn touch. They are not in the store
        #: buffer yet -- a turn stages its writes only after the stream ends, so that a staged
        #: effect never precedes a durable turn -- while reads are issued as their blocks parse.
        #: Without this, READ_AFTER_STAGED_WRITE could not fire for the one shape it exists to
        #: catch: a read that follows a write inside a single model turn.
        self._pending_write_keys: list[frozenset[str]] = []
        #: Slot tasks whose work was adopted from a confirmed speculation. They may be parked
        #: on an ack this branch has to drain, which an ordinary slot task never is.
        self._adopted_tasks: set[asyncio.Task[JsonValue]] = set()
        self._history: list[ToolCall] = []
        self._results_so_far: list[JsonValue] = []
        self._predicted: ToolCall | None = None
        self._speculative: Branch | None = None
        self._open: asyncio.Task[JsonValue] | None = None
        self._adopted: asyncio.Task[JsonValue] | None = None
        self._tier: int = 1
        #: What the open prediction cost to produce; charged to ``wasted_tokens`` on a squash.
        self._cost_tokens: int = 0
        #: The model's whole reply, from ``TurnComplete``, for a caller that continues the
        #: conversation.
        self.response: ModelResponse | None = None

    async def run(self, envelope: RequestEnvelope) -> list[JsonValue]:
        scheduler = self._scheduler
        branch = self._branch
        tools = BranchTools(scheduler, branch, self._node_id)
        # Program positions for this turn are reserved from here by ordinal, so block k always
        # occupies base + k + 1 whether it was issued early, staged after the stream, or run on
        # a speculation that was later adopted.
        self._base = branch.cursor.step_index

        # Slots are preallocated as blocks parse and filled by ordinal. Program order is
        # structural: a result never appends on completion, because the order the model asked
        # for its calls is the order it must be shown them in (Hard Rule 13).
        slots: list[asyncio.Task[JsonValue] | None] = []

        try:
            await self._consume(envelope, tools, slots)
        except BaseException:
            # A stream that raised, or a node task cancelled, with a guess still open: grade
            # it, discard what it staged, cancel its task and journal the resolution before
            # the exception continues. Without this the in-flight count leaked for the rest
            # of the scheduler's life, the guess was never graded, and the journal held a
            # ``branch_forked`` with no ``branch_resolved`` -- a branch that ended in none of
            # the three ways branch.py says every branch ends.
            await self._squash_open("turn_failed")
            await self._abandon(slots)
            raise

        # Any speculation still open when the turn ended predicted a call the model never made.
        await self._squash_open("turn_ended")
        try:
            results = await self._settle_turn(tools, slots)
        except BaseException:
            # Settling is where a node parks on its own staged write, so it is where a node is
            # cancelled when a sibling in its group fails, and where a write whose ack fails
            # raises. The reads this turn issued early for later blocks were still running
            # then: they reached upstream after the run returned, and journaled after
            # ``run_finished``. The stream's failure path already ended them; this one did not.
            await self._abandon(slots)
            raise
        return TurnResults(results, self.response)

    async def _abandon(self, slots: list[asyncio.Task[JsonValue] | None]) -> None:
        """Cancel and await every call this turn started, so none outlives the turn.

        A turn whose stream raised left its early-issued reads running. They reached upstream
        *after* the run had returned a failure to its caller, and each one appended a
        ``tool_result`` to the journal after ``run_finished`` -- which every reader of a run,
        recovery included, takes to be the run's last entry. A turn that failed is over, and
        so is everything it started.
        """
        pending = [task for task in slots if task is not None and not task.done()]
        if self._adopted is not None and not self._adopted.done():
            pending.append(self._adopted)
        self._adopted = None
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _consume(
        self,
        envelope: RequestEnvelope,
        tools: BranchTools,
        slots: list[asyncio.Task[JsonValue] | None],
    ) -> None:
        scheduler = self._scheduler
        async for event in scheduler.target.stream(envelope):
            if isinstance(event, ToolUseComplete):
                actual = ToolCall(name=event.block.name, args=event.block.args)
                await self._resolve_prediction(actual)

                self.decisions.append(actual)
                self.issued_at.append(time.monotonic())
                spec = scheduler.registry.get(actual.name)
                ordinal = len(slots)
                if spec.effect is not EffectClass.READ:
                    self._pending_write_keys.append(keys_touched(spec, actual.args))
                early = scheduler.policy.early_issue
                if spec.effect is EffectClass.READ and self._adopted is None and early:
                    self.reads_issued_early += 1
                    slots.append(asyncio.create_task(self._timed_read(tools, actual, ordinal)))
                elif self._adopted is not None:
                    # The speculation was right: its result is already in hand, so the call is
                    # not made twice. This is the latency the whole arrangement buys.
                    adopted = self._adopted
                    self._adopted = None
                    slots.append(adopted)
                    self.adopted += 1
                else:
                    slots.append(None)

                await self._speculate_next(actual)
            elif isinstance(event, TurnComplete):
                self.stream_ended_at = time.monotonic()
                self.response = event.response

    async def _settle_turn(
        self, tools: BranchTools, slots: list[asyncio.Task[JsonValue] | None]
    ) -> list[JsonValue]:
        scheduler = self._scheduler
        branch = self._branch
        # The turn is durable now (JournaledModel writes it before yielding TurnComplete), so
        # the writes it emitted may be staged. They are staged here rather than mid-stream so
        # that a staged effect never exists for a turn the journal does not yet record.
        results: list[JsonValue] = []
        for ordinal, emitted in enumerate(self.decisions):
            task = slots[ordinal]
            if task is not None:
                if task in self._adopted_tasks and not task.done():
                    # The only slot this branch cannot finish on its own: an adopted
                    # speculation parked on a staged write's ack, which only the drain
                    # completes. Signalled here rather than at adoption so that every write
                    # emitted before this block is already staged and the drain dispatches
                    # them in the order the model asked for.
                    scheduler.mark_parked(branch)
                results.append(await task)
                continue
            results.append(
                await tools.call(emitted.name, emitted.args, step=self._base + ordinal + 1)
            )
        # The turn consumed exactly one position per emitted block, however they were executed.
        # Set rather than accumulated: an adopted block was run on the child, so the parent's
        # cursor never passed through its position on the way here.
        branch.reserve_step(self._base + len(self.decisions))
        return results

    async def _speculate_next(self, after: ToolCall) -> None:
        """Guess the call after this one and start running it, on a branch of its own.

        This is the part that genuinely predicts, and therefore the part that can be wrong.
        A wrong guess costs the upstream reads its branch made and the tokens it spent; it
        cannot cost an effect, because its writes go into the store buffer and are discarded
        without ever being dispatched.
        """
        scheduler = self._scheduler
        if not scheduler.policy.speculation or scheduler.predictor is None:
            return
        if not scheduler.budget.may_speculate():
            # The gate has closed for this run. Say so, once, in the durable record. The gate
            # itself always worked -- ``may_speculate`` re-evaluates every time -- but
            # ``Budget.disable`` had no caller and no ``policy_event`` was ever journaled, so
            # Hard Rule 10's "speculation is disabled for that run by deterministic policy,
            # **and the ledger says so**" held for the first half and not the second. A run
            # that stopped speculating and a run that never started looked identical.
            await scheduler._journal_speculation_disabled(self._branch.cursor.step_index)
            return
        drafter = scheduler.predictor

        self._history.append(after)
        context = DraftContext(
            run_id=scheduler.run_id,
            branch_id=self._branch.id,
            step_index=self._branch.cursor.step_index,
            node_id=self._node_id,
            history=tuple(self._history),
            results=dict(enumerate(self._results_so_far)),
            known_tools=scheduler.registry.names(),
        )
        candidates = await drafter.predict(context)
        if not candidates:
            return

        prediction = candidates[0]
        decision = prediction.decision
        if not isinstance(decision, ToolCall):
            return
        spec = scheduler.registry.get(decision.name)
        child = self._branch.fork(
            new_ulid(),
            predicted=decision,
            step=self._branch.cursor.step_index,
            tier=prediction.tier,
        )
        hazard = analyse(
            child,
            decision,
            spec,
            scheduler.policy,
            staged_keys=scheduler.buffer.staged_keys(child),
            budget=scheduler.budget,
        )
        if hazard is not None:
            # Journaled, not merely counted. ``Branch.stall`` had no caller anywhere in the
            # runtime, so no branch ever reached STALLED and no ``branch_resolved{stalled}``
            # entry was ever written -- and the ledger's stall reader selects on exactly that
            # status. ``Ledger.stalls`` was therefore empty on every run ever produced, and
            # ``branches_stalled`` was always 0, on runs where hazards demonstrably fired.
            # branch.py's own docstring says a branch ends "in exactly one of retired, squashed
            # or stalled"; one of the three was unreachable.
            #
            # Recorded as a resolution and deliberately NOT as a fork. This branch never ran:
            # it was considered and refused before anything was issued on it, so journaling a
            # ``branch_forked`` for it would inflate the fork count and make a refusal look
            # like an attempt. The resolution carries the hazard, the node and the call that
            # was refused, which is everything needed to attribute it.
            scheduler.record_stall(child.cursor.step_index, hazard)
            self.stalled += 1
            child.stall(hazard.name)
            await scheduler.journal.append_async(
                scheduler.run_id,
                "branch_resolved",
                {
                    "v": 1,
                    "branch_id": child.id,
                    "step": child.cursor.step_index,
                    "status": "stalled",
                    "hazard": hazard.name,
                    "node_id": self._node_id,
                    "reason": hazard.name,
                    "refused": decision_payload(decision),
                },
            )
            scheduler.counters.branches_stalled += 1
            return

        await scheduler._journal_fork(child, self._node_id)
        scheduler.counters.branches_forked += 1
        self._predicted = decision
        self._speculative = child
        self._tier = prediction.tier
        self._cost_tokens = prediction.cost_tokens
        # An open guess. Counted up here and down at resolution, on either path. This was
        # never counted at all, so ``max_inflight_branches`` -- one of the three limits Hard
        # Rule 10 names -- could not be reached and the BUDGET hazard's inflight clause was
        # unreachable.
        scheduler.budget.inflight_branches += 1
        tools = BranchTools(scheduler, child, self._node_id)
        # The block this predicts would be the next one the model emits, so it occupies the
        # next ordinal. Reserving it here is what keeps a confirmed speculation's effect at the
        # same program position the sequential run would have given it.
        predicted_step = self._base + len(self.decisions) + 1
        self._open = asyncio.create_task(
            self._run_speculation(tools, child, decision, predicted_step)
        )

    async def _run_speculation(
        self, tools: BranchTools, child: Branch, decision: ToolCall, step: int
    ) -> JsonValue:
        """Run the predicted call on the child, at the program position the parent would use.

        Exactly one step is taken, by ``BranchTools.call`` -- the same single advance the
        canonical branch makes for the same call. An extra ``child.advance_step()`` here made
        the child burn two positions for one call, which the parent's single advance at confirm
        then cancelled *only by coincidence*: the two off-by-ones agreed when one parent-issued
        early read happened to be pending-unstarted at fork time, and diverged otherwise. When
        they diverged the adopted effect's step index was wrong, so its idempotency key was
        wrong, so a resume with speculation off would not dedupe against a crashed run that had
        it on -- and the effect would be delivered twice. Hard Rule 9 sees it as a ledger
        mismatch; Hard Rule 8 is what it actually breaks.
        """
        return await tools.call(decision.name, decision.args, step=step)

    async def _resolve_prediction(self, actual: ToolCall) -> None:
        """The model just said what it actually wants. Compare, and keep or throw away."""
        if self._predicted is None or self._speculative is None:
            return
        scheduler = self._scheduler
        child = self._speculative
        status = resolve_decision(self._predicted, actual)
        confirmed = status is BranchStatus.CONFIRMED

        if confirmed:
            # Only the hit is recorded here. The miss is recorded by ``_squash_open``, which
            # this falls through to -- recording it in both places double-counted it, and
            # recording it only here left end-of-turn squashes out of the window entirely.
            scheduler.budget.record_resolution(
                tier=self._tier, confirmed=True, tokens=self._cost_tokens
            )
            scheduler.budget.inflight_branches -= 1
            await scheduler._journal_alpha_observed(child.fork_step, child.id)
            self.confirmed += 1
            child.confirm()
            # The guess was right, so the work stops being speculative and becomes the
            # canonical branch's. Anything the child staged has to move with it: only the
            # canonical branch retires, and the drain dispatches by branch id, so an effect
            # left behind here is one no drain will ever find and one whose ack the node body
            # waits on forever.
            # No cursor arithmetic here any more. The canonical branch does not make this
            # call, but the position it would have occupied was reserved by ordinal before the
            # speculation ran, and the end of the turn sets the cursor past it. An advance here
            # used to compensate for the child taking two positions for one call, and the two
            # errors cancelled only by coincidence.
            await scheduler.buffer.adopt(child, self._branch)
            # Deliberately no park signal here. Waking the scheduler at this point drains a
            # buffer that holds the adopted effect and nothing else -- the writes from blocks
            # the model emitted *earlier* are staged after the stream ends, so they are not
            # there yet. The adopted call then reached the world before the call that preceded
            # it, and the order effects arrived depended on whether the runtime speculated.
            # The parent parks when it actually needs this ack, which is in the results loop.
            if self._open is not None:
                self._adopted_tasks.add(self._open)
            await scheduler.journal.append_async(
                scheduler.run_id,
                "branch_resolved",
                {
                    "v": 1,
                    "branch_id": child.id,
                    "step": child.fork_step,
                    "status": "confirmed",
                    "adopted_by": self._branch.id,
                },
            )
            self._adopted = self._open
            self._open = None
            self._predicted = None
            self._speculative = None
            return

        await self._squash_open("mismatch")

    async def _squash_open(self, reason: str) -> None:
        """Cancellation *is* the squash. The buffer is closed before the task is cancelled."""
        if self._speculative is None:
            return
        scheduler = self._scheduler
        child = self._speculative
        child.squash(reason)
        self.squashed += 1
        # Every squash is a miss in the alpha window, whatever its reason. End-of-turn
        # squashes -- a guess the model never got round to contradicting -- were not recorded
        # at all, so the window overstated the acceptance rate by exactly those. The tokens
        # the guess cost to produce are wasted from here on.
        scheduler.budget.record_resolution(
            tier=self._tier, confirmed=False, tokens=self._cost_tokens
        )
        scheduler.budget.inflight_branches -= 1
        await scheduler._journal_alpha_observed(child.fork_step, child.id)
        # Revoke first, cancel second: a tool that cannot be cancelled finishes anyway, and a
        # closed buffer is what stops its write from being staged into something nothing will
        # ever drain.
        discarded = await scheduler.buffer.discard_and_journal(child, reason)
        scheduler.counters.effects_discarded += discarded
        scheduler.counters.branches_squashed += 1
        if self._open is not None:
            self._open.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._open
        # The cost travels with the resolution. The ledger sums ``wasted_tokens`` from
        # ``branch_resolved`` and from nowhere else, so a squash that fed the budget but not
        # the journal left the receipt saying 0 beside a gate that had closed for tokens.
        scheduler.counters.wasted_tokens += self._cost_tokens
        await scheduler.journal.append_async(
            scheduler.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": child.id,
                "step": child.fork_step,
                "status": "squashed",
                "reason": reason,
                "tier": self._tier,
                "wasted_tokens": self._cost_tokens,
            },
        )
        self._open = None
        self._predicted = None
        self._speculative = None
        self._adopted = None

    async def _timed_read(self, tools: BranchTools, decision: ToolCall, ordinal: int) -> JsonValue:
        """Run a read the model has emitted but whose turn is not yet durable.

        Counted as an unjournaled read for the duration: the turn that asked for it is still
        streaming, so if the process died now there would be no journaled decision authorising
        it. It is reported as a speculative upstream read and re-validated at retirement if it
        carries a witness.

        The counter is incremented rather than the branch's ``status`` being flipped and
        restored. Flipping raced with retirement -- a read still in flight when the branch was
        confirmed put the old status back afterwards, demoting a CONFIRMED branch to
        SPECULATIVE and making the next drain fail. Overlapping the drain is the entire point
        of early issue, so that race was reachable by design rather than by accident.
        """
        self._branch.unjournaled_reads += 1
        try:
            value = await tools.call(
                decision.name,
                decision.args,
                step=self._base + ordinal + 1,
                pending_writes=tuple(self._pending_write_keys),
            )
        finally:
            self._branch.unjournaled_reads -= 1
        self._results_so_far.append(value)
        while len(self.completed_at) <= ordinal:
            self.completed_at.append(0.0)
        self.completed_at[ordinal] = time.monotonic()
        return value
