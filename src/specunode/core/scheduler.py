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
from dataclasses import dataclass, field
from enum import Enum

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import EffectOutcome, StoreBuffer
from specunode.canonical import JsonValue, chash
from specunode.core.branch import Branch, BranchStatus, ReadRecord, StepCursor
from specunode.core.decision import Decision, ToolCall, decision_key, decision_payload, is_barrier
from specunode.core.effects import EffectClass, ToolRegistry
from specunode.core.graph import END, GraphAdapter, NodeRef, RunSession, session_scope
from specunode.core.hazards import Hazard, analyse, keys_touched
from specunode.core.model import (
    CallScope,
    ModelClient,
    RequestEnvelope,
    ToolUseComplete,
    TurnComplete,
    call_scope,
)
from specunode.core.policy import Budget, Policy
from specunode.core.state import CommittedState, Reducer, patch_payload, resolve_reducers
from specunode.drafters.base import DraftContext, Drafter
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.journal.ledger import Ledger, build_ledger
from specunode.journal.replay import recover
from specunode.verify.gate import resolve_decision
from specunode.verify.witness import validate_reads

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
        speculative = branch.status is BranchStatus.SPECULATIVE or branch.unjournaled_reads > 0
        scope = CallScope(
            run_id=self.run_id,
            branch_id=branch.id,
            lineage=branch.lineage,
            step=step,
            node_id=node_id,
            speculative=speculative,
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
                issued_while_speculative=speculative,
            )
        )
        if speculative:
            self.counters.speculative_reads_upstream += 1
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
                "speculative": speculative,
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
        try:
            while True:
                node = self.graph.next(committed.to_dict())
                if node is END or isinstance(node, type(END)):
                    break
                assert isinstance(node, NodeRef)
                cursor, node_id = cursor.visit(node.structural_id)

                branch = self._fork_canonical(cursor, node_id)
                await self._journal_fork(branch, node_id)

                outcome, decision = await self._run_node(node, branch, node_id, committed)
                if outcome is BranchOutcome.FAULTED:
                    # The reason the node failed is the whole of the diagnostic value here.
                    # A replay that refuses because the prompt changed reports the step index
                    # and a field-level diff, and reducing that to "node act failed" throws
                    # away the one thing the operator needs to answer "what changed?".
                    ok = False
                    error = f"node {node.name} failed: {branch.reason or 'no reason recorded'}"
                    break

                drained, updated = await self._retire(branch, node_id, committed, reducers)
                if not drained:
                    # Two different failures reach here and they need different words. A node
                    # that *raised* after its writes went out is not a dispatch failure, and
                    # reporting it as one sends the operator to look at the wrong subsystem
                    # while an effect is already in the world.
                    ok = False
                    error = (
                        f"node {node.name} failed after its effects were dispatched: "
                        f"{branch.reason}"
                        if branch.reason
                        else f"effects from node {node.name} did not all dispatch"
                    )
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
        self._committed = CommittedState(recovery.state)
        self._cursor = recovery.cursor
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
                "policy": {"speculation": self.policy.speculation},
                "graph": {"adapter": self.graph.capabilities().framework},
                "target": {"provider": "configured", "model": "configured"},
                "recovered": {
                    "retired_branches": len(recovery.retired_branches),
                    "confirmed_not_retired": list(recovery.confirmed_not_retired),
                    "unresolved_dispatches": len(recovery.unresolved_dispatches),
                    "step_index": recovery.step_index,
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

    async def _journal_fork(self, branch: Branch, node_id: str) -> None:
        await self.journal.append_async(
            self.run_id,
            "branch_forked",
            {
                "v": 1,
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
        outcome = await self._quiesce(branch, task)
        if outcome is BranchOutcome.FAULTED:
            return outcome, ToolCall("", {})
        if outcome is BranchOutcome.DONE:
            return outcome, task.result()
        # PARKED: the decision is not available until the drain releases the node body.
        return outcome, ToolCall("", {})

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
            task.cancel()
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
            new_committed = await self._commit(branch, committed, reducers or {})

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
                "cursor_after": {
                    "step_index": branch.cursor.step_index,
                    "visits": [[name, count] for name, count in branch.cursor.visits],
                },
            },
        )
        return ok, new_committed

    async def _commit(
        self,
        branch: Branch,
        committed: CommittedState,
        reducers: Mapping[str, Reducer],
    ) -> CommittedState:
        """Apply the retiring branch's delta, and journal what it did to committed state."""
        patch = branch.state.delta()
        if not patch:
            return committed
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

    async def _journal_run_started(self, inputs: JsonValue) -> None:
        capabilities = self.graph.capabilities()
        await self.journal.append_async(
            self.run_id,
            "run_started",
            {
                "v": 1,
                "mode": "run",
                "config_hash": chash({"reducers": dict(self.reducers)}),
                "registry_hash": chash(sorted(self.registry.names())),
                "policy": {
                    "speculation": self.policy.speculation,
                    "max_inflight_branches": self.policy.max_inflight_branches,
                    "max_speculation_depth": self.policy.max_speculation_depth,
                    "stage_irreversible": self.policy.stage_irreversible,
                    "on_stale_read": self.policy.on_stale_read,
                },
                "graph": {
                    "adapter": capabilities.framework,
                    "nodes": [n.structural_id for n in self.graph.nodes()],
                },
                "target": {"provider": "configured", "model": "configured"},
                "inputs": inputs,
            },
        )

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
                if spec.effect is EffectClass.READ and self._adopted is None:
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

        # Any speculation still open when the turn ended predicted a call the model never made.
        await self._squash_open("turn_ended")

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
        if not scheduler.policy.speculation or not scheduler.budget.may_speculate():
            return
        drafter = scheduler.predictor
        if drafter is None:
            return

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
        scheduler.budget.record_resolution(tier=self._tier, confirmed=confirmed, tokens=0)

        if confirmed:
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
        await scheduler.journal.append_async(
            scheduler.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": child.id,
                "step": child.fork_step,
                "status": "squashed",
                "reason": reason,
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
