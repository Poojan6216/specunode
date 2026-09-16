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
from specunode.core.decision import Decision, ToolCall, is_barrier
from specunode.core.effects import EffectClass, ToolRegistry
from specunode.core.graph import END, GraphAdapter, NodeRef, RunSession, session_scope
from specunode.core.hazards import Hazard, analyse
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

    async def call(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        scheduler = self._scheduler
        branch = self._branch
        spec = scheduler.registry.get(name)
        call = ToolCall(name=name, args=dict(args))
        step = branch.advance_step()

        hazard = analyse(
            branch,
            call,
            spec,
            scheduler.policy,
            staged_keys=scheduler.buffer.staged_keys(branch),
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
                "program_order": len(branch.read_set),
            },
        )

        if spec.effect is EffectClass.READ:
            return await scheduler.execute_read(branch, call, spec, call_id, step, self._node_id)

        effect = await scheduler.buffer.stage(branch, call, spec, node_id=self._node_id)
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
        speculative = branch.status is BranchStatus.SPECULATIVE
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
                    ok, error = False, f"node {node.name} failed"
                    break

                drained, updated = await self._retire(branch, node_id, committed, reducers)
                if not drained:
                    ok, error = False, f"effects from node {node.name} did not all dispatch"
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
        await self._quiesce(branch, task)  # type: ignore[arg-type]
        # State on this path belongs to the framework's checkpointer, so no delta is journaled
        # and none is passed here.
        await self._retire(branch, node_id)
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
                # output rather than resolved against a guess.
                "predicted": None,
                "predicted_hash": "",
                "tier": None,
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
                branch.squash(f"{type(exc).__name__}: {exc}")
                return BranchOutcome.FAULTED
            return BranchOutcome.DONE
        return BranchOutcome.PARKED

    async def _retire(
        self,
        branch: Branch,
        node_id: str,
        committed: CommittedState | None = None,
        reducers: Mapping[str, Reducer] | None = None,
    ) -> tuple[bool, CommittedState | None]:
        """R5 through R9: confirm, make it durable, drain, then retire."""
        branch.confirm()
        # Nothing to rebuild in sequential mode: every request this branch sent carried real
        # values, because it never ran on a prediction. The count is what the ledger reports,
        # and it is zero here rather than a claim that something was checked.
        branch.context_verified = True
        confirmed_offset = await self.journal.append_async(
            self.run_id,
            "branch_resolved",
            {
                "v": 1,
                "branch_id": branch.id,
                "step": branch.fork_step,
                "status": "confirmed",
                "context_verified": True,
                "context_checks": [0, len(branch.prompts_sent)],
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
                results.append(await task)
                continue
            results.append(await tools.call(emitted.name, emitted.args))
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
            scheduler.record_stall(child.cursor.step_index, hazard)
            self.stalled += 1
            return

        await scheduler._journal_fork(child, self._node_id)
        scheduler.counters.branches_forked += 1
        self._predicted = decision
        self._speculative = child
        self._tier = prediction.tier
        tools = BranchTools(scheduler, child, self._node_id)
        self._open = asyncio.create_task(self._run_speculation(tools, child, decision))

    async def _run_speculation(
        self, tools: BranchTools, child: Branch, decision: ToolCall
    ) -> JsonValue:
        child.advance_step()
        return await tools.call(decision.name, decision.args)

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

        Marked speculative for the duration: the turn that asked for it is still streaming, so
        if the process died now there would be no journaled decision authorising it. It is
        counted as a speculative upstream read and re-validated at retirement if it carries a
        witness.
        """
        previous = self._branch.status
        self._branch.status = BranchStatus.SPECULATIVE
        try:
            value = await tools.call(decision.name, decision.args)
        finally:
            self._branch.status = previous
        self._results_so_far.append(value)
        while len(self.completed_at) <= ordinal:
            self.completed_at.append(0.0)
        self.completed_at[ordinal] = time.monotonic()
        return value
