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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import EffectOutcome, StoreBuffer
from specunode.canonical import JsonValue, chash
from specunode.core.branch import Branch, BranchStatus, ReadRecord, StepCursor
from specunode.core.decision import Decision, ToolCall, is_barrier
from specunode.core.effects import EffectClass, ToolRegistry
from specunode.core.graph import END, GraphAdapter, NodeRef, RunSession
from specunode.core.hazards import Hazard, analyse
from specunode.core.model import CallScope, ModelClient, call_scope
from specunode.core.policy import Policy
from specunode.core.state import CommittedState, Reducer, patch_payload, resolve_reducers
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.journal.ledger import Ledger, build_ledger

__all__ = ["BranchOutcome", "RunResult", "Scheduler", "SchedulerError"]


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
    reducers: Mapping[str, str] = field(default_factory=dict)

    run_id: str = ""
    counters: _Counters = field(default_factory=_Counters)
    #: Signalled by the tool port when a branch parks on a staged write's result. An Event
    #: rather than a flag because the scheduler has to wait for the *next* park, not merely
    #: observe that one happened: a node released by a drain may stage again, and spinning on
    #: a flag cannot tell the two apart.
    _park_events: dict[str, asyncio.Event] = field(default_factory=dict)
    _stalls: list[tuple[int, Hazard]] = field(default_factory=list)
    _retire_seq: int = 0

    # -- bookkeeping the ports call back into ------------------------------------------------

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
        self.buffer.scheduler_task = asyncio.current_task()
        committed = CommittedState.initial(inputs if isinstance(inputs, Mapping) else {})
        cursor = StepCursor()
        reducers = resolve_reducers(dict(self.reducers))
        await self._journal_run_started(inputs)

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

                report = await self._retire(branch, node_id)
                if not report:
                    ok, error = False, f"effects from node {node.name} did not all dispatch"

                committed, cursor = await self._commit(branch, committed, reducers)
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

    async def _retire(self, branch: Branch, node_id: str) -> bool:
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
            return False
        if undrained:
            raise SchedulerError(
                f"branch {branch.id} staged {len(undrained)} effect(s) that were never "
                "dispatched; an authorised write cannot be silently dropped"
            )
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
            },
        )
        return ok

    async def _commit(
        self,
        branch: Branch,
        committed: CommittedState,
        reducers: Mapping[str, Reducer],
    ) -> tuple[CommittedState, StepCursor]:
        """Apply the retiring branch's delta, and journal what it did to committed state."""
        patch = branch.state.delta()
        if not patch:
            return committed, branch.cursor
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
        return result.state, branch.cursor

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
