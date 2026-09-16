"""The store buffer: where a write waits until the model's real decision confirms it.

This is the mechanism the project is about. A speculative branch that wants to write does not
write; it *stages*, and receives a placeholder instead of a value. The effect leaves the
runtime only when :meth:`StoreBuffer.drain` is called, which happens only for a branch in
``CONFIRMED``, only after the journal entry that confirmed it is durable, and only in the order
the effects were staged (Hard Rule 3).

**Why a store buffer and not a saga.** A saga runs the write and undoes it if something later
fails. Compensation is a *second* effect that reaches the world, and for sending an email,
charging a card or POSTing a webhook it is imperfect or impossible. A store buffer never lets
the first effect out. Compensation is kept, as the ``COMPENSABLE`` class for effects that have
already retired and later need undoing, but it is not what makes speculation safe.

**A staged effect is frozen, and drain sends exactly the bytes that were staged.** There is no
back-patching of arguments at drain time and no handle-accepting tools in this version. Three
Hard Rules each independently forbid that feature: a predicted call carrying a handle can never
equal the model's actual output, which emits the real id, so it would squash every time
(Rule 4); the key is taken over the arguments, and a handle embeds a fresh identifier, so the
key would not be stable across resume and replay, and substituting at drain would mean the key
is unknown when the dedupe check has to happen (Rule 8); and a ledger row whose arguments show
a handle could never normalise equal to the sequential run's row showing the real value
(Rule 9). ``depends_on`` is kept in the shape section 7 specifies and is always empty here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from specunode.buffer.dispatcher import Dispatcher, ToolDispatchError
from specunode.buffer.idempotency import dedupe_key, idempotency_key
from specunode.canonical import JsonValue, canonical, chash
from specunode.core.branch import Branch, BranchClosed, BranchStatus
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolSpec
from specunode.core.hazards import (
    WILDCARD_KEY,
    HazardViolation,
    handle_for,
    has_handle,
    keys_conflict,
    keys_touched,
)
from specunode.ids import new_ulid
from specunode.journal.journal import Claim, Journal, PendingClaim

__all__ = [
    "DrainReport",
    "EffectOutcome",
    "ForwardHazard",
    "ForwardMiss",
    "ForwardResult",
    "StagedEffect",
    "StoreBuffer",
]


class EffectOutcome(Enum):
    DISPATCHED = "DISPATCHED"
    DEAD_LETTER = "DEAD_LETTER"
    #: Already acked by an earlier attempt or an earlier process. Sent nothing.
    SKIPPED_DEDUPE = "SKIPPED_DEDUPE"
    #: The drain halted before reaching this effect. Stage order is preserved across a
    #: resume precisely so these can be attempted later, in the same order.
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass(frozen=True, slots=True)
class StagedEffect:
    """A write that has not happened."""

    id: str
    branch_id: str
    lineage: tuple[str, ...]
    step: int
    node_id: str
    call: ToolCall
    effect: EffectClass
    #: Lineage-bearing. Indexes the buffer; never leaves the runtime.
    key: str
    #: Lineage-free. The dedupe primary key, and the token the tool is handed.
    nkey: str
    stage_index: int
    #: Resource keys this call touches, for read-after-staged-write detection.
    touches: frozenset[str] = frozenset({WILDCARD_KEY})
    #: Always empty in this version; see the module docstring. Kept for the section 7 shape.
    depends_on: frozenset[str] = frozenset()
    #: The symbolic handle the branch got instead of a value. Always present: a staged effect
    #: has no value to give, on any branch.
    placeholder: str | None = None
    compensator: str | None = None
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class ForwardMiss:
    """No staged write in this branch's lineage touches what this read touches."""


@dataclass(frozen=True, slots=True)
class ForwardHazard:
    """A staged write may have changed what this read would return."""

    reason: str
    effect_id: str | None = None


#: There is no ``ForwardHit`` in this version, and no policy switch that could produce one.
#: A projection has no resolution signal -- there is no later model output to compare it
#: against -- so it would retire unverified. Worse, a tool call whose arguments were computed
#: from a wrong projection reaches the world with different arguments from the sequential run,
#: so its equivalence key differs and Hard Rule 9's mandatory test fails *in a supported
#: configuration*. Rule 9 has no policy escape clause. Deleting the feature is simpler than
#: every alternative, and it removes the only setting that could void a Hard Rule.
ForwardResult: TypeAlias = ForwardMiss | ForwardHazard


@dataclass(frozen=True, slots=True)
class DrainReport:
    """The operational view of a drain. The ledger is rendered from journal entries, not this."""

    branch_id: str
    outcomes: tuple[tuple[str, EffectOutcome], ...]
    ok: bool
    halted_at: int | None = None
    #: Effects that were staged and never got an outcome. Always empty in a correct run; a
    #: non-empty value means an authorised write went missing, which is invisible to every
    #: other check because the world simply never hears about it.
    undrained: tuple[str, ...] = ()

    def count(self, outcome: EffectOutcome) -> int:
        return sum(1 for _, seen in self.outcomes if seen is outcome)


@dataclass
class StoreBuffer:
    """Per-branch staged effects, and the only path by which one reaches the world."""

    journal: Journal
    run_id: str

    _staged: dict[str, list[StagedEffect]] = field(default_factory=dict)
    _lineages: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: child branch id -> the branch its staged effects were adopted into, once a
    #: prediction was confirmed. Kept so a late stage on an adopted child lands where
    #: it will actually be drained rather than in a list nothing reads.
    _adopted_into: dict[str, str] = field(default_factory=dict)
    _closed: set[str] = field(default_factory=set)
    _drained: set[str] = field(default_factory=set)
    #: Effects that already reached a terminal outcome, so a second drain of the same branch
    #: neither re-claims nor re-reports them. A node body released by one drain can stage more
    #: writes, and the scheduler drains again; without this the second pass would re-walk the
    #: first pass's effects.
    _settled: dict[str, set[str]] = field(default_factory=dict)
    _dispatch_seq: dict[str, int] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    #: Futures a node body awaits for a staged write's real result. Completed only by the
    #: drain, and only after the confirming entry is durable.
    _acks: dict[str, asyncio.Future[JsonValue]] = field(default_factory=dict)
    #: Kept for callers that want to name the driving task explicitly. The guard that matters
    #: is in ``drain`` and is expressed against the *branch's* task, not this one: on a
    #: framework that owns its own loop the drain legitimately runs on the framework's task.
    scheduler_task: asyncio.Task[object] | None = None

    # -- staging ---------------------------------------------------------------------------

    async def stage(
        self, branch: Branch, call: ToolCall, spec: ToolSpec, *, node_id: str = ""
    ) -> StagedEffect:
        """Record a write without performing it. Never executes anything.

        Async, unlike section 7's signature, because the ``effect_staged`` entry is fsynced
        before the branch is told the effect exists. Logged as a signature change: staging
        without a durable record would let a resume lose an effect the branch believes it
        holds, and the branch would go on to compute later arguments from a write that no
        longer exists anywhere.
        """
        if branch.id in self._closed or not branch.running:
            raise BranchClosed(
                f"branch {branch.id} is {branch.status.value}; a tool that could not be "
                "cancelled must not resurrect a buffer nothing will drain"
            )
        if has_handle(canonical(dict(call.args))):
            # Defence in depth: hazard analysis should have stalled the branch before here.
            raise HazardViolation(
                f"{call.name} was staged with a placeholder in its arguments; this is a "
                "RETURN_VALUE_DEPENDENCY hazard and should have stalled the branch"
            )

        effect_id = new_ulid()
        staged = self._staged.setdefault(branch.id, [])
        effect = StagedEffect(
            id=effect_id,
            branch_id=branch.id,
            lineage=branch.lineage,
            step=branch.cursor.step_index,
            node_id=node_id,
            call=call,
            effect=spec.effect,
            key=idempotency_key(
                run_id=self.run_id,
                lineage=branch.lineage,
                node_id=node_id,
                step_index=branch.cursor.step_index,
                tool_name=call.name,
                args=call.args,
            ),
            nkey=dedupe_key(
                run_id=self.run_id,
                node_id=node_id,
                step_index=branch.cursor.step_index,
                tool_name=call.name,
                args=call.args,
            ),
            stage_index=len(staged),
            touches=keys_touched(spec, call.args),
            # Always, with no sequential special case. A node body that writes and then reads
            # the result -- ack = await charge_card(...); send_receipt(ack["charge_id"]) -- gets
            # a future here, not a value, and only the drain completes it. Handing back a real
            # ack instead would mean dispatching inside the call, which is task 1.6's planted
            # bug; and having the caller await the drain deadlocks the first write of the first
            # sequential run, before any speculation exists.
            placeholder=handle_for(effect_id),
            compensator=spec.compensator,
            idempotent=spec.idempotent,
        )
        if any(existing.nkey == effect.nkey for existing in staged):
            # Two effects with one key means dedupe will suppress the second at dispatch, and
            # an effect the run performed never reaches the world -- silently, and identically
            # in both arms, so no equivalence or leak check can see it. The cause is always a
            # caller that did not take a fresh step index for this call.
            raise HazardViolation(
                f"{call.name} would be staged under a key branch {branch.id} already holds "
                f"(step {effect.step}, node {node_id!r}). Take a fresh step index per call: "
                "two calls sharing one key means the second is silently never dispatched."
            )
        staged.append(effect)
        self._acks[effect.id] = asyncio.get_running_loop().create_future()
        self._lineages[branch.id] = branch.lineage
        await self.journal.append_async(
            self.run_id,
            "effect_staged",
            {
                "v": 1,
                "effect_id": effect.id,
                "branch_id": effect.branch_id,
                "lineage": list(effect.lineage),
                "step": effect.step,
                "node_id": effect.node_id,
                "tool": call.name,
                "args": dict(call.args),
                "args_hash": chash(dict(call.args)),
                "effect": effect.effect.value,
                "idempotent": effect.idempotent,
                "compensator": effect.compensator,
                "key": effect.key,
                "nkey": effect.nkey,
                "stage_index": effect.stage_index,
                "touches": sorted(effect.touches),
                "depends_on": sorted(effect.depends_on),
                "placeholder": effect.placeholder,
            },
        )
        return effect

    # -- what a branch can see ----------------------------------------------------------------

    def staged_in_lineage(self, branch: Branch) -> tuple[StagedEffect, ...]:
        """This branch's staged effects and its ancestors' -- never a sibling's.

        Hard Rule 6 defines what a branch sees as the committed state at its fork point plus
        its own staged effects. A child is a continuation of its parent, so the parent's
        staged writes are in the child's own past; siblings are mutually exclusive
        alternatives and stay invisible. Reading only ``branch.staged`` would break at depth
        two, which the default ``max_speculation_depth`` of 3 makes reachable.
        """
        out: list[StagedEffect] = []
        for branch_id in branch.lineage:
            out.extend(self._staged.get(branch_id, ()))
        return tuple(out)

    def staged_keys(self, branch: Branch) -> tuple[frozenset[str], ...]:
        return tuple(effect.touches for effect in self.staged_in_lineage(branch))

    def forward(self, branch: Branch, read: ToolCall, spec: ToolSpec) -> ForwardResult:
        """Store-to-load forwarding, which in this version can only miss or refuse.

        If a staged write in this branch's lineage may touch what this read touches, the read
        cannot be trusted: it would return the pre-write value while the branch goes on to
        compute later arguments as though the write had happened. That is the one hazard
        class with no backstop -- witness validation cannot catch it, because the branch's own
        staged write has not been dispatched and so cannot have made anything stale.
        """
        touched = keys_touched(spec, read.args)
        for effect in self.staged_in_lineage(branch):
            if keys_conflict(touched, effect.touches):
                unnameable = WILDCARD_KEY in touched or WILDCARD_KEY in effect.touches
                return ForwardHazard(
                    reason=(
                        "read or staged write did not declare forward_keys, so an overlap "
                        "cannot be ruled out"
                        if unnameable
                        else "read touches a resource key a staged write touches"
                    ),
                    effect_id=effect.id,
                )
        return ForwardMiss()

    # -- adoption ---------------------------------------------------------------------------

    async def adopt(self, child: Branch, parent: Branch) -> int:
        """Move a confirmed speculation's staged effects onto the branch that will retire.

        When the model emits exactly what was predicted, the speculative branch stops being a
        guess: the call it staged is the call the run was always going to make. But only the
        canonical branch retires, and :meth:`drain` dispatches by ``branch.id`` -- so without
        this the effect sits in the child's list forever. The node body awaiting its ack never
        wakes, and the run deadlocks rather than failing, which is the worst of the three
        possible outcomes because nothing reports it.

        That was a real defect: a tier-1 drafter that correctly predicted a *write* hung the
        run. It survived because the only confirmed predictions any test had ever made were
        reads, which stage nothing.

        The effect is re-attributed to the parent's ``branch_id`` and lineage, because that is
        what it would have carried had it been staged without speculation -- and Hard Rule 3's
        audit asks that every effect in the world trace to a branch that *retired*, which the
        child never does. What deliberately does **not** change is ``nkey``: it is the token
        the tool is handed, and an idempotency key that shifted when a guess turned out right
        would make a retry after adoption look like a different call.

        ``stage_index`` is renumbered onto the end of the parent's list, preserving order: the
        adoption happens mid-stream, before the parent stages anything the turn's later blocks
        ask for, so appending is the order the sequential run would have produced.
        """
        moved = self._staged.pop(child.id, [])
        if not moved:
            # Nothing staged -- a predicted read, which is the common case. Still record the
            # adoption so the child's lineage does not silently keep receiving stages.
            self._adopted_into[child.id] = parent.id
            return 0

        target = self._staged.setdefault(parent.id, [])
        adopted: list[StagedEffect] = []
        for offset, effect in enumerate(moved):
            adopted.append(
                replace(
                    effect,
                    branch_id=parent.id,
                    lineage=parent.lineage,
                    stage_index=len(target) + offset,
                )
            )
        target.extend(adopted)
        self._adopted_into[child.id] = parent.id
        self._lineages[parent.id] = parent.lineage
        await self.journal.append_async(
            self.run_id,
            "effect_adopted",
            {
                "v": 1,
                "branch_id": parent.id,
                "from_branch_id": child.id,
                "step": parent.cursor.step_index,
                "effect_ids": [effect.id for effect in adopted],
                "count": len(adopted),
            },
        )
        return len(adopted)

    def adopted_into(self, branch_id: str) -> str | None:
        """The branch a confirmed speculation's effects were moved to, if any."""
        return self._adopted_into.get(branch_id)

    # -- discard ---------------------------------------------------------------------------

    def discard(self, branch: Branch) -> int:
        """Drop this branch's staged effects and every descendant's. Never dispatches.

        Idempotent: the scheduler may also discard each squashed descendant individually, and
        without idempotence the ledger's discarded count would double and Demo 1 would print a
        number no run produced.
        """
        doomed = [
            branch_id
            for branch_id, lineage in self._lineages.items()
            if branch_id == branch.id or branch.id in lineage
        ]
        doomed.append(branch.id)
        dropped: list[StagedEffect] = []
        for branch_id in dict.fromkeys(doomed):
            dropped.extend(self._staged.pop(branch_id, []))
            self._closed.add(branch_id)
        for effect in dropped:
            # A node body parked on one of these must not wait forever for a value that is
            # never coming. Cancelling is what lets squash-by-cancellation actually free it.
            ack = self._acks.pop(effect.id, None)
            if ack is not None and not ack.done():
                ack.cancel()
        return len(dropped)

    async def discard_and_journal(self, branch: Branch, reason: str) -> int:
        dropped_ids = [effect.id for effect in self.staged_in_lineage(branch)]
        count = self.discard(branch)
        if count:
            await self.journal.append_async(
                self.run_id,
                "effect_discarded",
                {
                    "v": 1,
                    "branch_id": branch.id,
                    "step": branch.cursor.step_index,
                    "effect_ids": dropped_ids[:count],
                    "count": count,
                    "reason": reason,
                },
            )
        return count

    def pending(self, branch_id: str) -> tuple[StagedEffect, ...]:
        return tuple(self._staged.get(branch_id, ()))

    def ack_for(self, effect_id: str) -> asyncio.Future[JsonValue]:
        """The future a node body awaits for a staged write's real result.

        Awaiting one is a barrier for everything except the drain: a branch holding an
        unresolved ack has a staged write, so its next model request would carry a placeholder
        and is refused by Hard Rule 13's gate.
        """
        ack = self._acks.get(effect_id)
        if ack is None:
            raise KeyError(f"no staged effect {effect_id!r}")
        return ack

    # -- drain -------------------------------------------------------------------------------

    def _lock_for(self, branch_id: str) -> asyncio.Lock:
        lock = self._locks.get(branch_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[branch_id] = lock
        return lock

    async def drain(
        self,
        branch: Branch,
        dispatcher: Dispatcher,
        *,
        confirmed_offset: int,
        authorised_by_offset: int,
    ) -> DrainReport:
        """Send this branch's staged effects to the world, in stage order.

        The preconditions are re-asserted *here* rather than trusted from the caller. That is
        deliberate defence in depth: task 1.6 plants exactly this bug -- draining a CONFIRMED
        branch before the entry that confirmed it is durable -- and a precondition checked only
        at the call site would be planted around.
        """
        # The invariant is about the *branch*, not about which particular task drives: a node
        # body waiting on a staged write's result must never be the thing that dispatches it.
        # The repair that suggests itself when a node is parked -- drain inline from the call
        # that staged the effect -- dispatches before the branch is confirmed, which is task
        # 1.6's planted bug. Naming the branch's own task makes that unreachable while still
        # allowing a framework that owns its run loop to drain from its own driver task.
        if branch.task is not None and asyncio.current_task() is branch.task:
            raise BranchClosed(
                f"drain was called from branch {branch.id}'s own task. A node cannot dispatch "
                "the write it is waiting on: that is a dispatch before the branch is confirmed."
            )
        if branch.status is not BranchStatus.CONFIRMED:
            raise BranchClosed(
                f"drain called on a {branch.status.value} branch; only CONFIRMED branches "
                "dispatch (Hard Rule 3)"
            )
        if branch.speculative_prompts and not branch.context_verified:
            raise BranchClosed(
                f"branch {branch.id} sent {branch.speculative_prompts} request(s) while "
                "speculating and has not passed the context-identity check; nothing "
                "downstream of an unverified request may reach the world (Hard Rule 13)"
            )
        # Not "is the offset <= the head", which any later entry satisfies: read the entry and
        # check it really is this branch's confirmation. Otherwise a drain that ran before its
        # own confirming entry was written passes as soon as anything else has been journaled
        # since, and task 1.6's planted bug walks straight through.
        confirming = next(
            iter(self.journal.read(self.run_id, after=confirmed_offset - 1, chunk=1)), None
        )
        if (
            confirming is None
            or confirming.offset != confirmed_offset
            or confirming.kind != "branch_resolved"
            or confirming.payload.get("status") != "confirmed"
            or confirming.payload.get("branch_id") != branch.id
        ):
            raise BranchClosed(
                f"offset {confirmed_offset} is not a durable branch_resolved(confirmed) entry "
                f"for branch {branch.id}; nothing dispatches before it is (Hard Rule 3)"
            )

        async with self._lock_for(branch.id):
            report = await self._drain_locked(
                branch, dispatcher, authorised_by_offset, confirmed_offset
            )
            self._drained.add(branch.id)
            return report

    async def _drain_locked(
        self,
        branch: Branch,
        dispatcher: Dispatcher,
        authorised_by_offset: int,
        confirmed_offset: int,
    ) -> DrainReport:
        outcomes: list[tuple[str, EffectOutcome]] = []
        halted_at: int | None = None
        ok = True
        settled = self._settled.setdefault(branch.id, set())
        index = -1

        # The list is read live, by index, and never snapshotted: completing one effect's ack
        # can resume a node body that stages the next write from the value it just received.
        # Positional, never sorted -- not by key and not by effect id. Effect ids are ULIDs, so
        # sorting by one usually *reproduces* insertion order, which would make a reordering
        # bug invisible in testing and surface only under clock skew.
        while True:
            index += 1
            live = self._staged.get(branch.id, [])
            if index >= len(live):
                break
            effect = live[index]
            if effect.id in settled:
                continue

            if halted_at is not None:
                outcomes.append((effect.id, EffectOutcome.NOT_ATTEMPTED))
                continue

            dispatch_index = self._dispatch_seq.get(branch.id, 0)
            claim = await self.journal.claim_dispatch(
                PendingClaim(
                    run_id=self.run_id,
                    nkey=effect.nkey,
                    idem_key=effect.key,
                    effect_id=effect.id,
                    branch_id=branch.id,
                    tool=effect.call.name,
                )
            )
            if claim.outcome is Claim.ALREADY_DISPATCHED:
                settled.add(effect.id)
                self._complete_ack(effect.id, claim.ack)
                outcomes.append((effect.id, EffectOutcome.SKIPPED_DEDUPE))
                continue
            if claim.outcome is Claim.AMBIGUOUS and not effect.idempotent:
                # A previous attempt may already have taken effect and the tool has not said a
                # repeat is harmless. Guessing either way is worse than stopping.
                settled.add(effect.id)
                await self._dead_letter(
                    effect,
                    branch,
                    attempts=claim.attempt,
                    error="ambiguous_after_crash: a previous attempt may have taken effect "
                    "and this tool did not declare itself idempotent",
                    authorised_by_offset=authorised_by_offset,
                )
                outcomes.append((effect.id, EffectOutcome.DEAD_LETTER))
                halted_at, ok = dispatch_index, False
                continue

            result = await dispatcher.dispatch(
                effect.call.name,
                dict(effect.call.args),
                idempotency_key=effect.nkey,
                branch_id=branch.id,
            )
            settled.add(effect.id)
            self._dispatch_seq[branch.id] = dispatch_index + 1
            if result.ok:
                await self.journal.settle_dispatch(
                    run_id=self.run_id,
                    nkey=effect.nkey,
                    status="dispatched",
                    ack=result.ack,
                    attempt=result.attempts,
                    kind="effect_dispatched",
                    payload={
                        "v": 1,
                        "effect_id": effect.id,
                        "branch_id": branch.id,
                        "nkey": effect.nkey,
                        "key": effect.key,
                        "tool": effect.call.name,
                        "args_hash": chash(dict(effect.call.args)),
                        "stage_index": effect.stage_index,
                        "dispatch_index": dispatch_index,
                        "authorised_by_offset": authorised_by_offset,
                        "confirmed_by_offset": confirmed_offset,
                        "attempt": result.attempts,
                        "deduped": False,
                        "ack": result.ack,
                        "dry_run": result.dry_run,
                        "compensation_for": None,
                    },
                )
                self._complete_ack(effect.id, result.ack)
                outcomes.append((effect.id, EffectOutcome.DISPATCHED))
                continue

            if result.sent == "no":
                await self.journal.mark_not_sent(self.run_id, effect.nkey, result.attempts)
            await self._dead_letter(
                effect,
                branch,
                attempts=result.attempts,
                error=result.error or "dispatch failed",
                authorised_by_offset=authorised_by_offset,
            )
            outcomes.append((effect.id, EffectOutcome.DEAD_LETTER))
            # Halt rather than skip: the effects after this one were staged on the assumption
            # that this one happened, and sending them anyway would put the world in a state no
            # run ever produced.
            halted_at, ok = dispatch_index, False

        # An effect that was staged and never got an outcome is an authorised write that
        # vanished. Nothing else in the suite can see that, so it is reported here rather than
        # left to be noticed by its absence from the world.
        undrained = tuple(e.id for e in self.pending(branch.id) if e.id not in settled)
        return DrainReport(
            branch.id,
            tuple(outcomes),
            ok=ok and not undrained,
            halted_at=halted_at,
            undrained=undrained,
        )

    def _complete_ack(self, effect_id: str, ack: JsonValue) -> None:
        future = self._acks.get(effect_id)
        if future is not None and not future.done():
            future.set_result(ack)

    def _fail_ack(self, effect_id: str, error: str) -> None:
        future = self._acks.get(effect_id)
        if future is not None and not future.done():
            future.set_exception(ToolDispatchError(error, sent="maybe", retriable=False))

    async def _dead_letter(
        self,
        effect: StagedEffect,
        branch: Branch,
        *,
        attempts: int,
        error: str,
        authorised_by_offset: int,
    ) -> None:
        await self.journal.settle_dispatch(
            run_id=self.run_id,
            nkey=effect.nkey,
            status="dead_letter",
            ack=None,
            attempt=attempts,
            kind="effect_dead_lettered",
            payload={
                "v": 1,
                "effect_id": effect.id,
                "branch_id": branch.id,
                "nkey": effect.nkey,
                "tool": effect.call.name,
                "args_hash": chash(dict(effect.call.args)),
                "attempts": attempts,
                "last_error": {"type": "ToolDispatchError", "message": error},
                "authorised_by_offset": authorised_by_offset,
            },
        )
        self._fail_ack(effect.id, error)
