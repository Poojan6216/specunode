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
from dataclasses import dataclass, field
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

    def count(self, outcome: EffectOutcome) -> int:
        return sum(1 for _, seen in self.outcomes if seen is outcome)


@dataclass
class StoreBuffer:
    """Per-branch staged effects, and the only path by which one reaches the world."""

    journal: Journal
    run_id: str

    _staged: dict[str, list[StagedEffect]] = field(default_factory=dict)
    _lineages: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _closed: set[str] = field(default_factory=set)
    _drained: set[str] = field(default_factory=set)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    #: Futures a node body awaits for a staged write's real result. Completed only by the
    #: drain, and only after the confirming entry is durable.
    _acks: dict[str, asyncio.Future[JsonValue]] = field(default_factory=dict)
    #: When set, ``drain`` refuses any caller but this task. Without the guard, the obvious
    #: repair for "my node body is waiting for an ack" is to drain inline from the branch
    #: task -- which dispatches before the branch is confirmed.
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
        if self.scheduler_task is not None and asyncio.current_task() is not self.scheduler_task:
            raise BranchClosed(
                "drain was called from a branch task. Dispatch happens on the scheduler's own "
                "task, after the confirming entry is durable; draining from the branch that "
                "staged the effect is exactly the bug task 1.6 plants."
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
        last = self.journal.last_offset(self.run_id)
        if last is None or confirmed_offset > last:
            raise BranchClosed(
                f"the entry confirming branch {branch.id} (offset {confirmed_offset}) is not "
                f"durable; the journal head is {last} (Hard Rule 3)"
            )

        async with self._lock_for(branch.id):
            if branch.id in self._drained:
                return DrainReport(
                    branch.id,
                    tuple((e.id, EffectOutcome.SKIPPED_DEDUPE) for e in self.pending(branch.id)),
                    ok=True,
                )
            self._drained.add(branch.id)
            return await self._drain_locked(
                branch, dispatcher, authorised_by_offset, confirmed_offset
            )

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
        # Positional, never sorted -- not by key and not by effect id. Effect ids are ULIDs,
        # so sorting by one usually *reproduces* insertion order, which would make a
        # reordering bug invisible in testing and surface only under clock skew.
        effects = self.pending(branch.id)

        for dispatch_index, effect in enumerate(effects):
            if halted_at is not None:
                outcomes.append((effect.id, EffectOutcome.NOT_ATTEMPTED))
                continue

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
                self._complete_ack(effect.id, claim.ack)
                outcomes.append((effect.id, EffectOutcome.SKIPPED_DEDUPE))
                continue
            if claim.outcome is Claim.AMBIGUOUS and not effect.idempotent:
                # A previous attempt may already have taken effect and the tool has not said
                # a repeat is harmless. Guessing either way is worse than stopping.
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
            # that this one happened, and sending them anyway would put the world in a state
            # no run ever produced.
            halted_at, ok = dispatch_index, False

        return DrainReport(branch.id, tuple(outcomes), ok=ok, halted_at=halted_at)

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
