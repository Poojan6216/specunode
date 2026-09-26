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
    #: How many of each confirmed guess's reads have been moved to its parent (``adopt_reads``).
    _reads_adopted: dict[str, int] = field(default_factory=dict)
    #: What the last :meth:`discard` actually dropped, so the journal can name them.
    _last_discarded: tuple[StagedEffect, ...] = ()
    _closed: set[str] = field(default_factory=set)
    _drained: set[str] = field(default_factory=set)
    #: Effects that already reached a terminal outcome, so a second drain of the same branch
    #: neither re-claims nor re-reports them. A node body released by one drain can stage more
    #: writes, and the scheduler drains again; without this the second pass would re-walk the
    #: first pass's effects.
    _settled: dict[str, set[str]] = field(default_factory=dict)
    _dispatch_seq: dict[str, int] = field(default_factory=dict)
    #: Effects per branch known to be in the world: dispatched here, or found already
    #: dispatched by an earlier process. What an error has to admit to when a branch fails
    #: after its drain.
    _delivered: dict[str, int] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    #: Futures a node body awaits for a staged write's real result. Completed only by the
    #: drain, and only after the confirming entry is durable.
    _acks: dict[str, asyncio.Future[JsonValue]] = field(default_factory=dict)
    #: Kept for callers that want to name the driving task explicitly. The guard that matters
    #: is in ``drain`` and is expressed against the *branch's* task, not this one: on a
    #: framework that owns its own loop the drain legitimately runs on the framework's task.
    scheduler_task: asyncio.Task[object] | None = None
    #: The run a Scheduler is driving with this buffer, while it does. ``run_id`` could not
    #: say: a second run started on the same buffer overwrote it, and the first run's effects
    #: were journaled under the second's.
    driving: str | None = None

    # -- staging ---------------------------------------------------------------------------

    async def stage(
        self,
        branch: Branch,
        call: ToolCall,
        spec: ToolSpec,
        *,
        node_id: str = "",
        step: int | None = None,
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
        # Where this effect will actually be drained from, which is not always this branch: a
        # speculation whose prediction was confirmed has had its buffer adopted by the branch
        # that will retire, and anything it stages afterwards belongs there too.
        owner_id, owner_lineage = self._drain_owner(branch)
        staged = self._staged.setdefault(owner_id, [])

        # The position the *caller* took for this call, not wherever the cursor has since got
        # to. ``Branch.reserve_step`` is ``max(cursor, requested)``, so the cursor is a running
        # maximum over every position any block in the turn reserved -- and which blocks reserve
        # on this branch depends on whether a speculation ran them instead. Reading the cursor
        # here therefore made a write's idempotency key depend on the speculation outcome, which
        # is precisely what ``reserve_step`` exists to prevent and what its own docstring says
        # Hard Rule 9 forbids. The observable harm is a resume charging a card a second time:
        # crash after the dispatch, resume with a different speculation outcome, derive a
        # different nkey, and the dedupe table has nothing to match against.
        #
        # It also loses writes with no crash and no speculation at all. A turn of
        # ``[write, write, read]`` has the read bump the cursor past both writes' slots before
        # either is staged, so both collapse onto one step index, collide on ``nkey`` here, and
        # the second raises after the first has already reached the world.
        position = branch.cursor.step_index if step is None else step
        effect = StagedEffect(
            id=effect_id,
            branch_id=owner_id,
            lineage=owner_lineage,
            step=position,
            node_id=node_id,
            call=call,
            effect=spec.effect,
            key=idempotency_key(
                run_id=self.run_id,
                lineage=branch.lineage,
                node_id=node_id,
                step_index=position,
                tool_name=call.name,
                args=call.args,
            ),
            nkey=dedupe_key(
                run_id=self.run_id,
                node_id=node_id,
                step_index=position,
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
        self._lineages[owner_id] = owner_lineage
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

    def _drain_owner(self, branch: Branch) -> tuple[str, tuple[str, ...]]:
        """The branch id whose drain will dispatch an effect staged by ``branch``, and its lineage.

        Normally the branch itself. After :meth:`adopt`, a confirmed speculation's effects belong
        to the branch that will retire -- and so does anything the speculation stages *later*,
        because its task keeps running until it returns the ack it is waiting for. Following the
        map here is what stops a late stage from landing in a list no drain visits.

        The map is followed transitively and with a cycle guard. Adoption is a chain in principle
        (a confirmed speculation can itself be adopted), and a runtime that spun here would hang
        in a different place than the bug this method exists to close.
        """
        seen: set[str] = set()
        owner = branch.id
        lineage = branch.lineage
        while True:
            parent = self._adopted_into.get(owner)
            if parent is None or parent in seen:
                return owner, lineage
            seen.add(owner)
            owner = parent
            lineage = self._lineages.get(parent, lineage)

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

        ``stage_index`` is renumbered onto the end of the parent's list. The scheduler adopts a
        guess when the loop over the reply that confirmed it reaches its block, after the writes
        the model asked for ahead of it are staged and sent. The drain still sends by position
        (``step``), and the ledger checks the order effects left against position, not stage
        index, so neither depends on when the move happened.
        """
        # The record first, and the move only once it is on disk. Moved first, a failed append
        # -- a transient ``JournalBusy`` -- left the guess's write on the parent while the turn
        # settling it failed: the cleanup discarded the child, whose list was already empty,
        # and the next drain, a node that caught the failure and wrote something else, sent
        # it. If this raises, nothing has moved, and the caller discards the child's effects
        # where they are. An effect the child stages while it is being written moves with the
        # rest: its own ``effect_staged`` entry names it, and its dispatch names the branch
        # that sent it.
        held = self._staged.get(child.id, [])
        if held:
            await self.journal.append_async(
                self.run_id,
                "effect_adopted",
                {
                    "v": 1,
                    "branch_id": parent.id,
                    "from_branch_id": child.id,
                    "step": parent.cursor.step_index,
                    "effect_ids": [effect.id for effect in held],
                    "count": len(held),
                },
            )

        # No ``await`` from here on, so the child's task cannot stage in the middle of the move.
        #
        # Recorded with the move, and not only when there is something to move. ``adopt`` is a
        # point-in-time transfer, but the child's task may not have reached :meth:`stage` yet --
        # the model can emit the confirming block while the speculation is still awaiting its
        # own journal append. Anything it stages after this has to land where a drain will find
        # it, and :meth:`stage` reads this map to decide that. Recording the adoption only when
        # there was something to move left exactly that window open: the late effect went into
        # the child's list, no drain ever visits it, and the node body waits on an ack nobody
        # will ever complete. The run hangs with no error and no ``run_finished`` entry, which
        # is the worst of the three outcomes because nothing reports it.
        self._adopted_into[child.id] = parent.id
        self._lineages[parent.id] = parent.lineage

        # Its reads too, any not already moved (``adopt_reads``).
        self.adopt_reads(child, parent)

        moved = self._staged.pop(child.id, [])
        if not moved:
            return 0

        target = self._staged.setdefault(parent.id, [])
        adopted: list[StagedEffect] = [
            _reattributed(effect, parent, len(target) + offset)
            for offset, effect in enumerate(moved)
        ]
        target.extend(adopted)
        return len(adopted)

    def adopt_reads(self, child: Branch, parent: Branch) -> int:
        """Move the reads a confirmed guess has made, and not yet moved, to ``parent``.

        This is not bookkeeping. A confirmed speculation never retires, so ``validate_reads``
        never runs over it -- and the reads it made are by definition the ones issued on a
        guess, which are the only reads lattice rule E3 exists to re-check. Leaving them behind
        meant E3 validated the canonical branch's own reads (which need no validation by the
        module's own doctrine) and skipped the genuine guesses, so turning speculation *on*
        disabled the check that makes speculation safe.

        And not only when the guess's writes move (:meth:`adopt`). The scheduler moves a guess's
        reads as soon as the turn that confirmed it ends: moved with its writes, at the guess's
        place in the reply, they arrived after a write the model asked for earlier had parked
        the node and the check had run -- a stale read let a charge out with speculation on that
        speculation off refused. And again once the guess's own call returns, so a read still
        running when the turn ended lands on the branch as the same read issued early would.
        """
        reads = child.own_reads()
        already = self._reads_adopted.get(child.id, 0)
        fresh = list(reads[already:])
        parent.read_set.extend(fresh)
        self._reads_adopted[child.id] = len(reads)
        return len(fresh)

    def adopted_into(self, branch_id: str) -> str | None:
        """The branch a confirmed speculation's effects were moved to, if any."""
        return self._adopted_into.get(branch_id)

    def drain_owner_id(self, branch: Branch) -> str:
        """Which branch's drain will dispatch what ``branch`` stages, following adoption."""
        owner, _lineage = self._drain_owner(branch)
        return owner

    # -- discard ---------------------------------------------------------------------------

    def discard(self, branch: Branch) -> int:
        """Drop this branch's staged effects and every descendant's. Never dispatches.

        Returns the count. :meth:`discard_reporting` returns the effects themselves, which is
        what the journal entry needs -- naming them by re-deriving a list afterwards got the
        wrong ones.

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
        self._last_discarded = tuple(dropped)
        return len(dropped)

    def close(self, branch: Branch) -> None:
        """Refuse any further write from this branch, keeping what it has already staged.

        For a branch whose drain halted on a dead letter and whose node is about to be
        cancelled. A ``finally`` in that node that writes -- giving a lease back -- must be
        refused at once, not parked on an ack nothing will complete, or the wait for the
        cancelled node never returns. What the branch staged before stays unsent and unrevoked:
        it was authorised, and a resume re-derives it under the same key.
        """
        self._closed.add(branch.id)

    def discard_reporting(self, branch: Branch) -> tuple[StagedEffect, ...]:
        """Discard, and return exactly the effects that were dropped."""
        self.discard(branch)
        return self._last_discarded

    async def discard_and_journal(self, branch: Branch, reason: str) -> int:
        """Discard, and record in the journal exactly which effects were discarded.

        The ids used to be re-derived from ``staged_in_lineage(branch)``, which walks the
        branch's *lineage* and therefore lists every ancestor's staged effects first, and were
        then truncated with ``[:count]``. When the retiring parent still held staged entries --
        the staged list is not pruned after dispatch -- that slice took the parent's effects. So
        the durable record named writes that had reached the world as discarded, and the
        speculative write that really was thrown away unsent appeared in no discard entry at
        all. Both halves of that are wrong in the direction that matters: this journal is the
        only evidence there is about what did and did not happen.
        """
        dropped = self.discard_reporting(branch)
        dropped_ids = [effect.id for effect in dropped]
        count = len(dropped)
        if count:
            await self.journal.append_async(
                self.run_id,
                "effect_discarded",
                {
                    "v": 1,
                    "branch_id": branch.id,
                    "step": branch.cursor.step_index,
                    "effect_ids": dropped_ids,
                    "count": count,
                    "reason": reason,
                },
            )
        return count

    def pending(self, branch_id: str) -> tuple[StagedEffect, ...]:
        return tuple(self._staged.get(branch_id, ()))

    def delivered(self, branch_id: str) -> int:
        """How many of this branch's effects are known to be in the world."""
        return self._delivered.get(branch_id, 0)

    def branch_ids(self) -> tuple[str, ...]:
        """Every branch this buffer currently holds staged effects for."""
        return tuple(self._staged)

    def has_staged(self) -> bool:
        """Whether anything at all is held. Cheap, and does not expose the effects."""
        return any(self._staged.values())

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
        attempted: set[str] = set()

        async def still_held() -> None:
            # Before every attempt, not only before the claim: see Journal.check_run_lock.
            await self.journal.check_run_lock(self.run_id)

        # The list is read live and never snapshotted: completing one effect's ack can resume a
        # node body that stages the next write from the value it just received, and that write
        # has to be picked up by this same drain.
        #
        # Ordered by *program position*, not by insertion. Those are the same thing for a run
        # that never speculated, and differ for one that did: a confirmed speculation's effect
        # is adopted mid-stream, before the writes from blocks emitted *earlier* have been
        # staged at all. Dispatching by insertion order then put the later call into the world
        # first, so the order effects arrived depended on whether the runtime speculated --
        # which is Hard Rule 9 failing in the one dimension the ledger comparison is most
        # likely to be read for.
        #
        # This is deliberately not a sort by key or by effect id. Effect ids are ULIDs, so
        # sorting by one usually *reproduces* insertion order, which would make a reordering
        # bug invisible in testing and surface only under clock skew. ``step`` is the position
        # the model put the call at, which is the order being asserted.
        while True:
            live = self._staged.get(branch.id, [])
            pending = [
                effect for effect in live if effect.id not in settled and effect.id not in attempted
            ]
            if not pending:
                break
            effect = min(pending, key=lambda staged: (staged.step, staged.stage_index))
            attempted.add(effect.id)

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
                self._delivered[branch.id] = self._delivered.get(branch.id, 0) + 1
                self._complete_ack(effect.id, claim.ack)
                outcomes.append((effect.id, EffectOutcome.SKIPPED_DEDUPE))
                continue
            upstream_said_absent = False
            if claim.outcome is Claim.AMBIGUOUS and not effect.idempotent:
                # A previous attempt may already have taken effect and the tool has not said a
                # repeat is harmless. Guessing either way is worse than stopping -- unless the
                # upstream can be asked.
                verdict, why = await self._reconcile(dispatcher, effect)
                if verdict == "landed":
                    # Sent by a process that died before it heard back; the upstream says it
                    # took effect, and its own result is the ack.
                    settled.add(effect.id)
                    self._dispatch_seq[branch.id] = dispatch_index + 1
                    await self._settle_landed(
                        effect,
                        branch,
                        why,
                        claim.attempt,
                        dispatch_index,
                        authorised_by_offset,
                        confirmed_offset,
                    )
                    outcomes.append((effect.id, EffectOutcome.SKIPPED_DEDUPE))
                    continue
                if verdict == "unknown":
                    settled.add(effect.id)
                    await self._dead_letter(
                        effect,
                        branch,
                        attempts=claim.attempt,
                        error=f"ambiguous_after_crash: {why}",
                        authorised_by_offset=authorised_by_offset,
                        dispatch_index=dispatch_index,
                    )
                    outcomes.append((effect.id, EffectOutcome.DEAD_LETTER))
                    halted_at, ok = dispatch_index, False
                    continue
                # "absent": the upstream has no record of it, so it is sent now -- once.
                upstream_said_absent = True

            # An earlier process may have sent this and not heard back. Whatever happens now,
            # a dead letter must not claim nothing went out -- unless the upstream said so.
            earlier_may_have_landed = claim.outcome is Claim.AMBIGUOUS and not upstream_said_absent
            result = await dispatcher.dispatch(
                effect.call.name,
                dict(effect.call.args),
                idempotency_key=effect.nkey,
                branch_id=branch.id,
                before_attempt=still_held,
            )
            if not result.ok and result.sent == "maybe" and not effect.idempotent:
                # This process sent it and did not hear back: the lost reply a crash leaves,
                # without the crash. The dispatcher did not retry it, and neither does this --
                # it asks, as a resume would, and stops for a human if it cannot.
                verdict, why = await self._reconcile(dispatcher, effect)
                if verdict == "landed":
                    settled.add(effect.id)
                    self._dispatch_seq[branch.id] = dispatch_index + 1
                    await self._settle_landed(
                        effect,
                        branch,
                        why,
                        result.attempts,
                        dispatch_index,
                        authorised_by_offset,
                        confirmed_offset,
                    )
                    outcomes.append((effect.id, EffectOutcome.DISPATCHED))
                    continue
                if verdict == "absent":
                    # It did not take effect, so it is sent again -- once, and not asked about
                    # a second time: a second lost reply is a dead letter.
                    result = await dispatcher.dispatch(
                        effect.call.name,
                        dict(effect.call.args),
                        idempotency_key=effect.nkey,
                        branch_id=branch.id,
                        before_attempt=still_held,
                    )
                else:
                    result = replace(result, error=f"{result.error}; {why}")
            settled.add(effect.id)
            self._dispatch_seq[branch.id] = dispatch_index + 1
            if result.ok:
                self._delivered[branch.id] = self._delivered.get(branch.id, 0) + 1
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

            sent = "maybe" if earlier_may_have_landed or result.sent == "maybe" else "no"
            if sent == "no":
                await self.journal.mark_not_sent(self.run_id, effect.nkey, result.attempts)
            await self._dead_letter(
                effect,
                branch,
                attempts=result.attempts,
                error=result.error or "dispatch failed",
                authorised_by_offset=authorised_by_offset,
                dispatch_index=dispatch_index,
                sent=sent,
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

    async def _settle_landed(
        self,
        effect: StagedEffect,
        branch: Branch,
        ack: JsonValue,
        attempt: int,
        dispatch_index: int,
        authorised_by_offset: int,
        confirmed_offset: int,
    ) -> None:
        """Record an effect whose reply was lost as dispatched, on the upstream's word."""
        self._delivered[branch.id] = self._delivered.get(branch.id, 0) + 1
        await self.journal.settle_dispatch(
            run_id=self.run_id,
            nkey=effect.nkey,
            status="dispatched",
            ack=ack,
            attempt=attempt,
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
                "attempt": attempt,
                "deduped": True,
                "reconciled": True,
                "ack": ack,
                "dry_run": False,
                "compensation_for": None,
            },
        )
        self._complete_ack(effect.id, ack)

    async def _reconcile(
        self, dispatcher: Dispatcher, effect: StagedEffect
    ) -> tuple[str, JsonValue]:
        """Ask the upstream whether a call whose reply a crash lost took effect.

        ``("landed", ack)`` if it did, ``("absent", None)`` if it did not, and
        ``("unknown", reason)`` when there is no way to ask or the asking failed -- which is a
        dead letter, exactly as before this existed.
        """
        spec = dispatcher.registry.get(effect.call.name)
        if spec.reconcile is None:
            return "unknown", (
                "a previous attempt may have taken effect, and this tool neither declared itself "
                "idempotent nor says how to check"
            )
        try:
            landed = await spec.reconcile(effect.nkey, dict(effect.call.args))
        except Exception as exc:
            return "unknown", f"asking the upstream whether it took effect failed: {exc}"
        return ("landed", landed) if landed is not None else ("absent", None)

    async def _dead_letter(
        self,
        effect: StagedEffect,
        branch: Branch,
        *,
        attempts: int,
        error: str,
        authorised_by_offset: int,
        dispatch_index: int,
        sent: str = "maybe",
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
                # Where it was tried in the branch's send order, and its place in the branch's
                # list: without them the ledger took the index it was staged at, which for an
                # adopted guess is its index on the guess's own branch.
                "dispatch_index": dispatch_index,
                "stage_index": effect.stage_index,
                # "no" only when the request demonstrably never left this process. The claim
                # table forgets it once the dead letter settles; a resume deciding whether the
                # decision behind this effect still matters reads it here.
                "sent": sent,
            },
        )
        self._fail_ack(effect.id, error)


def _reattributed(effect: StagedEffect, parent: Branch, stage_index: int) -> StagedEffect:
    """An effect moved onto the branch that will retire it.

    ``branch_id`` and ``lineage`` follow the new owner because Hard Rule 3's audit asks that
    every effect in the world trace to a branch that *retired*, and the speculation never does.
    ``nkey`` deliberately does not move: it is the token the tool is handed, and an idempotency
    key that shifted when a guess turned out right would make a retry after adoption look to the
    upstream like a different call.
    """
    return replace(effect, branch_id=parent.id, lineage=parent.lineage, stage_index=stage_index)
