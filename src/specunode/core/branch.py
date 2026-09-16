"""Branches: speculative continuations of the graph, and the only thing that retires.

A branch is not a git branch and not a conditional edge. It is everything the runtime does on
the assumption that the model will decide *ŷ* at step *i*, before the model has produced *y*.

Branches fork copy-on-write, share nothing with siblings, and end in exactly one of retired,
squashed or stalled. There is no merge, ever: two speculative siblings are mutually exclusive
alternatives, at most one of them is right, and combining a decision the model made with one
it did not make is not a merge.

``lineage`` is the chain of branch ids from the last retirement to this branch, inclusive. It
resets at each retirement, so ``len(lineage) - 1`` is the speculation *depth* rather than the
length of the run -- which is what ``max_speculation_depth`` is meant to bound, and what keeps
the idempotency key preimage from growing without limit on a long run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from specunode.canonical import JsonValue
from specunode.core.decision import Decision
from specunode.core.model import Message
from specunode.core.state import BranchState

__all__ = [
    "Branch",
    "BranchClosed",
    "BranchStatus",
    "ReadRecord",
    "ResultSlot",
    "SlotStatus",
    "StepCursor",
    "TurnFrame",
]


class BranchClosed(RuntimeError):
    """A staging or state operation was attempted on a branch that is no longer running.

    Raised rather than silently resurrecting the branch's buffer. A tool that could not be
    cancelled can land after its branch was squashed, carrying the right branch id; letting
    it stage would put an effect into a buffer nothing will ever drain -- or worse, into one
    that a later operation does.
    """


class BranchStatus(Enum):
    """Where a branch is in its life. Exactly one terminal state is reached."""

    #: Running on a prediction, unresolved.
    SPECULATIVE = "speculative"
    #: The model's real decision matched, and it is journaled. Retirement is in progress.
    CONFIRMED = "confirmed"
    #: Store buffer drained, delta applied. Terminal.
    RETIRED = "retired"
    #: The prediction was wrong, or an ancestor was squashed. Buffer discarded, never
    #: dispatched. Terminal.
    SQUASHED = "squashed"
    #: A hazard. The runtime proceeds sequentially from the last confirmed decision.
    STALLED = "stalled"

    @property
    def terminal(self) -> bool:
        return self in (BranchStatus.RETIRED, BranchStatus.SQUASHED, BranchStatus.STALLED)

    @property
    def may_dispatch(self) -> bool:
        """Only a CONFIRMED branch may drain, and only after its confirming entry is durable."""
        return self is BranchStatus.CONFIRMED


@dataclass(frozen=True, slots=True)
class StepCursor:
    """The run's program position, forked by value and promoted only at retirement.

    Two siblings forked at the same point see the *same* ``step_index`` for their k-th action:
    they are alternative occupants of one program slot, and their idempotency keys differ only
    in lineage. That is what makes Hard Rule 9 hold -- if a discarded speculation shifted the
    counter, every later key in a speculative run would differ from the sequential run's and
    the ledgers could never normalise equal.

    ``visits`` counts prior canonical entries into each structural node, so a loop that
    revisits a node gets distinct keys while a replay reproduces them exactly. A squashed or
    stalled visit costs nothing, because the cursor that counted it dies with its branch.
    """

    step_index: int = 0
    visits: tuple[tuple[str, int], ...] = ()

    def advance(self, by: int = 1) -> StepCursor:
        return replace(self, step_index=self.step_index + by)

    def visit(self, structural_id: str) -> tuple[StepCursor, str]:
        """Return the cursor with this node's visit counted, and the node_id to use."""
        counts = dict(self.visits)
        index = counts.get(structural_id, 0)
        counts[structural_id] = index + 1
        return replace(self, visits=tuple(sorted(counts.items()))), f"{structural_id}#{index}"

    def peek(self, structural_id: str) -> str:
        return f"{structural_id}#{dict(self.visits).get(structural_id, 0)}"


@dataclass(frozen=True, slots=True)
class ReadRecord:
    """One read a branch performed, kept so retirement can check it went stale (rule E3)."""

    tool: str
    #: The arguments, kept so retirement can re-issue the read against its witness. Held on
    #: the record rather than in a lookup keyed by hash: a module-level cache would outlive
    #: the run, grow without bound, and let one run's arguments answer another's question.
    args: Mapping[str, JsonValue]
    args_hash: str
    result_hash: str
    witness: JsonValue
    at_step: int
    #: Whether this read was issued while the branch was still a guess. Only those are
    #: revalidated at retirement: a read a CONFIRMED branch issued happened *after* the
    #: model's decision was already durable, so there is no speculation to invalidate -- and
    #: revalidating it turns a workload with a competing writer into a livelock, on the
    #: sequential arm, where validation is meaningless.
    issued_while_speculative: bool = True
    #: Whether this read overlapped an ancestor's drain. Reported in the ledger rather than
    #: blocked: blocking would cost exactly the latency past-write speculation exists to buy,
    #: to close a race that can only arise from under-declared forward_keys -- which is
    #: already a named developer-side trust boundary.
    raced_drain: bool = False

    @property
    def witnessed(self) -> bool:
        """A read with no witness cannot be validated, and is reported so -- never as fresh."""
        return self.witness is not None


class SlotStatus(Enum):
    """Where a tool result slot is. A staged slot never fills, and that is not a bug."""

    PENDING = "pending"
    FILLED = "filled"
    #: The call was staged, so there is no result and there never will be on this branch.
    #: Any request that would have to include this slot is refused by Hard Rule 13's gate.
    STAGED = "staged"


@dataclass
class ResultSlot:
    """One tool result's place in a turn, allocated when the call was *requested*.

    Preallocated at parse time and filled by ordinal, so program order is structural rather
    than a sort. A sort would leave a window in which an unsorted list could be serialised,
    and appending on completion produces completion order -- which is the bug spec task 3.8
    plants.
    """

    ordinal: int
    tool_use_id: str
    call: Decision | None = None
    status: SlotStatus = SlotStatus.PENDING
    content: JsonValue = None
    handle: str | None = None
    journal_offset: int | None = None
    future: object = None


@dataclass
class TurnFrame:
    """One assistant turn and the result slots its tool calls will fill."""

    step: int
    slots: list[ResultSlot] = field(default_factory=list)
    complete: bool = False

    def slot(self, ordinal: int) -> ResultSlot:
        return self.slots[ordinal]

    @property
    def pending(self) -> list[ResultSlot]:
        return [s for s in self.slots if s.status is SlotStatus.PENDING]

    @property
    def staged(self) -> list[ResultSlot]:
        return [s for s in self.slots if s.status is SlotStatus.STAGED]


@dataclass
class Branch:
    """A speculative continuation of the graph."""

    id: str
    parent_id: str | None = None
    fork_step: int = 0
    predicted: Decision | None = None
    #: Branch ids from the last retirement to self, inclusive. Never empty on a running
    #: branch; reset at each retirement so it measures depth, not run length.
    lineage: tuple[str, ...] = ()
    status: BranchStatus = BranchStatus.SPECULATIVE
    reason: str | None = None
    cursor: StepCursor = field(default_factory=StepCursor)
    #: The branch's working copy. A BranchState rather than a plain dict so that the fork is
    #: genuinely copy-on-write and the retirement delta is measured from the fork point -- a
    #: shared dict is how Hard Rule 6 gets violated by accident.
    state: BranchState = field(default_factory=BranchState)
    context: list[Message] = field(default_factory=list)
    read_set: list[ReadRecord] = field(default_factory=list)
    #: (step, request_hash) for every *target* request this branch sent. Hard Rule 13
    #: rebuilds each one at retirement and compares.
    prompts_sent: list[tuple[int, str]] = field(default_factory=list)
    #: Set only by the retirement-time context check. A branch that sent a request while
    #: speculating cannot drain until this is true (Hard Rule 13).
    context_verified: bool = False
    #: How many requests this branch sent while it was still a guess. Zero means Rule 13 has
    #: nothing to check: every request it sent carried real values.
    speculative_prompts: int = 0
    #: Tokens spent on this branch, which are wasted if it squashes (Hard Rule 10).
    tokens: int = 0
    tier: int | None = None
    #: The structural node this branch is executing, for key derivation and the ledger.
    node_id: str = ""
    #: Offset of the ``branch_resolved{confirmed}`` entry. Nothing dispatches before this is
    #: durable, and a resume resurrects a branch only when this entry exists (Hard Rule 3).
    confirmed_offset: int | None = None
    #: Set at R1: no further forks at this step, resolution is under way.
    frozen: bool = False
    #: Turn frames, in program order. Never sorted, never appended out of order.
    frames: list[TurnFrame] = field(default_factory=list)
    #: Child branch ids, so a squash can cascade without consulting a global index.
    children: list[str] = field(default_factory=list)
    #: Tool calls executed after this branch's first staged write. This -- not
    #: max_speculation_depth -- is what past-write speculation actually buys, because a branch
    #: can never make another model call after staging, so the benchmark reports it by name.
    post_write_span: int = 0
    #: The asyncio task running this branch. Cancellation is the squash primitive.
    task: object = None

    def __post_init__(self) -> None:
        if not self.lineage:
            self.lineage = (self.id,)

    @property
    def depth(self) -> int:
        """Speculation depth. ``max_speculation_depth`` bounds this."""
        return len(self.lineage) - 1

    @property
    def running(self) -> bool:
        return not self.status.terminal

    def fork(
        self, child_id: str, *, predicted: Decision, step: int, tier: int | None = None
    ) -> Branch:
        """A child continuation. Shares nothing mutable with this branch or its siblings."""
        if not self.running:
            raise BranchClosed(f"branch {self.id} is {self.status.value} and cannot fork")
        return Branch(
            id=child_id,
            parent_id=self.id,
            fork_step=step,
            predicted=predicted,
            lineage=(*self.lineage, child_id),
            status=BranchStatus.SPECULATIVE,
            cursor=self.cursor,
            # Copy-on-write at the level that matters: a child must never be able to reach a
            # parent's mutable container and change what the parent (or a sibling) sees.
            state=self.state.fork(),
            context=list(self.context),
            read_set=list(self.read_set),
            tier=tier,
        )

    def advance_step(self) -> int:
        """Take the next program position for a call this branch is about to make.

        Every port call takes one, not just every model-emitted block. A node body can issue
        two calls the model never separately emitted -- a fan-out, a retry loop, two writes in
        one node -- and if they shared a step index, two identical calls would derive an
        identical idempotency key, dedupe would suppress the second, and an effect the
        sequential run performed would never reach the world. None of the three mandatory
        tests can see that: the leak invariant is a subset over branch ids and cannot see a
        *missing* effect, and both equivalence arms would collide identically.
        """
        self.cursor = self.cursor.advance()
        return self.cursor.step_index

    def record_prompt(self, step: int, request_hash: str) -> None:
        self.prompts_sent.append((step, request_hash))
        if self.status is BranchStatus.SPECULATIVE:
            self.speculative_prompts += 1

    def squash(self, reason: str) -> None:
        self.status = BranchStatus.SQUASHED
        self.reason = reason

    def stall(self, reason: str) -> None:
        self.status = BranchStatus.STALLED
        self.reason = reason

    def confirm(self) -> None:
        self.status = BranchStatus.CONFIRMED

    def retire(self) -> None:
        if self.status is not BranchStatus.CONFIRMED:
            raise BranchClosed(
                f"branch {self.id} is {self.status.value}; only a CONFIRMED branch retires "
                "(Hard Rule 3)"
            )
        self.status = BranchStatus.RETIRED

    def is_ancestor_of(self, other: Branch) -> bool:
        return self.id in other.lineage[:-1]

    def unwitnessed_reads(self) -> Sequence[ReadRecord]:
        return [record for record in self.read_set if not record.witnessed]

    def reads_to_validate(self) -> Sequence[ReadRecord]:
        """Reads issued while this branch was still a guess -- the only ones E3 re-checks."""
        return [record for record in self.read_set if record.issued_while_speculative]

    def has_staged_slot(self) -> bool:
        """Whether any turn is waiting on a result that will never arrive."""
        return any(frame.staged for frame in self.frames)
