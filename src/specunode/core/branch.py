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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from specunode.canonical import JsonValue
from specunode.core.decision import Decision
from specunode.core.model import Message

__all__ = [
    "Branch",
    "BranchClosed",
    "BranchStatus",
    "ReadRecord",
    "StepCursor",
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
    args_hash: str
    result_hash: str
    witness: JsonValue
    at_step: int
    speculative: bool

    @property
    def witnessed(self) -> bool:
        """A read with no witness cannot be validated, and is reported so -- never as fresh."""
        return self.witness is not None


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
    state: dict[str, JsonValue] = field(default_factory=dict)
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
    #: True once this branch consumed a projected value, which bars every later model call.
    projected: bool = False
    #: Tokens spent on this branch, which are wasted if it squashes (Hard Rule 10).
    tokens: int = 0
    tier: int | None = None

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
            state=dict(self.state),
            context=list(self.context),
            read_set=list(self.read_set),
            tier=tier,
        )

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
