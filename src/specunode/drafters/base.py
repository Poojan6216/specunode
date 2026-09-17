"""Drafters: where predictions come from, and what they are allowed to be.

A drafter proposes; it never decides. Its output is a *guess* that the gate must confirm by
exact canonical equality against the model's real output before anything it caused can reach
the world. That separation is Hard Rule 1: the only models in the runtime are the target, whose
output is ground truth, and an optional draft model, whose output is only ever a candidate.

Three tiers, in increasing order of how much they can be wrong:

**Tier 0 — early issue.** Parses the target's own stream and proposes each ``tool_use`` block
the instant it finishes parsing, before the turn ends. These are calls the model has already
emitted, so acceptance is 1 by construction and the only thing being speculated is *time*.
This is Claude Code's streaming tool executor, credited as theirs.

**Tier 1 — pattern index.** An order-*k* Markov model over tool signatures mined from the run's
own journal, with argument templates that reference earlier tool outputs. This is PASTE's
mechanism, credited as theirs. It genuinely guesses, and its acceptance rate is measured rather
than assumed.

**Tier 2 — draft model.** A small model asked what comes next. Optional, off by default,
because a miss costs tokens as well as time.

A drafter returns candidates most-likely-first and never raises: an empty list means "no
opinion", which is the sequential case, and is how the whole runtime degrades to one code path
when nothing has anything to say.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from specunode.canonical import JsonValue
from specunode.core.decision import Decision

__all__ = ["DraftContext", "Drafter", "Prediction"]


@dataclass(frozen=True)
class DraftContext:
    """What a drafter is allowed to look at.

    Deliberately narrow, and deliberately *journaled* material only. A drafter that could see a
    sibling branch's state could make a prediction that only makes sense on that sibling's
    premise, and the gate would confirm it against a turn that never saw any of it.
    """

    run_id: str
    branch_id: str
    step_index: int
    node_id: str
    #: Tool calls already decided on this branch's lineage, oldest first. The pattern index
    #: keys off the tail of this.
    history: tuple[Decision, ...] = ()
    #: Results of those calls, by call index, for argument templates that reference them.
    results: Mapping[int, JsonValue] = field(default_factory=dict)
    #: Tool names the registry knows. A prediction outside this set is unusable.
    known_tools: frozenset[str] = frozenset()
    #: Blocks the target has already emitted in the in-flight turn, for tier 0.
    partial_turn: tuple[Decision, ...] = ()


@dataclass(frozen=True, slots=True)
class Prediction:
    """One candidate, and where it came from."""

    decision: Decision
    tier: Literal[0, 1, 2]
    #: The drafter's own confidence, for ranking only. Never a threshold the gate consults:
    #: confirmation is exact equality, and a confident wrong guess is still wrong.
    score: float = 1.0
    #: What producing this prediction cost, in model tokens. Zero for tier 0 and tier 1, which
    #: are free; the tier-2 draft model's usage otherwise. This is what ``wasted_tokens`` is
    #: fed when the prediction is squashed -- it was always fed 0, so ``max_wasted_tokens``
    #: was a limit nothing could ever reach.
    cost_tokens: int = 0


@runtime_checkable
class Drafter(Protocol):
    """Proposes zero or more candidate decisions, most likely first."""

    @property
    def tier(self) -> Literal[0, 1, 2]: ...

    async def predict(self, ctx: DraftContext) -> Sequence[Prediction]:
        """Candidates for the next decision. Never raises; empty means no opinion."""
        ...
