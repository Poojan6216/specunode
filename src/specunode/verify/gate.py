"""The gate: does the model's real decision match what a branch guessed?

Hard Rule 4, and it is one line of logic guarded by a lot of reasoning.

A speculation is confirmed **iff** ``canonical(ŷ) == canonical(y)``. Never semantic
similarity, never fuzzy argument matching, never a model asked whether two calls are "the
same". The reason is not purity: a branch that guessed *ŷ* has already executed reads and
staged writes on that premise, so if *y* differs in any argument -- a different ticket id, a
different amount, a different path -- every downstream call it made was computed on a premise
that turned out to be false. There is no useful notion of "close enough" for a tool call, and
any tolerance is a way for a wrong branch's effects to retire.

Free text never confirms, including against itself. That asymmetry is what makes a free-text
node a barrier rather than a speculation that trivially succeeds.
"""

from __future__ import annotations

from specunode.core.branch import Branch, BranchStatus
from specunode.core.decision import Decision, decisions_equal, is_barrier

__all__ = ["GateError", "resolve", "resolve_decision"]


class GateError(RuntimeError):
    """The gate was asked to resolve something that is not a speculation."""


def resolve_decision(predicted: Decision, actual: Decision) -> BranchStatus:
    """CONFIRMED iff the two are canonically identical, SQUASHED otherwise. Pure."""
    if is_barrier(predicted) or is_barrier(actual):
        return BranchStatus.SQUASHED
    return BranchStatus.CONFIRMED if decisions_equal(predicted, actual) else BranchStatus.SQUASHED


def resolve(branch: Branch, actual: Decision) -> BranchStatus:
    """Resolve one branch against the model's real decision.

    A branch with no prediction is the canonical path, not a speculation: it is confirmed
    directly by the turn it owns and never passes through here. Asking the gate to resolve one
    is a bug in the scheduler, so it raises rather than quietly confirming -- a gate that says
    CONFIRMED for a branch that predicted nothing is a gate that confirms everything.
    """
    if branch.predicted is None:
        raise GateError(
            f"branch {branch.id} predicted nothing, so there is nothing to resolve. The "
            "canonical branch is confirmed by the turn it owns, not by the gate."
        )
    return resolve_decision(branch.predicted, actual)
