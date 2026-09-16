"""The gate (spec task 3.4, Hard Rule 4).

One line of logic and a lot of reasoning behind it. A branch that guessed *ŷ* has already run
reads and staged writes on that premise; if the model's real decision differs in any argument,
every downstream call it made was computed from something that turned out to be false. So the
relation is exact canonical equality, and these tests are mostly about the ways a looser one
would let a wrong branch retire.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specunode.core.branch import Branch, BranchStatus
from specunode.core.decision import FreeText, Route, Structured, ToolCall
from specunode.verify.gate import GateError, resolve, resolve_decision

EXAMPLES = settings(max_examples=1000, deadline=None)

json_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=8),
)
decisions = st.one_of(
    st.builds(ToolCall, st.text(max_size=8), st.dictionaries(st.text(max_size=5), json_leaves)),
    st.builds(Route, st.text(max_size=8)),
    st.builds(Structured, st.text(max_size=8), json_leaves),
)


@given(decisions)
@EXAMPLES
def test_a_decision_confirms_against_itself(decision: object) -> None:
    assert resolve_decision(decision, decision) is BranchStatus.CONFIRMED  # type: ignore[arg-type]


@given(decisions, decisions)
@EXAMPLES
def test_different_decisions_squash(a: object, b: object) -> None:
    from specunode.core.decision import decision_key

    if decision_key(a) != decision_key(b):  # type: ignore[arg-type]
        assert resolve_decision(a, b) is BranchStatus.SQUASHED  # type: ignore[arg-type]


def test_an_argument_that_differs_by_one_value_squashes() -> None:
    """Every downstream call the branch made was computed from the wrong premise."""
    predicted = ToolCall("charge_card", {"customer_id": "cus-1", "amount": 25.0})
    actual = ToolCall("charge_card", {"customer_id": "cus-1", "amount": 24.0})
    assert resolve_decision(predicted, actual) is BranchStatus.SQUASHED


def test_an_integer_amount_does_not_confirm_against_a_float_one() -> None:
    """Python says 1 == 1.0. A payment API handed one did not receive the other."""
    assert (
        resolve_decision(
            ToolCall("charge_card", {"amount": 1}), ToolCall("charge_card", {"amount": 1.0})
        )
        is BranchStatus.SQUASHED
    )


def test_argument_order_does_not_matter() -> None:
    """The same call written two ways is the same call."""
    assert (
        resolve_decision(ToolCall("t", {"a": 1, "b": 2}), ToolCall("t", {"b": 2, "a": 1}))
        is BranchStatus.CONFIRMED
    )


def test_a_different_tool_with_the_same_arguments_squashes() -> None:
    assert (
        resolve_decision(ToolCall("refund", {"id": "x"}), ToolCall("charge", {"id": "x"}))
        is BranchStatus.SQUASHED
    )


def test_a_decision_never_confirms_against_another_kind() -> None:
    assert resolve_decision(Route("go"), Structured("s", "go")) is BranchStatus.SQUASHED


@given(st.text(max_size=20))
@EXAMPLES
def test_free_text_never_confirms_even_against_itself(text: str) -> None:
    """The asymmetry that makes a prose node a barrier rather than a free win."""
    prose = FreeText.of(text)
    assert resolve_decision(prose, prose) is BranchStatus.SQUASHED
    assert resolve_decision(prose, FreeText.of(text)) is BranchStatus.SQUASHED


def test_free_text_on_either_side_squashes() -> None:
    assert resolve_decision(FreeText.of("x"), ToolCall("t", {})) is BranchStatus.SQUASHED
    assert resolve_decision(ToolCall("t", {}), FreeText.of("x")) is BranchStatus.SQUASHED


# -- resolving a branch --------------------------------------------------------------------


def test_a_branch_resolves_on_its_prediction() -> None:
    call = ToolCall("restart_job", {"job_id": "etl-1"})
    branch = Branch(id="br-1", predicted=call)
    assert resolve(branch, call) is BranchStatus.CONFIRMED
    assert resolve(branch, ToolCall("restart_job", {"job_id": "etl-2"})) is BranchStatus.SQUASHED


def test_the_canonical_branch_is_refused_rather_than_confirmed() -> None:
    """A gate that confirms a branch which predicted nothing is a gate that confirms anything.

    The canonical branch is confirmed by the turn it owns, never by comparison, so asking the
    gate about one is a scheduler bug and is reported as one.
    """
    branch = Branch(id="br-canon", predicted=None)
    with pytest.raises(GateError, match="predicted nothing"):
        resolve(branch, ToolCall("t", {}))
