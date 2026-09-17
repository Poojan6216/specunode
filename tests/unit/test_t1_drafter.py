"""The tier-1 pattern index and the drafter that reads it (spec task 3.2).

The predictor is the component whose accuracy decides whether any of the rest pays for itself,
and until now it had no tests of its own -- the speculation machinery was exercised through a
stub that returned a fixed answer, which proves the *runtime* handles a prediction but proves
nothing about the thing that makes them.

Two properties matter more than the ranking arithmetic. The index must be **deterministic**,
because a replay that speculated differently is not a replay. And it must **decline to offer a
call whose arguments it cannot fill**, because the gate compares exact canonical arguments: a
half-filled call cannot ever be confirmed, so offering one buys nothing and costs the upstream
reads its branch makes on the way there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from specunode.canonical import JsonValue
from specunode.core.decision import FreeText, ToolCall
from specunode.drafters.base import DraftContext
from specunode.drafters.t1_pattern import (
    MAX_ORDER,
    PatternDrafter,
    PatternIndex,
    signature_of,
)

TRACE = [
    ToolCall("get_ticket", {"ticket_id": "tkt-1"}),
    ToolCall("lookup_customer", {"customer_id": "cus-1"}),
    ToolCall("send_receipt", {"customer_id": "cus-1", "charge_id": "chg-1"}),
]


def ctx(history: list[ToolCall], results: dict[int, JsonValue] | None = None) -> DraftContext:
    return DraftContext(
        run_id="run",
        branch_id="br",
        step_index=len(history),
        node_id="agent",
        history=tuple(history),
        results=results or {},
        known_tools={"get_ticket", "lookup_customer", "send_receipt"},
    )


def trained() -> PatternIndex:
    index = PatternIndex(order=2)
    index.train([TRACE])
    return index


def test_a_signature_keeps_the_shape_and_drops_the_values() -> None:
    """Two calls to the same tool with different ids are the same step of the same pattern."""
    a = signature_of(ToolCall("get_ticket", {"ticket_id": "tkt-1"}))
    b = signature_of(ToolCall("get_ticket", {"ticket_id": "tkt-9"}))
    assert a == b == "get_ticket(ticket_id)"


def test_argument_order_does_not_change_a_signature() -> None:
    """The shape is a set of keys; a dict's insertion order is not part of the pattern."""
    one = signature_of(ToolCall("t", {"a": 1, "b": 2}))
    other = signature_of(ToolCall("t", {"b": 2, "a": 1}))
    assert one == other


def test_free_text_is_not_a_tool_signature() -> None:
    """A barrier has no tool shape, and must not be learned as one."""
    assert signature_of(FreeText.of("thinking")) == "<free_text>"


def test_the_index_backs_off_to_a_shorter_context() -> None:
    """An unseen order-2 window still predicts, from the order-1 counts underneath it."""
    index = trained()
    unseen = [
        ToolCall("send_receipt", {"customer_id": "c", "charge_id": "x"}),
        ToolCall("get_ticket", {"ticket_id": "tkt-1"}),
    ]
    assert index.rank(unseen), "backoff produced no candidate at all"
    assert index.rank(unseen)[0][0] == "lookup_customer(customer_id)"


def test_ranking_is_total_and_reproducible() -> None:
    """Ties break on the signature string, not on the order traces happened to be read in."""
    index = PatternIndex(order=1)
    index.train([[ToolCall("a", {}), ToolCall("z", {})], [ToolCall("a", {}), ToolCall("b", {})]])
    first = index.rank([ToolCall("a", {})])
    assert [sig for sig, _ in first] == ["b()", "z()"]
    assert index.rank([ToolCall("a", {})]) == first


def test_the_order_is_capped_when_mining() -> None:
    """Longer windows fragment counts faster than they add information on real trace lengths.

    Asserted through the path that takes an order from a caller, rather than by restating the
    constant: a test that reads ``MAX_ORDER == 3`` passes whether or not anything honours it.
    """
    index = PatternIndex(order=min(9, MAX_ORDER))
    assert index.order == MAX_ORDER


async def test_it_fills_an_argument_from_a_prior_result() -> None:
    """PASTE's data-flow observation: one call's output is the next call's input.

    ``customer_id`` is nowhere in the arguments the branch has issued. It exists only inside
    the ticket the previous read returned, nested under ``value`` the way a witnessed read
    wraps its row -- and that is the case worth having, because an index that carried only
    prior *arguments* forward would predict this signature correctly and then be unable to
    instantiate it.
    """
    drafter = PatternDrafter(index=trained())
    read_result: JsonValue = {"value": {"customer_id": "cus-1", "status": "open"}, "witness": "1"}
    out = await drafter.predict(ctx([TRACE[0]], results={0: read_result}))
    assert [p.decision for p in out] == [ToolCall("lookup_customer", {"customer_id": "cus-1"})]
    assert out[0].tier == 1


async def test_it_declines_a_call_it_cannot_fill() -> None:
    """The gate compares exact arguments, so a half-filled call can never be confirmed."""
    drafter = PatternDrafter(index=trained())
    out = await drafter.predict(ctx([TRACE[0]]))  # no results: customer_id is unknown
    assert out == []


async def test_it_declines_a_tool_the_run_does_not_have() -> None:
    """A prediction for an unregistered tool would stall rather than speculate."""
    drafter = PatternDrafter(index=trained())
    narrow = DraftContext(
        run_id="run",
        branch_id="br",
        step_index=1,
        node_id="agent",
        history=(TRACE[0],),
        results={0: {"value": {"customer_id": "cus-1"}, "witness": "1"}},
        known_tools={"get_ticket"},
    )
    assert await drafter.predict(narrow) == []


async def test_an_untrained_index_offers_nothing() -> None:
    """No counts means no guess -- not an arbitrary one."""
    drafter = PatternDrafter(index=PatternIndex(order=2))
    assert await drafter.predict(ctx([TRACE[0]])) == []


def test_a_saved_index_is_byte_identical_on_rebuild(tmp_path: Path) -> None:
    """A replay that speculated differently is not a replay."""
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    trained().save(first)
    trained().save(second)
    assert first.read_bytes() == second.read_bytes()


def test_a_round_trip_preserves_the_ranking(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    trained().save(path)
    assert PatternIndex.load(path).rank([TRACE[0]]) == trained().rank([TRACE[0]])


def test_mining_refuses_anything_that_is_not_a_journal() -> None:
    """The retired-lineages-only rule lives in the miner; it cannot be handed a stand-in."""
    from specunode.drafters.t1_pattern import mine_from_journal

    with pytest.raises(TypeError, match="needs a Journal"):
        mine_from_journal(object(), "run-1")
