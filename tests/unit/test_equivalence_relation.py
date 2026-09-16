"""The equivalence relation (Hard Rule 9, spec task 5.1 pulled forward).

Rule 9 says the effect ledger of a speculative run equals the ledger of the same journaled run
with speculation off. The danger in implementing that is not subtle and it cuts both ways: a
relation that strips too much passes for two runs that did nothing alike, and a relation that
strips too little fails on bookkeeping that exists only because speculation happened.

So these tests are written in two halves. One half asserts the relation HOLDS across the
differences that are supposed to be invisible -- branch ids, effect ids, run ids, retire
sequence. The other asserts it FAILS on each difference that matters, including the two planted
bugs the spec names by hand: retiring a squashed branch, and reversing dispatch order. A
relation that passed the second half would be worse than no relation, because the mandatory
test built on it would be green forever.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from specunode.buffer.idempotency import dedupe_key
from specunode.canonical import JsonValue
from specunode.core.decision import ToolCall
from specunode.verify.equivalence import (
    EquivalenceError,
    assert_equivalent,
    equivalence_digest,
    ledger_matches_world,
    normalise_cross_adapter,
    normalise_for_equivalence,
    normalise_world_mutations,
    normalised_rows,
)


@dataclass(frozen=True)
class Row:
    """A ledger row, structurally satisfying LedgerRowLike."""

    call: ToolCall
    node_id: str = "agent#0"
    step_index: int = 1
    authorised_by_step: int = 0
    status: str = "DISPATCHED"
    nkey: str = "nk"
    retire_seq: int = 0
    dispatch_index: int = 0


@dataclass(frozen=True)
class Led:
    rows: tuple[Row, ...] = ()
    terminal: bool = True
    discarded_effects: int = 0


@dataclass(frozen=True)
class Mut:
    sequence: int
    effect_key: str
    tool: str
    args_hash: str


def row(
    tool: str,
    args: dict[str, JsonValue],
    *,
    step: int = 1,
    seq: int = 0,
    index: int = 0,
    **kw: object,
) -> Row:
    call = ToolCall(tool, args)
    return Row(
        call=call,
        step_index=step,
        retire_seq=seq,
        dispatch_index=index,
        nkey=str(
            dedupe_key(run_id="r", node_id="agent#0", step_index=step, tool_name=tool, args=args)
        ),
        **kw,  # type: ignore[arg-type]
    )


def world_for(rows: tuple[Row, ...]) -> list[Mut]:
    from specunode.canonical import chash

    return [
        Mut(
            sequence=index,
            effect_key=r.nkey,
            tool=r.call.name,
            args_hash=chash(dict(r.call.args)),
        )
        for index, r in enumerate(rows)
        if r.status == "DISPATCHED"
    ]


#: Two effects, as an ordinary support-agent step produces.
CHARGE = row("charge_card", {"customer_id": "cus-1", "amount": 10.0}, step=1, seq=0, index=0)
RECEIPT = row(
    "send_receipt", {"customer_id": "cus-1", "charge_id": "chg-1"}, step=2, seq=1, index=0
)
BASE = (CHARGE, RECEIPT)


# -- the relation holds across what is meant to be invisible ---------------------------------


def test_the_same_effects_normalise_equal() -> None:
    assert normalise_for_equivalence(Led(BASE)) == normalise_for_equivalence(Led(BASE))


def test_branch_ids_effect_ids_and_run_ids_are_invisible() -> None:
    """The two arms are different runs by construction; only what reached the world compares."""
    speculative = tuple(replace(r, nkey="a-different-run-entirely") for r in BASE)
    assert normalise_for_equivalence(Led(BASE)) == normalise_for_equivalence(Led(speculative))


def test_a_sequential_and_a_speculative_arm_of_one_workload_agree() -> None:
    """The relation must be weak enough not to fail on speculation's own bookkeeping."""
    assert_equivalent(
        Led(BASE),
        Led(BASE, discarded_effects=3),
        world_for(BASE),
        world_for(BASE),
        expect_rows=2,
        min_squashed_with_staged=1,
    )


# -- the relation fails on every difference that matters ---------------------------------------


def test_an_extra_row_fails_the_relation() -> None:
    """Planted bug: retire on SQUASHED. The squashed branch's effect appears in the on arm."""
    leaked = row("charge_card", {"customer_id": "cus-1", "amount": 10.0}, step=1, seq=0, index=1)
    with pytest.raises(EquivalenceError, match="row"):
        assert_equivalent(
            Led(BASE),
            Led((*BASE, leaked)),
            world_for(BASE),
            world_for((*BASE, leaked)),
            expect_rows=2,
        )


def test_reversing_dispatch_order_fails_the_relation() -> None:
    """Planted bug: dispatch order reversed.

    Both effects retire together, so the only thing that separates them is the order they left.
    A relation ordered by stage_index -- recorded when the effect was *staged* -- would sort the
    reversal back into stage order and pass, which is why the ordering key is dispatch_index.
    """
    first = replace(CHARGE, retire_seq=0, dispatch_index=0)
    second = replace(RECEIPT, retire_seq=0, dispatch_index=1)
    forward = (first, second)
    reversed_drain = (replace(first, dispatch_index=1), replace(second, dispatch_index=0))
    assert normalise_for_equivalence(Led(forward)) != normalise_for_equivalence(Led(reversed_drain))


def test_an_integer_argument_is_not_a_float_argument() -> None:
    """A billing API handed {"amount": 1} did not receive {"amount": 1.0}."""
    integral = (row("charge_card", {"customer_id": "cus-1", "amount": 10}),)
    fractional = (row("charge_card", {"customer_id": "cus-1", "amount": 10.0}),)
    assert normalise_for_equivalence(Led(integral)) != normalise_for_equivalence(Led(fractional))


def test_a_different_tool_fails_the_relation() -> None:
    other = (replace(CHARGE, call=ToolCall("refund_card", CHARGE.call.args)), RECEIPT)
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(other))


def test_a_different_authorising_step_fails_the_relation() -> None:
    """Which model decision authorised an effect is the question the ledger exists to answer."""
    reauthorised = (replace(CHARGE, authorised_by_step=7), RECEIPT)
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(reauthorised))


def test_a_dead_letter_is_not_a_dispatch() -> None:
    failed = (replace(CHARGE, status="DEAD_LETTER"), RECEIPT)
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(failed))


def test_the_same_call_at_a_different_step_fails_the_relation() -> None:
    """step_index survives inside the rewritten key, so a drifting counter is still caught."""
    moved = (replace(CHARGE, step_index=9), RECEIPT)
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(moved))


def test_the_same_call_at_a_different_node_fails_the_relation() -> None:
    """A loop iteration attributed to the wrong node is a different effect."""
    elsewhere = (replace(CHARGE, node_id="other#0"), RECEIPT)
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(elsewhere))


# -- the relation is not vacuous ------------------------------------------------------------------


def test_two_empty_ledgers_are_refused_rather_than_compared() -> None:
    """The failure mode this anchor exists for: a runtime that dispatched nothing passes."""
    with pytest.raises(EquivalenceError, match="were not compared"):
        assert_equivalent(Led(()), Led(()), [], [], expect_rows=2)


def test_an_unfinished_run_is_refused() -> None:
    """An unfinished run's ledger is a prefix, and a prefix that matches proves nothing."""
    with pytest.raises(EquivalenceError, match="terminal"):
        assert_equivalent(
            Led(BASE, terminal=False), Led(BASE), world_for(BASE), world_for(BASE), expect_rows=2
        )


def test_a_speculative_arm_that_never_squashed_anything_is_refused() -> None:
    """Zero discarded effects means the store buffer was never asked to hold anything back."""
    with pytest.raises(EquivalenceError, match="never exercised"):
        assert_equivalent(
            Led(BASE),
            Led(BASE, discarded_effects=0),
            world_for(BASE),
            world_for(BASE),
            expect_rows=2,
            min_squashed_with_staged=1,
        )


def test_genuinely_different_runs_do_not_normalise_equal() -> None:
    other = (row("post_summary", {"channel": "#ops", "text": "hi"}),)
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(other))
    assert equivalence_digest(Led(BASE)) != equivalence_digest(Led(other))


# -- the ledger and the world are checked against each other, not assumed to agree --------------


def test_a_ledger_that_matches_its_world_passes_the_join() -> None:
    assert ledger_matches_world(Led(BASE), world_for(BASE))


def test_an_effect_that_reached_the_world_with_no_row_is_caught() -> None:
    """The leak shape, seen from the other side: the world moved and nothing claims authority."""
    extra = [
        *world_for(BASE),
        Mut(sequence=9, effect_key="nk-unknown", tool="charge_card", args_hash="h"),
    ]
    assert not ledger_matches_world(Led(BASE), extra)
    with pytest.raises(EquivalenceError, match="does not match its world"):
        assert_equivalent(Led(BASE), Led(BASE), world_for(BASE), extra, expect_rows=2)


def test_a_row_claiming_an_effect_the_world_never_saw_is_caught() -> None:
    assert not ledger_matches_world(Led(BASE), world_for(BASE)[:1])


def test_the_two_arms_worlds_must_agree_too() -> None:
    """Two ledgers can agree while the worlds differ if the join is not checked on both sides."""
    assert normalise_world_mutations(world_for(BASE)) == normalise_world_mutations(world_for(BASE))
    swapped = list(reversed(world_for(BASE)))
    for index, mutation in enumerate(swapped):
        swapped[index] = replace(mutation, sequence=index)
    assert normalise_world_mutations(world_for(BASE)) != normalise_world_mutations(swapped)


# -- the cross-adapter relation is weaker, and is never used by Rule 9's own test -----------------


def test_the_cross_adapter_relation_ignores_the_node() -> None:
    """A generic MCP client reports no node; the proxy gate compares without one."""
    elsewhere = (replace(CHARGE, node_id="somewhere-else#3"), RECEIPT)
    assert normalise_cross_adapter(Led(BASE)) == normalise_cross_adapter(Led(elsewhere))
    assert normalise_for_equivalence(Led(BASE)) != normalise_for_equivalence(Led(elsewhere))


def test_the_two_relations_are_labelled_so_they_cannot_be_confused() -> None:
    """A weaker comparison must not be able to masquerade as Hard Rule 9's own."""
    assert normalise_cross_adapter(Led(BASE)) != normalise_for_equivalence(Led(BASE))


def test_normalised_rows_are_inspectable_for_a_failure_message() -> None:
    rows = normalised_rows(Led(BASE))
    assert len(rows) == 2
    assert rows[0]["call"] == {"kind": "tool_call", "name": "charge_card", "args": CHARGE.call.args}
