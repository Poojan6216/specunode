"""Idempotency keys and the dispatch dedupe protocol (spec task 1.3, Hard Rule 8)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specunode.buffer.idempotency import (
    EQUIV_DOMAIN,
    KEY_DOMAIN,
    dedupe_key,
    equivalence_key,
    idempotency_key,
    key_preimage,
)
from specunode.canonical import JsonValue
from specunode.journal.journal import Claim, Journal, JournalWriteError, PendingClaim

BASE: dict[str, object] = {
    "run_id": "01RUN",
    "lineage": ("br-1", "br-2"),
    "node_id": "agent#0",
    "step_index": 4,
    "tool_name": "charge_card",
    "args": {"customer_id": "cus-1", "amount": 10.0},
}


def key(**overrides: object) -> str:
    return idempotency_key(**{**BASE, **overrides})  # type: ignore[arg-type]


# -- derivation ------------------------------------------------------------------------------


def test_the_same_inputs_give_the_same_key() -> None:
    assert key() == key()


@pytest.mark.parametrize(
    ("component", "changed"),
    [
        ("run_id", "01OTHER"),
        ("lineage", ("br-1", "br-3")),
        ("node_id", "agent#1"),
        ("step_index", 5),
        ("tool_name", "refund_card"),
        ("args", {"customer_id": "cus-1", "amount": 11.0}),
    ],
)
def test_changing_any_component_changes_the_key(component: str, changed: object) -> None:
    assert key(**{component: changed}) != key()


def test_the_preimage_is_framed_not_concatenated() -> None:
    """("ab","c") and ("a","bc") concatenate identically; two effects must not share a key."""
    left = idempotency_key(**{**BASE, "node_id": "ab", "tool_name": "c"})  # type: ignore[arg-type]
    right = idempotency_key(**{**BASE, "node_id": "a", "tool_name": "bc"})  # type: ignore[arg-type]
    assert left != right


def test_argument_order_does_not_change_the_key() -> None:
    reordered = {"amount": 10.0, "customer_id": "cus-1"}
    assert key(args=reordered) == key()


def test_an_integer_amount_and_a_float_amount_are_different_effects() -> None:
    assert key(args={"customer_id": "cus-1", "amount": 10}) != key()


def test_the_three_derivations_differ_and_are_domain_separated() -> None:
    nkey = dedupe_key(
        run_id="01RUN",
        node_id="agent#0",
        step_index=4,
        tool_name="charge_card",
        args=BASE["args"],  # type: ignore[arg-type]
    )
    ekey = equivalence_key(
        node_id="agent#0",
        step_index=4,
        tool_name="charge_card",
        args=BASE["args"],  # type: ignore[arg-type]
    )
    assert len({key(), nkey, ekey}) == 3
    assert KEY_DOMAIN != EQUIV_DOMAIN


def test_dedupe_key_ignores_lineage_so_a_re_staged_effect_is_recognised() -> None:
    """After a stall, the sequential re-execution re-stages under a new branch id."""
    speculative = dedupe_key(
        run_id="01RUN", node_id="agent#0", step_index=4, tool_name="t", args={"a": 1}
    )
    re_staged = dedupe_key(
        run_id="01RUN", node_id="agent#0", step_index=4, tool_name="t", args={"a": 1}
    )
    assert speculative == re_staged
    assert key(lineage=("br-9",)) != key(lineage=("br-1",)), "the internal key still separates them"


def test_equivalence_key_ignores_the_run_so_two_runs_can_be_compared() -> None:
    a = equivalence_key(node_id="n", step_index=1, tool_name="t", args={})
    b = equivalence_key(node_id="n", step_index=1, tool_name="t", args={})
    assert a == b


def test_keys_are_stable_across_processes() -> None:
    """A key derived after a resume must equal the one derived before the crash."""
    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from specunode.buffer.idempotency import idempotency_key\n"
        "print(idempotency_key(run_id='01RUN', lineage=('br-1','br-2'), node_id='agent#0',"
        " step_index=4, tool_name='charge_card',"
        " args={'customer_id':'cus-1','amount':10.0}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=Path.cwd()
    )
    assert out.stdout.strip() == key()


@given(
    st.text(max_size=8),
    st.integers(min_value=0, max_value=10_000),
    st.dictionaries(st.text(max_size=5), st.integers(), max_size=4),
)
@settings(max_examples=500, deadline=None)
def test_the_preimage_is_canonical_bytes(node: str, step: int, args: dict[str, int]) -> None:
    once = key_preimage(
        run_id="r", lineage=(), node_id=node, step_index=step, tool_name="t", args=args
    )
    twice = key_preimage(
        run_id="r", lineage=(), node_id=node, step_index=step, tool_name="t", args=dict(args)
    )
    assert once == twice


# -- the dedupe protocol ----------------------------------------------------------------------


def pending(nkey: str = "nk-1", attempt: int = 1, effect_id: str | None = None) -> PendingClaim:
    # One effect id per key: the (run_id, effect_id) unique index enforces that an effect is
    # keyed exactly one way, so a fixture that reuses an id across keys is the bug, not the index.
    return PendingClaim(
        run_id="01RUN",
        nkey=nkey,
        idem_key="ik-1",
        effect_id=effect_id or f"ef-{nkey}",
        branch_id="br-1",
        tool="charge_card",
        attempt=attempt,
    )


def dispatched_payload(**extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "v": 1,
        "effect_id": "ef-nk-1",
        "branch_id": "br-1",
        "nkey": "nk-1",
        "stage_index": 0,
        "dispatch_index": 0,
        "authorised_by_offset": 3,
        "deduped": False,
        **extra,
    }


async def test_a_fresh_claim_is_owned(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    claim = await journal.claim_dispatch(pending())
    assert claim.outcome is Claim.OWNED and claim.may_send


async def test_a_second_claim_on_an_acked_effect_sends_nothing(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending())
    await journal.settle_dispatch(
        run_id="01RUN",
        nkey="nk-1",
        status="dispatched",
        ack={"id": "chg-1"},
        attempt=1,
        kind="effect_dispatched",
        payload=dispatched_payload(),
    )
    again = await journal.claim_dispatch(pending())
    assert again.outcome is Claim.ALREADY_DISPATCHED
    assert not again.may_send
    assert again.ack == {"id": "chg-1"}


async def test_the_ack_and_its_journal_entry_land_together(tmp_path: Path) -> None:
    """One transaction. Split, a crash between them makes a resume re-charge a card."""
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending())
    offset = await journal.settle_dispatch(
        run_id="01RUN",
        nkey="nk-1",
        status="dispatched",
        ack={"ok": True},
        attempt=1,
        kind="effect_dispatched",
        payload=dispatched_payload(),
    )
    entries = list(journal.read("01RUN"))
    assert [e.kind for e in entries] == ["effect_dispatched"]
    assert entries[0].offset == offset
    assert journal.verify_chain("01RUN").ok


async def test_a_failure_that_never_left_the_process_is_safe_to_retry(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending())
    await journal.mark_not_sent("01RUN", "nk-1", attempt=1)
    again = await journal.claim_dispatch(pending(attempt=2))
    assert again.outcome is Claim.RETRY_SAFE and again.may_send


async def test_an_unresolved_claim_after_a_crash_is_ambiguous_not_safe(tmp_path: Path) -> None:
    """The two-generals boundary. Calling it safe here is how a card gets charged twice."""
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending())
    again = await journal.claim_dispatch(pending(attempt=2))
    assert again.outcome is Claim.AMBIGUOUS
    assert not again.may_send


async def test_a_dead_lettered_effect_is_retried_by_a_resume(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending())
    await journal.settle_dispatch(
        run_id="01RUN",
        nkey="nk-1",
        status="dead_letter",
        ack=None,
        attempt=5,
        kind="effect_dead_lettered",
        payload={
            "v": 1,
            "effect_id": "ef-nk-1",
            "branch_id": "br-1",
            "nkey": "nk-1",
            "attempts": 5,
            "last_error": {"type": "Partitioned", "message": "unreachable"},
        },
    )
    again = await journal.claim_dispatch(pending(attempt=6))
    assert again.outcome is Claim.RETRY_SAFE


async def test_unresolved_dispatches_are_the_resume_reconciliation_list(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending(nkey="nk-1"))
    await journal.claim_dispatch(pending(nkey="nk-2"))
    await journal.settle_dispatch(
        run_id="01RUN",
        nkey="nk-2",
        status="dispatched",
        ack={},
        attempt=1,
        kind="effect_dispatched",
        payload=dispatched_payload(nkey="nk-2", effect_id="ef-nk-2"),
    )
    unresolved = journal.unresolved_dispatches("01RUN")
    assert [row["nkey"] for row in unresolved] == ["nk-1"]


async def test_two_different_effects_do_not_share_a_claim(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    assert (await journal.claim_dispatch(pending(nkey="a"))).outcome is Claim.OWNED
    assert (await journal.claim_dispatch(pending(nkey="b"))).outcome is Claim.OWNED


async def test_one_effect_keyed_two_ways_is_refused(tmp_path: Path) -> None:
    """An effect with two keys could be dispatched twice; the index makes that impossible."""
    journal = Journal(tmp_path / "j.db")
    await journal.claim_dispatch(pending(nkey="a", effect_id="ef-shared"))
    with pytest.raises(JournalWriteError, match="one effect is one idempotency key"):
        await journal.claim_dispatch(pending(nkey="b", effect_id="ef-shared"))
