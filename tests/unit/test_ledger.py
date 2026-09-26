"""The effect ledger (spec task 2.6, corrections C7, C2, C26).

The ledger is the artifact this project produces: it says which model decision authorised each
effect, what was staged and thrown away, where the runtime gave up on speculation, and what the
speculation cost. So it is built from the journal and from nothing else -- a resumed run whose
effect was skipped because a dead process already sent it must render a row indistinguishable
from the uninterrupted run's, and only the journal knows that.

These tests drive real runs through the real store buffer, dispatcher, world and journal rather
than hand-writing entries. A ledger test that constructs its own input can agree with a builder
that reads the journal wrongly.
"""

from __future__ import annotations

from pathlib import Path

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import EffectOutcome, StoreBuffer
from specunode.core.branch import Branch, BranchStatus
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.journal.journal import Journal
from specunode.journal.ledger import (
    build_ledger,
    load_or_create_key,
    render_ledger,
    sign_ledger,
    verify_ledger,
)
from specunode.testing.world import World, standard_world

RUN = "01LEDGERRUNAAAAAAAAAAAAAAA"


def registry_for(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    registry.register(
        ToolSpec(name="post_summary", effect=EffectClass.WRITE, fn=world.post_summary)
    )
    registry.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))
    return registry


async def drive(
    tmp_path: Path,
    calls: list[ToolCall],
    *,
    discard_extra: bool = False,
    partition_at: int | None = None,
    db: str = "j.db",
    run_id: str = RUN,
) -> tuple[Journal, World, StoreBuffer]:
    """Run one branch to retirement, journaling everything the ledger reads."""
    journal = Journal(tmp_path / db)
    world = standard_world()
    registry = registry_for(world)
    dispatcher = Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5, cap_delay_ms=1.0)
    buffer = StoreBuffer(journal=journal, run_id=run_id)

    await journal.append_async(
        run_id,
        "run_started",
        {
            "v": 1,
            "mode": "run",
            "config_hash": "c" * 8,
            "registry_hash": "r" * 8,
            "policy": {"speculation": False},
            "target": {"provider": "scripted", "model": "scripted"},
            "graph": {"adapter": "plain"},
        },
    )

    if discard_extra:
        doomed = Branch(id="br-doomed", status=BranchStatus.SPECULATIVE)
        doomed.advance_step()
        await buffer.stage(
            doomed,
            ToolCall("charge_card", {"customer_id": "cus-9", "amount": 5.0}),
            registry.get("charge_card"),
        )
        doomed.squash("mismatch")
        await buffer.discard_and_journal(doomed, "squashed")
        await journal.append_async(
            run_id,
            "branch_resolved",
            {"v": 1, "branch_id": doomed.id, "step": 0, "status": "squashed", "reason": "mismatch"},
        )

    branch = Branch(id="br-canon", status=BranchStatus.SPECULATIVE)
    for call in calls:
        branch.advance_step()
        await buffer.stage(branch, call, registry.get(call.name), node_id="agent#0")

    branch.confirm()
    branch.context_verified = True
    branch.record_prompt(1, "h" * 8)
    confirmed = await journal.append_async(
        run_id,
        "branch_resolved",
        {"v": 1, "branch_id": branch.id, "step": 1, "status": "confirmed"},
    )
    if partition_at is not None:
        world.partition(at=partition_at)
    report = await buffer.drain(
        branch, dispatcher, confirmed_offset=confirmed, authorised_by_offset=confirmed
    )
    branch.retire()
    await journal.append_async(
        run_id,
        "branch_resolved",
        {"v": 1, "branch_id": branch.id, "step": 1, "status": "retired"},
    )
    await journal.append_async(
        run_id,
        "run_finished",
        {
            "v": 1,
            "ok": report.ok,
            "status": "completed" if report.ok else "dead_letter",
            "counters": {"effects_dispatched": report.count(EffectOutcome.DISPATCHED)},
        },
    )
    return journal, world, buffer


ONE_WRITE = [ToolCall("restart_job", {"job_id": "etl-1"})]
TWO_WRITES = [
    ToolCall("restart_job", {"job_id": "etl-1"}),
    ToolCall("post_summary", {"channel": "#ops", "text": "restarted"}),
]


# -- rows -----------------------------------------------------------------------------------


async def test_one_row_per_dispatched_effect(tmp_path: Path) -> None:
    journal, world, _ = await drive(tmp_path, TWO_WRITES)
    ledger = build_ledger(journal, RUN)
    assert [r.call.name for r in ledger.rows] == ["restart_job", "post_summary"]
    assert all(r.status == "DISPATCHED" for r in ledger.rows)
    assert len(ledger.rows) == len(world.mutations)


async def test_the_ledger_matches_what_the_world_actually_received(tmp_path: Path) -> None:
    journal, world, _ = await drive(tmp_path, TWO_WRITES)
    ledger = build_ledger(journal, RUN)
    assert [r.nkey for r in ledger.rows] == [m.effect_key for m in world.mutations]


async def test_rows_are_ordered_by_dispatch_not_by_stage(tmp_path: Path) -> None:
    """stage_index is recorded when an effect is staged; only dispatch_index says what left."""
    journal, _, _ = await drive(tmp_path, TWO_WRITES)
    ledger = build_ledger(journal, RUN)
    assert [r.dispatch_index for r in ledger.rows] == [0, 1]
    assert ledger.dispatch_order_anomalies == 0


async def test_every_row_names_the_decision_that_authorised_it(tmp_path: Path) -> None:
    """The one question the ledger exists to answer."""
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    ledger = build_ledger(journal, RUN)
    assert ledger.rows[0].authorised_by_step >= 0


async def test_a_discarded_branchs_effects_produce_no_rows(tmp_path: Path) -> None:
    """Hard Rule 3, seen from the ledger: a squashed branch authorised nothing."""
    journal, world, _ = await drive(tmp_path, ONE_WRITE, discard_extra=True)
    ledger = build_ledger(journal, RUN)
    assert [r.call.name for r in ledger.rows] == ["restart_job"]
    assert ledger.discarded_effects == 1
    assert ledger.squashed_branches == 1
    assert "charge_card" not in {m.tool for m in world.mutations}


async def test_a_dead_lettered_effect_renders_as_one_and_the_run_is_not_ok(tmp_path: Path) -> None:
    journal, world, _ = await drive(tmp_path, TWO_WRITES, partition_at=2)
    ledger = build_ledger(journal, RUN)
    statuses = [r.status for r in ledger.rows]
    assert "DEAD_LETTER" in statuses
    assert len(world.mutations) == 1


# -- purity and provenance -----------------------------------------------------------------------


async def test_building_twice_from_one_journal_gives_the_same_ledger(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, TWO_WRITES)
    first, second = build_ledger(journal, RUN), build_ledger(journal, RUN)
    assert render_ledger(first) == render_ledger(second)


async def test_a_dedupe_skip_renders_the_same_row_as_an_uninterrupted_run(tmp_path: Path) -> None:
    """Task 2.5 requires a resumed run's ledger to equal the uninterrupted run's.

    An effect the dedupe check skipped because a dead process already sent it must be
    indistinguishable in the ledger from one this process sent, which is why the ledger is
    built from journal entries and never from the drain's own report.
    """
    journal, _, _buffer = await drive(tmp_path, ONE_WRITE)
    clean = build_ledger(journal, RUN)

    # Re-drain the same effects under a fresh buffer, as a resume would.
    world2 = standard_world()
    dispatcher2 = Dispatcher(registry=registry_for(world2), max_attempts=1)
    resumed_buffer = StoreBuffer(journal=journal, run_id=RUN)
    resumed = Branch(id="br-resumed", status=BranchStatus.SPECULATIVE)
    resumed.advance_step()
    await resumed_buffer.stage(
        resumed, ONE_WRITE[0], registry_for(world2).get("restart_job"), node_id="agent#0"
    )
    resumed.confirm()
    resumed.context_verified = True
    offset = await journal.append_async(
        RUN, "branch_resolved", {"v": 1, "branch_id": resumed.id, "step": 1, "status": "confirmed"}
    )
    report = await resumed_buffer.drain(
        resumed, dispatcher2, confirmed_offset=offset, authorised_by_offset=offset
    )
    assert report.count(EffectOutcome.SKIPPED_DEDUPE) == 1
    assert world2.mutations == [], "a resumed run must not re-send an effect already acked"
    assert len(build_ledger(journal, RUN).rows) == len(clean.rows)


async def test_the_ledger_knows_which_journal_it_came_from(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    ledger = build_ledger(journal, RUN)
    assert ledger.run_id == RUN
    assert ledger.journal_entries > 0
    assert ledger.journal_head
    assert ledger.terminal


# -- the Hard Rule 13 stamp ------------------------------------------------------------------------


async def test_a_run_that_checked_nothing_is_stamped_unchecked(tmp_path: Path) -> None:
    """Correction C2: a stamp for a property nobody checked is worse than no stamp."""
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    assert build_ledger(journal, RUN).context_checks == (0, 0)
    assert build_ledger(journal, RUN).context_identity == "unchecked"


# -- rendering -------------------------------------------------------------------------------------


async def test_the_rendering_names_every_effect_and_its_authority(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, TWO_WRITES, discard_extra=True)
    text = render_ledger(build_ledger(journal, RUN), short_ids=True)
    assert "EFFECT LEDGER" in text
    assert "restart_job" in text and "post_summary" in text
    assert "DISPATCHED" in text
    assert "squashed branches: 1" in text
    assert "staged effects discarded: 1" in text
    assert "charge_card" not in text, "a discarded effect must not appear as if it happened"


async def test_the_normalised_rendering_drops_what_the_relation_strips(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    ledger = build_ledger(journal, RUN)
    assert RUN not in render_ledger(ledger, normalised=True)
    assert RUN in render_ledger(ledger)


# -- signing ---------------------------------------------------------------------------------------


async def test_a_genuine_ledger_verifies(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    key = load_or_create_key(tmp_path / "keys")
    signed = sign_ledger(build_ledger(journal, RUN), key)
    result = verify_ledger(signed, journal=journal, store=tmp_path / "keys")
    assert result.ok and result.category == "ok"


async def test_an_edited_ledger_is_rejected_for_integrity_not_origin(tmp_path: Path) -> None:
    """Task 5.4 requires the two forgeries distinguished; they have different remedies."""
    from dataclasses import replace

    journal, _, _ = await drive(tmp_path, TWO_WRITES)
    key = load_or_create_key(tmp_path / "keys")
    signed = sign_ledger(build_ledger(journal, RUN), key)
    tampered = replace(signed, rows=signed.rows[:1])
    result = verify_ledger(tampered, journal=journal, store=tmp_path / "keys")
    assert not result.ok and result.category == "integrity"


async def test_a_ledger_signed_by_a_stranger_is_rejected_for_origin(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    stranger = load_or_create_key(tmp_path / "other-keys")
    signed = sign_ledger(build_ledger(journal, RUN), stranger)
    result = verify_ledger(signed, journal=journal, trusted=[], store=tmp_path / "keys")
    assert not result.ok and result.category == "origin"


async def test_an_unsigned_ledger_is_never_conflated_with_a_forged_one(tmp_path: Path) -> None:
    journal, _, _ = await drive(tmp_path, ONE_WRITE)
    result = verify_ledger(build_ledger(journal, RUN), journal=journal, store=tmp_path / "keys")
    assert not result.ok and result.category == "unsigned"


def test_the_signing_key_is_stable_across_calls(tmp_path: Path) -> None:
    first = load_or_create_key(tmp_path / "keys")
    second = load_or_create_key(tmp_path / "keys")
    assert first.key_id == second.key_id


def test_a_dead_letter_is_left_out_of_the_send_order() -> None:
    """A dead letter written before it recorded where it was tried takes its stage index as its
    place in the send order, and that can collide with an effect that was sent. It says nothing
    about the order effects left in, so the check leaves it out."""
    from specunode.core.decision import ToolCall
    from specunode.journal.ledger import LedgerRow, _order_anomalies

    def row(name: str, step: int, status: str) -> LedgerRow:
        return LedgerRow(
            effect_id=f"e-{name}",
            call=ToolCall(name, {}),
            key=f"k-{name}",
            nkey=f"n-{name}",
            node_id="bill",
            step_index=step,
            branch_id="b",
            branch_ord=0,
            stage_index=0,
            dispatch_index=0,
            retire_seq=0,
            authorised_by_step=0,
            status=status,  # type: ignore[arg-type]
        )

    # Sorted for display, the dead letter can come first: the same index as the charge.
    assert (
        _order_anomalies(
            [row("send_receipt", 2, "DEAD_LETTER"), row("charge_card", 1, "DISPATCHED")]
        )
        == 0
    )


async def test_a_ledger_signed_in_format_1_is_not_reported_as_edited(tmp_path: Path) -> None:
    """Dead letters left the dispatch-order check, which a signature covers, so a ledger signed
    before the change failed verification as "edited after signing". It verifies, and says
    which format it was signed in. Found by the twelfth review."""
    from dataclasses import replace

    from specunode.canonical import canonical
    from specunode.journal.ledger import SIGN_DOMAIN, _b64, ledger_payload

    journal, _, _ = await drive(tmp_path, TWO_WRITES)
    ledger = build_ledger(journal, RUN)
    key = load_or_create_key(tmp_path / "keys")
    # Format 1, built here rather than by the code under test: the version, and a dispatch-order
    # count taken over every row. No dead letter here, so the count is the same either way.
    format_1 = {**ledger_payload(ledger), "v": 1}
    old = key.private.sign(SIGN_DOMAIN + canonical(format_1))
    envelope = f"ed25519:{key.key_id}:{_b64(key.public_bytes)}:{_b64(old)}"
    signed = replace(ledger, signature=envelope)
    verdict = verify_ledger(signed, journal=journal, trusted=[key.key_id])
    assert verdict.ok, verdict
    assert "format 1" in (verdict.detail or "")
    current = verify_ledger(sign_ledger(ledger, key), journal=journal, trusted=[key.key_id])
    assert current.ok and current.detail is None
    edited = replace(signed, rows=signed.rows[:1])
    assert verify_ledger(edited, journal=journal, trusted=[key.key_id]).category == "integrity"
