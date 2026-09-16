"""The store buffer (spec task 1.4) and the dispatcher (1.5).

The claim under test is the one the project is named for: a write staged by a branch that has
not retired does not reach the world, and a write staged by one that has reaches it exactly
once, in stage order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from specunode.buffer.dispatcher import Dispatcher, ToolDispatchError
from specunode.buffer.store_buffer import (
    EffectOutcome,
    ForwardHazard,
    ForwardMiss,
    StoreBuffer,
)
from specunode.core.branch import Branch, BranchClosed, BranchStatus
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec, forward_keys_from_template
from specunode.core.hazards import HazardViolation, handle_for
from specunode.journal.journal import Journal, PendingClaim
from specunode.testing.faults import Partitioned
from specunode.testing.world import World, standard_world

RUN = "01BUFRUNAAAAAAAAAAAAAAAAAA"


def build(tmp_path: Path) -> tuple[StoreBuffer, Journal, World, Dispatcher]:
    journal = Journal(tmp_path / "j.db")
    world = standard_world()
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="restart_job",
            effect=EffectClass.WRITE,
            fn=world.restart_job,
            idempotent=True,
            forward_keys=forward_keys_from_template("job:{args.job_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="charge_card",
            effect=EffectClass.WRITE,
            fn=world.charge_card,
            forward_keys=forward_keys_from_template("customer:{args.customer_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="post_summary",
            effect=EffectClass.WRITE,
            fn=world.post_summary,
            forward_keys=forward_keys_from_template("channel:{args.channel}"),
        )
    )
    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=world.get_pipeline_status,
            witness=True,
            forward_keys=forward_keys_from_template("job:{args.pipeline_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="fetch_runbook",
            effect=EffectClass.READ,
            fn=world.fetch_runbook,
            forward_keys=forward_keys_from_template("doc:{args.section}"),
        )
    )
    return (
        StoreBuffer(journal=journal, run_id=RUN),
        journal,
        world,
        Dispatcher(registry=registry, base_delay_ms=1.0, cap_delay_ms=5.0),
    )


def speculative(branch_id: str = "br-1", parent: Branch | None = None) -> Branch:
    if parent is not None:
        return parent.fork(branch_id, predicted=ToolCall("x", {}), step=1)
    return Branch(id=branch_id, status=BranchStatus.SPECULATIVE)


def registry_of(dispatcher: Dispatcher, name: str) -> ToolSpec:
    return dispatcher.registry.get(name)


# -- staging ---------------------------------------------------------------------------------


async def test_staging_a_write_touches_nothing(tmp_path: Path) -> None:
    buffer, _, world, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    assert world.mutations == []
    assert len(buffer.pending(branch.id)) == 1


async def test_a_staged_write_returns_a_handle_not_a_value(tmp_path: Path) -> None:
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    effect = await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    assert effect.placeholder == handle_for(effect.id)


async def test_the_canonical_path_mints_no_handle(tmp_path: Path) -> None:
    """On the canonical path the caller gets the real ack, so there is nothing to stand in for."""
    buffer, _, _, dispatcher = build(tmp_path)
    branch = Branch(id="br-canon", status=BranchStatus.CONFIRMED)
    effect = await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    assert effect.placeholder is None


async def test_staging_journals_the_effect(tmp_path: Path) -> None:
    buffer, journal, _, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    kinds = [e.kind for e in journal.read(RUN)]
    assert kinds == ["effect_staged"]


async def test_staging_a_call_containing_a_handle_is_refused(tmp_path: Path) -> None:
    """Hazard analysis should have stalled first; this is the defence behind it."""
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    with pytest.raises(HazardViolation, match="RETURN_VALUE_DEPENDENCY"):
        await buffer.stage(
            branch,
            ToolCall("post_summary", {"channel": handle_for("01ABC"), "text": "x"}),
            registry_of(dispatcher, "post_summary"),
        )


async def test_staging_on_a_squashed_branch_is_refused(tmp_path: Path) -> None:
    """An uncancellable tool can land after its branch is gone, carrying the right id."""
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    branch.squash("mismatch")
    with pytest.raises(BranchClosed):
        await buffer.stage(
            branch,
            ToolCall("restart_job", {"job_id": "etl-1"}),
            registry_of(dispatcher, "restart_job"),
        )


async def test_stage_index_counts_in_order(tmp_path: Path) -> None:
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    for job in ("etl-1", "etl-2", "etl-3"):
        await buffer.stage(
            branch, ToolCall("restart_job", {"job_id": job}), registry_of(dispatcher, "restart_job")
        )
    assert [e.stage_index for e in buffer.pending(branch.id)] == [0, 1, 2]


# -- isolation (Hard Rule 6) ---------------------------------------------------------------------


async def test_a_branch_sees_its_ancestors_staged_writes(tmp_path: Path) -> None:
    buffer, _, _, dispatcher = build(tmp_path)
    parent = speculative("br-parent")
    await buffer.stage(
        parent, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    child = speculative("br-child", parent=parent)
    await buffer.stage(
        child, ToolCall("restart_job", {"job_id": "etl-2"}), registry_of(dispatcher, "restart_job")
    )
    assert len(buffer.staged_in_lineage(child)) == 2


async def test_a_branch_never_sees_a_siblings_staged_write(tmp_path: Path) -> None:
    buffer, _, _, dispatcher = build(tmp_path)
    parent = speculative("br-parent")
    left = speculative("br-left", parent=parent)
    right = speculative("br-right", parent=parent)
    await buffer.stage(
        left,
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 1.0}),
        registry_of(dispatcher, "charge_card"),
    )
    assert buffer.staged_in_lineage(right) == ()


# -- forwarding -----------------------------------------------------------------------------------


async def test_a_read_of_an_unrelated_key_is_a_miss(tmp_path: Path) -> None:
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    result = buffer.forward(
        branch,
        ToolCall("fetch_runbook", {"section": "restart"}),
        registry_of(dispatcher, "fetch_runbook"),
    )
    assert isinstance(result, ForwardMiss)


async def test_a_read_of_a_staged_writes_key_is_a_hazard(tmp_path: Path) -> None:
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    result = buffer.forward(
        branch,
        ToolCall("get_pipeline_status", {"pipeline_id": "etl-1"}),
        registry_of(dispatcher, "get_pipeline_status"),
    )
    assert isinstance(result, ForwardHazard)


async def test_an_undeclared_read_after_any_staged_write_is_a_hazard(tmp_path: Path) -> None:
    """Fails closed. This is the one hazard class with no backstop at retirement."""
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    undeclared = ToolSpec(name="list_open_tickets", effect=EffectClass.READ)
    result = buffer.forward(branch, ToolCall("list_open_tickets", {}), undeclared)
    assert isinstance(result, ForwardHazard)
    assert "forward_keys" in result.reason


# -- discard --------------------------------------------------------------------------------------


async def test_discard_drops_everything_and_dispatches_nothing(tmp_path: Path) -> None:
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    for job in ("etl-1", "etl-2"):
        await buffer.stage(
            branch, ToolCall("restart_job", {"job_id": job}), registry_of(dispatcher, "restart_job")
        )
    branch.squash("mismatch")
    assert await buffer.discard_and_journal(branch, "squashed") == 2
    assert world.mutations == []
    discards = [e for e in journal.read(RUN) if e.kind == "effect_discarded"]
    assert len(discards) == 1 and discards[0].payload["count"] == 2


async def test_discard_cascades_to_descendants(tmp_path: Path) -> None:
    buffer, _, world, dispatcher = build(tmp_path)
    parent = speculative("br-parent")
    child = speculative("br-child", parent=parent)
    await buffer.stage(
        parent, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    await buffer.stage(
        child, ToolCall("restart_job", {"job_id": "etl-2"}), registry_of(dispatcher, "restart_job")
    )
    assert buffer.discard(parent) == 2
    assert world.mutations == []


async def test_discard_is_idempotent(tmp_path: Path) -> None:
    """Otherwise the ledger's discarded count doubles and Demo 1 prints a number no run made."""
    buffer, _, _, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    assert buffer.discard(branch) == 1
    assert buffer.discard(branch) == 0


# -- drain (task 1.4) and dead-lettering (task 1.5) -------------------------------------------


async def confirm(journal: Journal, branch: Branch) -> int:
    """Journal the entry that confirms a branch, and return its offset (Hard Rule 3)."""
    branch.confirm()
    branch.context_verified = True
    return await journal.append_async(
        RUN,
        "branch_resolved",
        {"v": 1, "branch_id": branch.id, "step": 0, "status": "confirmed"},
    )


async def test_draining_a_confirmed_branch_dispatches_each_effect_once_in_stage_order(
    tmp_path: Path,
) -> None:
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    for job in ("etl-1", "etl-2", "etl-3"):
        await buffer.stage(
            branch, ToolCall("restart_job", {"job_id": job}), registry_of(dispatcher, "restart_job")
        )
    offset = await confirm(journal, branch)

    report = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    assert report.ok
    assert report.count(EffectOutcome.DISPATCHED) == 3
    assert [m.tool for m in world.mutations] == ["restart_job"] * 3
    assert [m.row_id for m in world.mutations] == ["etl-1", "etl-2", "etl-3"]


async def test_draining_twice_sends_nothing_twice(tmp_path: Path) -> None:
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    offset = await confirm(journal, branch)
    await buffer.drain(branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset)
    second = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    assert second.count(EffectOutcome.SKIPPED_DEDUPE) == 1
    assert len(world.mutations) == 1


async def test_a_speculative_branch_cannot_drain(tmp_path: Path) -> None:
    buffer, _journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    with pytest.raises(BranchClosed, match="Hard Rule 3"):
        await buffer.drain(branch, dispatcher, confirmed_offset=0, authorised_by_offset=0)
    assert world.mutations == []


async def test_draining_before_the_confirming_entry_is_durable_is_refused(tmp_path: Path) -> None:
    """Task 1.6 plants exactly this bug, so the precondition is checked inside drain."""
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    branch.confirm()
    branch.context_verified = True
    head = journal.last_offset(RUN) or 0
    with pytest.raises(BranchClosed, match="durable"):
        await buffer.drain(
            branch, dispatcher, confirmed_offset=head + 5, authorised_by_offset=head + 5
        )
    assert world.mutations == []


async def test_a_branch_that_speculated_a_prompt_cannot_drain_unverified(tmp_path: Path) -> None:
    """Hard Rule 13: nothing downstream of an unverified request reaches the world."""
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    branch.record_prompt(0, "some-hash")
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    branch.confirm()
    offset = await journal.append_async(
        RUN, "branch_resolved", {"v": 1, "branch_id": branch.id, "step": 0, "status": "confirmed"}
    )
    with pytest.raises(BranchClosed, match="Hard Rule 13"):
        await buffer.drain(branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset)
    assert world.mutations == []


async def test_a_partition_mid_drain_dead_letters_and_halts(tmp_path: Path) -> None:
    """Task 1.5's Verify: 1 dispatched, 2 dead-lettered after retries, 3 not attempted."""
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    for job in ("etl-1", "etl-2", "etl-3"):
        await buffer.stage(
            branch, ToolCall("restart_job", {"job_id": job}), registry_of(dispatcher, "restart_job")
        )
    offset = await confirm(journal, branch)
    world.partition(at=2)

    report = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    assert not report.ok
    outcomes = [outcome for _, outcome in report.outcomes]
    assert outcomes == [
        EffectOutcome.DISPATCHED,
        EffectOutcome.DEAD_LETTER,
        EffectOutcome.NOT_ATTEMPTED,
    ]
    assert [m.row_id for m in world.mutations] == ["etl-1"]
    assert [e.kind for e in journal.read(RUN)].count("effect_dead_lettered") == 1


async def test_resuming_after_the_partition_heals_sends_the_rest_and_no_duplicate(
    tmp_path: Path,
) -> None:
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    calls = [ToolCall("restart_job", {"job_id": job}) for job in ("etl-1", "etl-2", "etl-3")]
    for call in calls:
        await buffer.stage(branch, call, registry_of(dispatcher, "restart_job"))
    offset = await confirm(journal, branch)
    world.partition(at=2)
    await buffer.drain(branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset)
    world.heal()

    # A resume is a fresh buffer over the same journal, re-staging under a new branch id --
    # which is exactly why the dedupe key drops the lineage.
    resumed_buffer = StoreBuffer(journal=journal, run_id=RUN)
    resumed = speculative("br-after-resume")
    for call in calls:
        await resumed_buffer.stage(resumed, call, registry_of(dispatcher, "restart_job"))
    offset2 = await confirm(journal, resumed)
    report = await resumed_buffer.drain(
        resumed, dispatcher, confirmed_offset=offset2, authorised_by_offset=offset2
    )

    assert report.ok
    assert report.count(EffectOutcome.SKIPPED_DEDUPE) == 1, "etl-1 already went; it must not repeat"
    assert report.count(EffectOutcome.DISPATCHED) == 2
    assert [m.row_id for m in world.mutations] == ["etl-1", "etl-2", "etl-3"]


async def test_an_ambiguous_claim_on_a_non_idempotent_tool_dead_letters(tmp_path: Path) -> None:
    """A card that may already have been charged is a question for a human, not a guess."""
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    call = ToolCall("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    effect = await buffer.stage(branch, call, registry_of(dispatcher, "charge_card"))
    offset = await confirm(journal, branch)
    # A previous process claimed this key and never settled it.
    await journal.claim_dispatch(
        PendingClaim(
            run_id=RUN,
            nkey=effect.nkey,
            idem_key=effect.key,
            effect_id="ef-from-dead-process",
            branch_id="br-dead",
            tool="charge_card",
        )
    )
    report = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    assert not report.ok and report.count(EffectOutcome.DEAD_LETTER) == 1
    assert world.mutations_by("charge_card") == []


async def test_a_partition_reports_that_nothing_left_the_process() -> None:
    """Unreachable and timed-out are different facts, and only one is safe to retry."""
    assert Partitioned("x").sent == "no"
    assert ToolDispatchError("x").sent == "maybe"
