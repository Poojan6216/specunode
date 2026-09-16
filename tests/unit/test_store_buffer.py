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


async def test_even_the_canonical_path_gets_a_handle_and_a_future(tmp_path: Path) -> None:
    """A staged effect has no value to give, on any branch.

    Returning a real ack here would mean dispatching inside the call -- task 1.6's planted bug --
    and making the caller await the drain deadlocks the first write of the first sequential run,
    before any speculation exists. So the caller gets a future only the drain can complete.
    """
    buffer, _, _, dispatcher = build(tmp_path)
    branch = Branch(id="br-canon", status=BranchStatus.CONFIRMED)
    effect = await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    assert effect.placeholder == handle_for(effect.id)
    assert not buffer.ack_for(effect.id).done()


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
    first = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    second = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    assert first.count(EffectOutcome.DISPATCHED) == 1
    assert second.outcomes == (), "a settled effect is not re-walked on a later pass"
    assert second.ok and second.undrained == ()
    assert len(world.mutations) == 1, "the world must not receive the effect twice"


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


# -- the staged-ack future (correction C4) ------------------------------------------------------


async def test_the_drain_completes_the_ack_a_node_body_is_waiting_on(tmp_path: Path) -> None:
    """The shape a real node has: ack = await charge_card(...); then use ack["charge_id"]."""
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    effect = await buffer.stage(
        branch,
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 10.0}),
        registry_of(dispatcher, "charge_card"),
    )
    ack = buffer.ack_for(effect.id)
    assert not ack.done(), "the value must not exist before the branch retires"

    offset = await confirm(journal, branch)
    await buffer.drain(branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset)

    assert ack.done()
    value = await ack
    assert isinstance(value, dict) and value["charge_id"] == "chg-1"
    assert len(world.mutations) == 1


async def test_a_discarded_branch_cancels_the_acks_it_was_waiting_on(tmp_path: Path) -> None:
    """Otherwise a squashed branch's node body waits forever for a value never coming."""
    buffer, _journal, _world, dispatcher = build(tmp_path)
    branch = speculative()
    effect = await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    ack = buffer.ack_for(effect.id)
    branch.squash("mismatch")
    buffer.discard(branch)
    assert ack.cancelled()


async def test_a_dead_lettered_effect_fails_its_ack_rather_than_hanging(tmp_path: Path) -> None:
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    effect = await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    ack = buffer.ack_for(effect.id)
    offset = await confirm(journal, branch)
    world.partition()
    await buffer.drain(branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset)
    assert ack.done()
    with pytest.raises(ToolDispatchError):
        await ack


async def test_drain_refuses_a_caller_that_is_not_the_scheduler_task(tmp_path: Path) -> None:
    """The repair an implementer reaches for -- drain inline so my node gets its ack -- dispatches
    before the branch is confirmed. The guard makes that unreachable rather than discouraged."""
    import asyncio

    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    offset = await confirm(journal, branch)
    current = asyncio.current_task()
    assert current is not None
    buffer.scheduler_task = current

    async def from_a_branch_task() -> None:
        with pytest.raises(BranchClosed, match="branch task"):
            await buffer.drain(
                branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
            )

    await asyncio.create_task(from_a_branch_task())
    assert world.mutations == []


def test_forward_never_returns_a_hit() -> None:
    """Correction C12: there is no ForwardHit in this version and no switch that makes one."""
    from specunode.buffer import store_buffer

    assert not hasattr(store_buffer, "ForwardHit")
    assert store_buffer.ForwardResult.__args__ == (  # type: ignore[attr-defined]
        store_buffer.ForwardMiss,
        store_buffer.ForwardHazard,
    )


def test_no_projection_switch_ships() -> None:
    """A projection retires unverified and can void Hard Rule 9 in a supported configuration."""
    root = Path(__file__).resolve().parents[2] / "src"
    offenders = [
        path.name
        for path in root.rglob("*.py")
        if "allow_projection" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"a projection switch reappeared in {offenders}"


async def test_a_write_staged_from_a_resumed_node_body_still_reaches_the_world(
    tmp_path: Path,
) -> None:
    """The ordinary shape: ack = await charge_card(...); then send_receipt(ack["charge_id"]).

    The second write is staged only after the first one's ack arrives, which happens *during*
    the drain. This is the buffer's half of that: a later drain picks up what was staged since
    the earlier one, and reports nothing left undrained. The scheduler's half -- knowing when
    to drain again, because the node needs several event-loop turns to resume and stage -- is
    covered end to end in tests/integration/test_sequential_run.py.

    An effect staged and never dispatched is an authorised write dropped silently, identically
    in both arms, so the equivalence test would stay green while the world never received it.
    Nothing else in the suite can see that, which is why undrained is asserted here.
    """
    import asyncio

    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    first = await buffer.stage(
        branch,
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 10.0}),
        registry_of(dispatcher, "charge_card"),
    )

    async def node_body() -> None:
        ack = await buffer.ack_for(first.id)
        assert isinstance(ack, dict)
        await buffer.stage(
            branch,
            ToolCall("post_summary", {"channel": "#ops", "text": str(ack["charge_id"])}),
            registry_of(dispatcher, "post_summary"),
        )

    parked = asyncio.create_task(node_body())
    await asyncio.sleep(0)

    offset = await confirm(journal, branch)
    first = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )
    await parked  # the node resumes on the ack and stages its second write
    second = await buffer.drain(
        branch, dispatcher, confirmed_offset=offset, authorised_by_offset=offset
    )

    assert first.count(EffectOutcome.DISPATCHED) == 1
    assert second.count(EffectOutcome.DISPATCHED) == 1
    assert second.undrained == (), f"an authorised write was never dispatched: {second.undrained}"
    assert [m.tool for m in world.mutations] == ["charge_card", "post_summary"]
    summary = world.mutations[1]
    assert summary.args_hash, "the second write carried the first one's real result"


async def test_two_identical_calls_in_one_node_need_two_step_indices(tmp_path: Path) -> None:
    """A node body that issues the same call twice must produce two effects, not one.

    Sharing a step index makes both derive the same idempotency key, so dedupe suppresses the
    second at dispatch and an effect the sequential run performed never reaches the world.
    Nothing else in the suite can see that: the leak invariant is a subset over branch ids and
    cannot see a *missing* effect, and both equivalence arms collide identically. So staging a
    duplicate key is refused outright rather than quietly accepted.
    """
    buffer, _journal, _world, dispatcher = build(tmp_path)
    branch = speculative()
    spec = registry_of(dispatcher, "post_summary")
    call = ToolCall("post_summary", {"channel": "#ops", "text": "retry"})

    branch.advance_step()
    first = await buffer.stage(branch, call, spec)

    with pytest.raises(HazardViolation, match="fresh step index"):
        await buffer.stage(branch, call, spec)

    branch.advance_step()
    second = await buffer.stage(branch, call, spec)
    assert first.nkey != second.nkey
    assert first.key != second.key


async def test_advancing_the_step_is_what_separates_them(tmp_path: Path) -> None:
    buffer, _journal, _world, _dispatcher = build(tmp_path)
    branch = speculative()
    assert branch.advance_step() == 1
    assert branch.advance_step() == 2
    assert branch.cursor.step_index == 2
    assert buffer.pending(branch.id) == ()


async def test_drain_refuses_an_offset_that_is_not_this_branchs_confirmation(
    tmp_path: Path,
) -> None:
    """Comparing the offset to the journal head is satisfied by any later entry at all."""
    buffer, journal, world, dispatcher = build(tmp_path)
    branch = speculative()
    branch.advance_step()
    await buffer.stage(
        branch, ToolCall("restart_job", {"job_id": "etl-1"}), registry_of(dispatcher, "restart_job")
    )
    branch.confirm()
    branch.context_verified = True
    # Some other entry exists at this offset -- under an offset-vs-head check this passes.
    other = await journal.append_async(
        RUN, "policy_event", {"v": 1, "event": "alpha_update", "reason": "x"}
    )
    with pytest.raises(BranchClosed, match="branch_resolved"):
        await buffer.drain(branch, dispatcher, confirmed_offset=other, authorised_by_offset=other)
    assert world.mutations == []


async def test_a_branch_with_a_staged_slot_may_not_call_the_model() -> None:
    """Rule 13's structural half: a staged slot never fills, so no honest turn includes it."""
    from specunode.core.branch import ResultSlot, SlotStatus, TurnFrame
    from specunode.core.hazards import Hazard, analyse_model_request
    from specunode.core.policy import Policy

    branch = Branch(id="br-slot", status=BranchStatus.SPECULATIVE)
    assert analyse_model_request(branch, b'{"messages":[]}', Policy()) is None

    frame = TurnFrame(step=1, slots=[ResultSlot(ordinal=0, tool_use_id="u1")])
    frame.slots[0].status = SlotStatus.STAGED
    branch.frames.append(frame)
    assert (
        analyse_model_request(branch, b'{"messages":[]}', Policy())
        is Hazard.MODEL_TURN_AFTER_STAGED_WRITE
    )
