"""Adoption of a confirmed speculation (Hard Rules 3, 8 and 9).

When the model emits exactly the call a speculation predicted, that speculation stops being a
guess: its work is the work the run was always going to do. Only the canonical branch retires,
though, and the drain dispatches by branch id -- so the confirmed child's buffer has to move to
the branch that will retire it, and so does anything the child stages *afterwards*, because its
task keeps running until it gets the ack it is waiting for.

Both halves of that sentence were wrong at different times, and both failures were silent:

* The move alone was not enough. ``adopt`` is a point-in-time transfer, and the model can emit
  the confirming block while the speculation is still awaiting its own journal append. The late
  effect then landed in a list no drain visits, and the run hung with no error and no
  ``run_finished`` entry -- on the product's headline success path, a correctly predicted write.
* Placing it correctly was not enough either. Park events are keyed by branch id, so the late
  stage signalled the child's key, which nothing waits on. The effect was in the right list and
  still never left.

Every test here carries a timeout. The failure mode is a hang, and a hanging test is worse than
a failing one: it looks like slowness rather than like a defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.integration.test_speculation import FixedDrafter, OneTurnGraph, registry_for

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import ToolCall
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world
from specunode.verify.equivalence import assert_equivalent

#: A read, then the write the drafter predicts. Two blocks and not three, so the confirming
#: block arrives immediately after the prediction is made -- which is the window that was open.
CONFIRMED_WRITE_TURN = (
    ("fetch_runbook", {"section": "restart"}),
    ("restart_job", {"job_id": "etl-1"}),
)

#: Sub-millisecond gaps. The real race is the child's journal append (one fsync, ``synchronous
#: = FULL``) against the inter-block gap, so a slower journal widens this window considerably --
#: the Postgres backend or a slow disk reaches it at the tens of milliseconds the shipped bench
#: actually uses. Testing at zero is testing the same bug where it is cheap to reproduce.
TIGHT_GAPS_MS = (0.0, 0.1, 0.5)


async def run_turn(
    tmp_path: Path,
    turn: tuple[tuple[str, dict[str, object]], ...],
    *,
    predicted: ToolCall | None,
    gap_ms: float,
    db: str,
    speculation: bool = True,
) -> tuple[RunResult, World]:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(*turn, turn=0)], block_delay_ms=gap_ms),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=speculation),
        predictor=FixedDrafter(predicted) if (speculation and predicted) else None,
    )
    return await scheduler.run(new_ulid(), {}), world


@pytest.mark.timeout(60)
@pytest.mark.parametrize("gap_ms", TIGHT_GAPS_MS)
async def test_a_confirmed_write_prediction_completes_however_tight_the_stream(
    tmp_path: Path, gap_ms: float
) -> None:
    """The headline success path: speculate past a write, be right, and finish.

    This hung indefinitely. Not failed -- hung, with no error, no timeout and no journal entry
    saying the run had stopped.
    """
    result, world = await run_turn(
        tmp_path,
        CONFIRMED_WRITE_TURN,
        predicted=ToolCall("restart_job", {"job_id": "etl-1"}),
        gap_ms=gap_ms,
        db=f"tight-{gap_ms}.db",
    )

    assert result.ok, result.error
    assert [m.tool for m in world.mutations] == ["restart_job"], (
        "the authorised write never reached the world"
    )
    assert len(result.ledger.rows) == 1


@pytest.mark.timeout(60)
async def test_an_effect_staged_after_adoption_belongs_to_the_branch_that_retires(
    tmp_path: Path,
) -> None:
    """Asserted at the buffer, so the reason is visible rather than inferred from a hang.

    ``_adopted_into`` was recorded, exposed by a public accessor, described in a comment as the
    guard against exactly this -- and read by nothing in the repository.
    """
    journal = Journal(tmp_path / "late-stage.db")
    buffer = StoreBuffer(journal=journal, run_id=new_ulid())
    world = standard_world()
    registry = registry_for(world)

    from specunode.core.branch import Branch, BranchStatus

    parent = Branch(id=new_ulid())
    child = parent.fork(new_ulid(), predicted=ToolCall("restart_job", {"job_id": "etl-1"}), step=0)
    parent.status = BranchStatus.CONFIRMED
    child.confirm()

    # Adopt before the child has staged anything at all -- the window that was open.
    assert await buffer.adopt(child, parent) == 0
    assert buffer.adopted_into(child.id) == parent.id
    assert buffer.drain_owner_id(child) == parent.id

    effect = await buffer.stage(
        child, ToolCall("restart_job", {"job_id": "etl-1"}), registry.get("restart_job")
    )

    assert effect.branch_id == parent.id, "a late stage landed on the branch that will not retire"
    assert [e.id for e in buffer.pending(parent.id)] == [effect.id]
    assert buffer.pending(child.id) == (), "the child still holds an effect no drain will visit"


@pytest.mark.timeout(60)
async def test_a_speculation_takes_one_program_position_per_call(tmp_path: Path) -> None:
    """Hard Rule 8: the same logical effect gets the same key whether or not it was predicted.

    The child used to advance its cursor once itself and once more inside ``BranchTools.call``,
    burning two positions for one call. The parent's single advance at confirm cancelled that
    only by coincidence, and when the coincidence did not hold the adopted effect's key was
    derived one step off -- so a resume with speculation off would not dedupe against a crashed
    run that had it on, and the effect would go out twice.
    """
    # Both tools are registered WRITEs in this fixture. Two writes in one turn is the shape
    # that exposed the drift: the predicted second write is adopted, and the first write's
    # position is what the second's key is derived relative to.
    turn = (
        ("restart_job", {"job_id": "etl-1"}),
        ("charge_card", {"customer_id": "cus-1", "amount": 25.0}),
    )
    off, off_world = await run_turn(
        tmp_path, turn, predicted=None, gap_ms=5.0, db="steps-off.db", speculation=False
    )
    on, on_world = await run_turn(
        tmp_path,
        turn,
        predicted=ToolCall("charge_card", {"customer_id": "cus-1", "amount": 25.0}),
        gap_ms=5.0,
        db="steps-on.db",
    )
    assert off.ok and on.ok

    assert [(r.call.name, r.step_index) for r in off.ledger.rows] == [
        (r.call.name, r.step_index) for r in on.ledger.rows
    ], "speculating moved an effect to a different program position"

    # And the relation agrees, which is the form Hard Rule 9 is actually stated in.
    assert_equivalent(off.ledger, on.ledger, off_world.mutations, on_world.mutations, expect_rows=2)


@pytest.mark.timeout(60)
async def test_the_adoption_is_recorded_in_the_journal(tmp_path: Path) -> None:
    """A confirmed speculation's lifecycle must close in the durable record, not just in RAM."""
    result, _world = await run_turn(
        tmp_path,
        CONFIRMED_WRITE_TURN,
        predicted=ToolCall("restart_job", {"job_id": "etl-1"}),
        gap_ms=5.0,
        db="journalled.db",
    )
    assert result.ok

    journal = Journal(tmp_path / "journalled.db")
    adoptions = list(journal.read(result.run_id, kinds=["effect_adopted"]))
    resolved = [
        entry.payload
        for entry in journal.read(result.run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "confirmed" and entry.payload.get("adopted_by")
    ]
    assert resolved, "no branch was journaled as confirmed-and-adopted"
    # Whether the effect moved with adopt() or was staged late and routed by it, exactly one
    # branch must end up owning it, and the journal must say which.
    assert len(adoptions) <= 1
