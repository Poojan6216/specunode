"""A whole agent, end to end, through the real runtime (spec task 2.2).

Task 2.2's Verify: the support example runs to completion against the fake world with a
scripted model; the ledger has one row per write; and ``world.mutations`` matches the ledger
rows exactly. That last clause is the one that matters -- the ledger claiming an effect and the
world having received it are different facts, and a runtime that got them out of step would
still produce a plausible-looking ledger.

The example is deliberately the uncomfortable shape: the node charges a card and then reads the
charge id back out of the result to send a receipt. The node cannot be handed a real value
before the branch retires (nothing may reach the world from an unretired branch), and it cannot
be left waiting for one either, so this is the case that fails if staging returns a value, if
the drain is inlined into the tool call, or if the scheduler simply awaits the node task.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from examples.support_agent.agent import build

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.effects import ToolRegistry
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})


def make_run(tmp_path: Path, world: World | None = None) -> tuple[Scheduler, World, Journal, str]:
    world = world or standard_world()
    adapter, registry = build(world)
    assert isinstance(registry, ToolRegistry)
    journal = Journal(tmp_path / "journal.db")
    run_id = new_ulid()
    model = ScriptedModel(turns=[tool_turn(CHARGE, turn=0)])
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=run_id),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=False),
    )
    return scheduler, world, journal, run_id


async def test_the_support_example_runs_to_completion(tmp_path: Path) -> None:
    scheduler, _world, _journal, run_id = make_run(tmp_path)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert result.ok, result.error
    assert result.state["receipted"] is True
    assert result.steps == 3


async def test_the_ledger_has_one_row_per_write(tmp_path: Path) -> None:
    scheduler, _world, _journal, run_id = make_run(tmp_path)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert [row.call.name for row in result.ledger.rows] == ["charge_card", "send_receipt"]
    assert all(row.status == "DISPATCHED" for row in result.ledger.rows)


async def test_the_ledger_rows_match_what_the_world_received(tmp_path: Path) -> None:
    """The ledger claiming an effect and the world receiving it are different facts."""
    scheduler, world, _journal, run_id = make_run(tmp_path)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert [m.tool for m in world.mutations] == [row.call.name for row in result.ledger.rows]
    assert [m.effect_key for m in world.mutations] == [row.nkey for row in result.ledger.rows]


async def test_the_read_ran_but_never_mutated_anything(tmp_path: Path) -> None:
    scheduler, world, _journal, run_id = make_run(tmp_path)
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    # The second is lattice rule E3's witness re-check at retirement, which is an upstream
    # call the design counts rather than hides.
    assert [r.tool for r in world.reads] == ["lookup_customer", "lookup_customer"]
    assert "lookup_customer" not in {m.tool for m in world.mutations}


async def test_the_receipt_carries_the_real_charge_id(tmp_path: Path) -> None:
    """The node read the first write's result, so a placeholder here would be visible."""
    scheduler, world, _journal, run_id = make_run(tmp_path)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    receipt = next(row for row in result.ledger.rows if row.call.name == "send_receipt")
    assert receipt.call.args["charge_id"] == "chg-1"
    assert "$specunode.handle:" not in str(receipt.call.args)
    assert world.tables["messages"]


async def test_nothing_reached_the_world_before_its_branch_was_confirmed(tmp_path: Path) -> None:
    """Hard Rule 3, over a real run rather than a generated tree."""
    scheduler, world, journal, run_id = make_run(tmp_path)
    await scheduler.run(run_id, {"customer_id": "cus-1"})

    confirmed = {
        entry.payload["branch_id"]
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    assert {m.branch_id for m in world.mutations} - {"<external>"} <= confirmed


async def test_every_write_was_staged_before_it_was_dispatched(tmp_path: Path) -> None:
    scheduler, _world, journal, run_id = make_run(tmp_path)
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    kinds = [e.kind for e in journal.read(run_id)]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    for kind in ("effect_staged", "effect_dispatched", "model_request", "model_response"):
        assert kind in kinds, f"{kind} was never journaled"
    assert kinds.index("effect_staged") < kinds.index("effect_dispatched")


async def test_the_journal_chain_verifies_after_a_real_run(tmp_path: Path) -> None:
    scheduler, _world, journal, run_id = make_run(tmp_path)
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert journal.verify_chain(run_id).ok


async def test_the_run_is_stamped_unchecked_rather_than_claiming_a_check(tmp_path: Path) -> None:
    """Correction C2: a sequential run compares no rebuilt prompts, and must not say it did."""
    scheduler, _world, _journal, run_id = make_run(tmp_path)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert result.ledger.context_identity == "unchecked"


async def test_a_partition_dead_letters_rather_than_half_charging(tmp_path: Path) -> None:
    world = standard_world()
    world.partition(at=2)
    scheduler, _world, _journal, run_id = make_run(tmp_path, world)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert not result.ok
    statuses = [row.status for row in result.ledger.rows]
    assert "DEAD_LETTER" in statuses
    assert len(world.mutations_by("send_receipt")) == 0


async def test_two_runs_of_the_same_script_produce_the_same_effects(tmp_path: Path) -> None:
    """Determinism at the level a user cares about: same script, same world outcome."""
    first_scheduler, first_world, _j1, run1 = make_run(tmp_path)
    await first_scheduler.run(run1, {"customer_id": "cus-1"})
    second_scheduler, second_world, _j2, run2 = make_run(tmp_path / "second")
    await second_scheduler.run(run2, {"customer_id": "cus-1"})

    assert [(m.tool, m.args_hash) for m in first_world.mutations] == [
        (m.tool, m.args_hash) for m in second_world.mutations
    ]


@pytest.mark.timeout(20)
async def test_the_run_does_not_deadlock_on_a_node_that_reads_its_own_write(
    tmp_path: Path,
) -> None:
    """The failure this whole arrangement exists to avoid, asserted as a timeout."""
    scheduler, _world, _journal, run_id = make_run(tmp_path)
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert result.ok
