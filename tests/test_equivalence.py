"""THE EQUIVALENCE TEST (spec task 5.1, Hard Rule 9). Mandatory. Never skipped.

For any journaled run, the effect ledger with speculation on equals the ledger with speculation
off. That is the claim that makes the rest of the project usable: if speculating changed what
reached the world, every latency number would be bought with a behaviour change nobody asked
for.

Two things stop this from being a test that passes because it checks nothing.

``expect_rows`` -- the workload declares how many effects it produces. A runtime that
dispatched nothing would otherwise compare two empty ledgers and report success for a run that
did not happen.

``ledger_matches_world`` -- each arm's ledger is joined against what its world actually
received, in order and by idempotency key. Two ledgers agreeing with each other while both
disagree with the world is a failure the relation alone cannot see.
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
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world
from specunode.verify.equivalence import (
    EquivalenceError,
    assert_equivalent,
    ledger_matches_world,
    normalise_for_equivalence,
)

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})

#: Each workload declares what it produces, so an empty comparison cannot pass.
WORKLOADS = [
    pytest.param("support_agent", 2, id="support_agent"),
]


async def run_arm(tmp_path: Path, *, speculation: bool, db: str) -> tuple[RunResult, World]:
    world = standard_world()
    adapter, registry = build(world)
    assert isinstance(registry, ToolRegistry)
    journal = Journal(tmp_path / db)
    model = ScriptedModel(turns=[tool_turn(CHARGE, turn=0)])
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=speculation),
    )
    return await scheduler.run(new_ulid(), {"customer_id": "cus-1"}), world


@pytest.mark.parametrize(("workload", "expect_rows"), WORKLOADS)
async def test_the_two_arms_reach_the_same_world(
    tmp_path: Path, workload: str, expect_rows: int
) -> None:
    """Hard Rule 9, over a real run of a real workload rather than synthetic ledgers."""
    off, off_world = await run_arm(tmp_path, speculation=False, db=f"{workload}-off.db")
    on, on_world = await run_arm(tmp_path, speculation=True, db=f"{workload}-on.db")
    assert off.ok and on.ok

    assert_equivalent(
        off.ledger,
        on.ledger,
        off_world.mutations,
        on_world.mutations,
        expect_rows=expect_rows,
        # Zero, and stated rather than assumed: with tier 0 alone nothing mispredicts, because
        # it only ever proposes calls the target has already emitted. A workload with a real
        # predictor declares a non-zero minimum so the comparison cannot pass without the store
        # buffer ever having held anything back.
        min_squashed_with_staged=0,
    )


@pytest.mark.parametrize(("workload", "expect_rows"), WORKLOADS)
async def test_each_arms_ledger_matches_its_own_world(
    tmp_path: Path, workload: str, expect_rows: int
) -> None:
    """A ledger claiming an effect and a world receiving it are different facts."""
    for speculation in (False, True):
        result, world = await run_arm(
            tmp_path, speculation=speculation, db=f"{workload}-{speculation}.db"
        )
        assert len(result.ledger.rows) == expect_rows
        assert ledger_matches_world(result.ledger, world.mutations)


async def test_the_comparison_is_not_vacuous(tmp_path: Path) -> None:
    """A run that dispatched nothing must fail rather than compare two empty ledgers."""
    off, off_world = await run_arm(tmp_path, speculation=False, db="vac-off.db")
    on, on_world = await run_arm(tmp_path, speculation=True, db="vac-on.db")
    with pytest.raises(EquivalenceError, match="were not compared"):
        assert_equivalent(
            off.ledger, on.ledger, off_world.mutations, on_world.mutations, expect_rows=99
        )


async def test_the_relation_notices_an_extra_effect(tmp_path: Path) -> None:
    """The shape a leak has: the speculative arm did something the sequential one did not."""
    from dataclasses import replace

    off, _off_world = await run_arm(tmp_path, speculation=False, db="extra-off.db")
    on, _on_world = await run_arm(tmp_path, speculation=True, db="extra-on.db")

    leaked = replace(on.ledger.rows[0], dispatch_index=9, retire_seq=9)
    tampered = replace(on.ledger, rows=(*on.ledger.rows, leaked))
    assert normalise_for_equivalence(off.ledger) != normalise_for_equivalence(tampered)


async def test_both_arms_journal_chains_verify(tmp_path: Path) -> None:
    for speculation in (False, True):
        journal = Journal(tmp_path / f"chain-{speculation}.db")
        world = standard_world()
        adapter, registry = build(world)
        assert isinstance(registry, ToolRegistry)
        run_id = new_ulid()
        scheduler = Scheduler(
            graph=adapter,
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
            target=JournaledModel(
                ScriptedModel(turns=[tool_turn(CHARGE, turn=0)]), journal, provider="scripted"
            ),
            policy=Policy(speculation=speculation),
        )
        await scheduler.run(run_id, {"customer_id": "cus-1"})
        assert journal.verify_chain(run_id).ok
