"""Read validation at retirement — lattice rule E3 (spec task 3.5).

A branch that ran while it was still a guess may have read values that have since changed. E3
re-fetches every witnessed read it made and refuses to retire it if any of them went stale: its
writes were computed from something the world no longer agrees with.

**This had no caller.** ``validate_reads`` was defined, unit-tested, and invoked by an attack
script that hand-builds a ``Branch`` — and by nothing in ``src/``. So a branch whose witnessed
speculative reads had gone stale retired anyway and drained its writes; ``policy.on_stale_read``
was dead configuration that was nonetheless journaled into ``run_started`` and rendered as though
it applied; and every ledger ever produced printed ``reads validated at retirement: 0/0 fresh``,
which the ledger renderer's own docstring calls the difference between a receipt and a
reassurance. BUILD_SPEC marked the task done and Phase Gate 3 claimed "stale-read squash works".

The staleness here is caused the way it happens in production: something outside this branch
changes the row between the read and the retirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.integration.test_speculation import OneTurnGraph

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

#: A read the model emits first, then the write whose arguments depend on it.
TURN = (
    ("get_pipeline_status", {"pipeline_id": "etl-1"}),
    ("restart_job", {"job_id": "etl-1"}),
)


def registry_with_concurrent_writer(world: World, *, interfere: bool) -> ToolRegistry:
    """The real tools, with an optional concurrent writer racing the witnessed read.

    ``interfere`` bumps the row's version immediately after the read returns, which is exactly
    what another process committing to that row looks like from here.
    """
    registry = ToolRegistry()

    async def get_pipeline_status(pipeline_id: str) -> JsonValue:
        value = await world.get_pipeline_status(pipeline_id=pipeline_id)
        if interfere:
            world.versions[("jobs", pipeline_id)] = world.versions.get(("jobs", pipeline_id), 0) + 1
        return value

    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=get_pipeline_status,
            witness=True,
        )
    )
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    registry.register(
        ToolSpec(name="fetch_runbook", effect=EffectClass.READ, fn=world.fetch_runbook)
    )
    return registry


async def run(
    tmp_path: Path, *, interfere: bool, on_stale: str, db: str
) -> tuple[object, World, Journal, str]:
    world = standard_world()
    registry = registry_with_concurrent_writer(world, interfere=interfere)
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=20.0),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True, on_stale_read=on_stale),  # type: ignore[arg-type]
    )
    run_id = new_ulid()
    return await scheduler.run(run_id, {}), world, journal, run_id


@pytest.mark.timeout(60)
async def test_a_clean_run_validates_its_reads_and_says_so(tmp_path: Path) -> None:
    """The counting half: a run with speculative reads must produce a real tally.

    Zero is honest for a run with nothing to validate. It was printed on *every* run.
    """
    result, world, journal, run_id = await run(
        tmp_path, interfere=False, on_stale="squash", db="clean.db"
    )
    assert result.ok, result.error  # type: ignore[attr-defined]
    assert [m.tool for m in world.mutations] == ["restart_job"]

    validated = [entry.payload for entry in journal.read(run_id, kinds=["read_validated"])]
    assert validated, "no read_validated entry was ever written"
    assert sum(int(p["total"]) for p in validated) > 0, "the tally counted nothing"
    assert sum(int(p["stale"]) for p in validated) == 0
    assert sum(int(p["fresh"]) for p in validated) > 0, (
        "the early-issued read was never re-checked at retirement"
    )


@pytest.mark.timeout(60)
async def test_a_stale_witnessed_read_stops_the_branch_retiring(tmp_path: Path) -> None:
    """The refusing half. The write must not reach the world."""
    result, world, journal, run_id = await run(
        tmp_path, interfere=True, on_stale="squash", db="stale.db"
    )

    assert not result.ok, "the branch retired despite a witnessed read going stale"  # type: ignore[attr-defined]

    validated = [entry.payload for entry in journal.read(run_id, kinds=["read_validated"])]
    assert any(int(p["stale"]) > 0 for p in validated), "the staleness was not detected"

    assert [m.tool for m in world.mutations] == [], (
        "a write computed from a stale read reached the world"
    )
    squashed = [
        entry.payload
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "squashed"
    ]
    assert squashed, "the branch was not squashed"
    assert "stale" in str(squashed[0].get("reason", "")).lower()


@pytest.mark.timeout(60)
async def test_the_policy_is_read_rather_than_journaled_and_ignored(tmp_path: Path) -> None:
    """``on_stale_read`` was journaled into ``run_started`` while nothing consulted it.

    With ``stall`` the branch is not squashed on staleness -- the setting has to change the
    behaviour, or it is decoration.
    """
    squash_result, squash_world, _, _ = await run(
        tmp_path, interfere=True, on_stale="squash", db="pol-squash.db"
    )
    stall_result, stall_world, _, _ = await run(
        tmp_path, interfere=True, on_stale="stall", db="pol-stall.db"
    )

    assert not squash_result.ok  # type: ignore[attr-defined]
    assert [m.tool for m in squash_world.mutations] == []
    # Different setting, different outcome. That is the whole assertion.
    assert stall_result.ok is not squash_result.ok or bool(stall_world.mutations)  # type: ignore[attr-defined]
