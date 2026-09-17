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


# -- the reads E3 exists for --------------------------------------------------------------------


async def read_arm(
    tmp_path: Path, *, speculate: bool, db: str
) -> tuple[object, World, Journal, str]:
    """A three-call turn whose middle read is the one a speculation runs ahead."""
    from tests.integration.test_speculation import FixedDrafter, OneTurnGraph

    from specunode.core.decision import ToolCall

    world = standard_world()
    registry = registry_with_concurrent_writer(world, interfere=True)
    registry.register(
        ToolSpec(name="fetch_runbook", effect=EffectClass.READ, fn=world.fetch_runbook)
    )
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(
                turns=[
                    tool_turn(
                        ("fetch_runbook", {"section": "restart"}),
                        ("get_pipeline_status", {"pipeline_id": "etl-1"}),
                        ("restart_job", {"job_id": "etl-1"}),
                        turn=0,
                    )
                ],
                block_delay_ms=25.0,
            ),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True, on_stale_read="squash"),
        predictor=(
            FixedDrafter(ToolCall("get_pipeline_status", {"pipeline_id": "etl-1"}))
            if speculate
            else None
        ),
    )
    run_id = new_ulid()
    return await scheduler.run(run_id, {}), world, journal, run_id


@pytest.mark.timeout(60)
async def test_a_stale_read_made_by_a_confirmed_speculation_is_still_caught(
    tmp_path: Path,
) -> None:
    """Turning speculation ON used to disable the check that makes speculation safe.

    ``adopt`` moved a confirmed speculation's staged *effects* to the branch that retires, and
    left its ``read_set`` behind on the child. A confirmed speculation never retires, so
    ``validate_reads`` never ran over it -- and the reads it made are by definition the ones
    issued on a guess, which are the only reads lattice rule E3 exists to re-check. E3 was
    validating the canonical branch's own reads, which by the module's own doctrine need no
    validation, and skipping the genuine guesses.

    Measured before the fix on exactly this turn: with speculation off the run refused and
    nothing reached the world; with it on the run returned ok=True, ``restart_job`` reached the
    world, and the tally showed one read validated instead of two.
    """
    sequential, seq_world, _, _ = await read_arm(tmp_path, speculate=False, db="e3-seq.db")
    speculative, spec_world, journal, run_id = await read_arm(
        tmp_path, speculate=True, db="e3-spec.db"
    )

    assert not sequential.ok  # type: ignore[attr-defined]
    assert [m.tool for m in seq_world.mutations] == []

    assert not speculative.ok, (  # type: ignore[attr-defined]
        "the speculative arm retired despite a stale read -- E3 skipped the read the "
        "speculation itself made"
    )
    assert [m.tool for m in spec_world.mutations] == [], (
        "a write computed from a stale speculative read reached the world"
    )

    validated = [entry.payload for entry in journal.read(run_id, kinds=["read_validated"])]
    assert any(int(p["stale"]) > 0 for p in validated), "the speculation's stale read was missed"
    assert sum(int(p["total"]) for p in validated) >= 2, (
        "the speculation's read was never counted, so it was never re-checked"
    )


@pytest.mark.timeout(60)
async def test_a_reprobe_that_raises_does_not_count_as_fresh(tmp_path: Path) -> None:
    """An unchecked read is not a checked one -- the docstring said so and the gate did not.

    ``validate_reads`` recorded ``unreadable`` for a probe that raised, and nothing consulted
    it: ``_retire`` gated on ``stale`` alone, so the write dispatched. ``unreadable`` also had
    no counter, was absent from the journal payload, and was excluded from ``witnessed`` -- so
    the read vanished from both the numerator and the denominator, and a run where every probe
    errored rendered as ``0/0 fresh``. Any flaky or hostile upstream turned E3 off silently.
    """
    world = standard_world()
    registry = ToolRegistry()
    calls = {"n": 0}

    async def flaky_status(pipeline_id: str) -> JsonValue:
        calls["n"] += 1
        if calls["n"] > 1:  # the retirement-time re-probe
            raise RuntimeError("upstream refused the re-check")
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=flaky_status,
            witness=True,
        )
    )
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    journal = Journal(tmp_path / "flaky.db")
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
        policy=Policy(speculation=True, on_stale_read="squash", on_unverifiable_read="squash"),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})

    assert not result.ok, "a read whose re-check raised was treated as freshly validated"
    assert [m.tool for m in world.mutations] == [], (
        "a write backed by a read nobody could re-check reached the world"
    )
    validated = [entry.payload for entry in journal.read(run_id, kinds=["read_validated"])]
    assert any(int(p.get("unreadable", 0)) > 0 for p in validated), (
        "the failed re-probe was not recorded as unreadable"
    )
    # And it is never folded into the fresh count, under either policy.
    assert all(int(p["fresh"]) == 0 for p in validated), (
        "an unchecked read was counted as a checked one"
    )


@pytest.mark.timeout(60)
async def test_an_unreadable_probe_is_visible_even_when_the_policy_proceeds(
    tmp_path: Path,
) -> None:
    """The default lets the drain decide, and still refuses to call the read fresh.

    ``on_unverifiable_read`` defaults to ``proceed`` because an upstream that cannot answer a
    re-probe is usually one that is about to fail the dispatch too, and the drain's handling is
    strictly more informative there: it dead-letters the effect by name. Refusing first replaces
    a specific dead letter with "unverifiable" and pre-empts the project's whole answer to the
    two-generals problem -- the chaos matrix and the partition test both stopped exercising it.

    What must hold under either policy is that the read is never reported as checked.
    """
    world = standard_world()
    registry = ToolRegistry()
    calls = {"n": 0}

    async def flaky_status(pipeline_id: str) -> JsonValue:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("upstream refused the re-check")
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=flaky_status,
            witness=True,
        )
    )
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    journal = Journal(tmp_path / "flaky-proceed.db")
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
        policy=Policy(speculation=True),  # on_unverifiable_read defaults to "proceed"
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    validated = [entry.payload for entry in journal.read(run_id, kinds=["read_validated"])]
    assert any(int(p.get("unreadable", 0)) > 0 for p in validated), (
        "the failed re-probe left no trace at all, which is the silent-disable this guards"
    )
    assert all(int(p["fresh"]) == 0 for p in validated), (
        "an unchecked read was counted as a checked one"
    )
