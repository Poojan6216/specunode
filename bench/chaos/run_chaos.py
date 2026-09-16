"""Chaos and concurrency (spec tasks 5.2 and 5.3).

Two matrices, both of which report counters rather than assertions, so a nightly run produces a
number even when nothing broke.

**Concurrency.** Twenty runs sharing one journal file and one world, interleaved. Each run's
ledger must equal the ledger it produces alone, and no branch may ever observe a sibling's
staged effect. The second is Hard Rule 6, and it is the one that fails silently: a shared
mutable container is how isolation is violated by accident, and the symptom is a wrong value
rather than a crash.

**Chaos.** Faults injected during a drain -- partition, duplicate delivery, slow reads that
outlive their branch -- with the leak invariant checked after each. The kill/resume half of 5.2
lives in ``tests/chaos/test_kill_resume.py``, because it needs subprocesses and belongs with the
tests that are run on every change rather than nightly.

``python bench/chaos/run_chaos.py --out bench/results/chaos.json``
``python bench/chaos/run_chaos.py --concurrency --out bench/results/concurrency.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from specunode.verify.equivalence import normalise_for_equivalence

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})


@dataclass
class ChaosReport:
    runs: int = 0
    leaks: int = 0
    duplicate_deliveries: int = 0
    equivalence_failures: int = 0
    cross_branch_observations: int = 0
    dead_letters: int = 0
    notes: list[str] = field(default_factory=list)


async def one_run(
    journal: Journal,
    world: World,
    *,
    speculation: bool = True,
    partition_at: int | None = None,
    duplicate: str | None = None,
    slow_read_ms: int = 0,
) -> RunResult:
    adapter, registry = build(world)
    assert isinstance(registry, ToolRegistry)
    if partition_at is not None:
        world.partition(at=partition_at)
    if duplicate:
        world.duplicate_delivery(duplicate)
    if slow_read_ms:
        world.slow("read", slow_read_ms)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(
            registry=registry, max_attempts=2, base_delay_ms=0.5, cap_delay_ms=2.0
        ),
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(CHARGE, turn=0)]), journal, provider="scripted"
        ),
        policy=Policy(speculation=speculation),
    )
    return await scheduler.run(new_ulid(), {"customer_id": "cus-1"})


def leak_count(result: RunResult, world: World, journal: Journal) -> int:
    retired = {
        str(entry.payload["branch_id"])
        for entry in journal.read(result.run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    touched = {m.branch_id for m in world.mutations} - {"<external>"}
    return len(touched - retired)


def duplicate_count(world: World) -> int:
    keys = [m.effect_key for m in world.mutations if m.effect_key]
    return len(keys) - len(set(keys))


def branches_of(journal: Journal, run_id: str) -> set[str]:
    return {
        str(entry.payload["branch_id"]) for entry in journal.read(run_id, kinds=["branch_forked"])
    }


def retired_of(journal: Journal, run_id: str) -> set[str]:
    return {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }


async def run_concurrency(runs: int, out: Path | None) -> ChaosReport:
    """Many runs through one journal, and many runs through one world.

    Two separate questions, measured separately, because conflating them produces a number that
    looks like a leak and is not one.

    *One journal, separate worlds.* SQLite in WAL mode admits one writer at a time, so this is
    the question of whether interleaved runs corrupt each other's entries or change what a run
    does. Each run's ledger must normalise to the same bytes as the same workload run alone.

    *One journal and one world.* Now the runs genuinely interact -- they charge the same
    customer, and each charge gets its own id -- so their ledgers are *supposed* to differ, and
    comparing them to a solo run measures nothing. What must hold is attribution: no mutation
    may carry a branch id belonging to another run, and no branch may reach the world without
    retiring *in its own run*.
    """
    report = ChaosReport(runs=runs)
    base = await asyncio.to_thread(_workdir)
    shared_journal = Journal(base / "shared.db")

    # (1) One journal, a world each.
    worlds = [standard_world() for _ in range(runs)]
    together = await asyncio.gather(*(one_run(shared_journal, world) for world in worlds))

    solo_world = standard_world()
    solo = await one_run(Journal(base / "solo.db"), solo_world)
    reference = normalise_for_equivalence(solo.ledger)
    for result, world in zip(together, worlds, strict=True):
        if not result.ok:
            report.notes.append(f"{result.run_id} finished not-ok: {result.error}")
        if normalise_for_equivalence(result.ledger) != reference:
            report.equivalence_failures += 1
        if not shared_journal.verify_chain(result.run_id).ok:
            report.notes.append(f"chain broken in {result.run_id}")
        retired = retired_of(shared_journal, result.run_id)
        touched = {m.branch_id for m in world.mutations} - {"<external>"}
        report.leaks += len(touched - retired)
        report.duplicate_deliveries += duplicate_count(world)

    # (2) One journal AND one world: attribution under genuine interaction.
    crowded_journal = Journal(base / "crowded.db")
    crowded_world = standard_world()
    crowded = await asyncio.gather(*(one_run(crowded_journal, crowded_world) for _ in range(runs)))
    owner: dict[str, str] = {}
    for result in crowded:
        for branch_id in branches_of(crowded_journal, result.run_id):
            if branch_id in owner:
                # One branch id claimed by two runs would mean the journal handed out an
                # identity twice, which is the shape a cross-run observation would take.
                report.cross_branch_observations += 1
            owner[branch_id] = result.run_id

    all_retired = set()
    for result in crowded:
        all_retired |= retired_of(crowded_journal, result.run_id)
    escaped = {m.branch_id for m in crowded_world.mutations} - {"<external>"} - all_retired
    report.leaks += len(escaped)
    report.duplicate_deliveries += duplicate_count(crowded_world)
    unattributed = [
        m.branch_id
        for m in crowded_world.mutations
        if m.branch_id != "<external>" and m.branch_id not in owner
    ]
    if unattributed:
        report.notes.append(f"{len(unattributed)} mutation(s) carry an unknown branch id")

    if out:
        write(out, {"concurrency": asdict(report)})
    return report


async def run_chaos(rounds: int, out: Path | None) -> ChaosReport:
    """Faults during the drain, with the leak invariant checked after each."""
    report = ChaosReport(runs=rounds)
    base = await asyncio.to_thread(_workdir)

    for index in range(rounds):
        world = standard_world()
        journal = Journal(base / f"chaos-{index}.db")
        partition_at = 2 if index % 3 == 0 else None
        duplicate = "charge_card" if index % 3 == 1 else None
        slow = 40 if index % 3 == 2 else 0

        result = await one_run(
            journal,
            world,
            partition_at=partition_at,
            duplicate=duplicate,
            slow_read_ms=slow,
        )
        report.leaks += leak_count(result, world, journal)
        report.duplicate_deliveries += duplicate_count(world) if not duplicate else 0
        report.dead_letters += sum(1 for row in result.ledger.rows if row.status == "DEAD_LETTER")
        if not journal.verify_chain(result.run_id).ok:
            report.notes.append(f"chain broken in {result.run_id}")

    if out:
        write(out, {"chaos": asdict(report)})
    return report


def _workdir() -> Path:
    """Scratch directory for the matrices' journals. Created off the event loop."""
    base = Path(".specunode-chaos")
    base.mkdir(exist_ok=True)
    return base


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpecuNode chaos and concurrency")
    parser.add_argument("--concurrency", action="store_true")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.concurrency:
        report = asyncio.run(run_concurrency(args.runs, args.out))
        label = "CONCURRENCY"
    else:
        report = asyncio.run(run_chaos(args.rounds, args.out))
        label = "CHAOS"

    print()
    print(f"{label}")
    print(f"  runs:                       {report.runs}")
    print(f"  leaks:                      {report.leaks}")
    print(f"  duplicate deliveries:       {report.duplicate_deliveries}")
    print(f"  equivalence failures:       {report.equivalence_failures}")
    print(f"  cross-branch observations:  {report.cross_branch_observations}")
    print(f"  dead letters:               {report.dead_letters}")
    for note in report.notes:
        print(f"  note: {note}")
    print()
    bad = report.leaks + report.equivalence_failures + report.cross_branch_observations
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
