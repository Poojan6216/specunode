"""Subprocess helper: run the support agent, optionally dying at an exact point in the run.

``python _kill_agent.py <dir> <run_id> <kill> [resume] [changed-mind]``, where ``kill`` is one of

* ``-1`` -- run to the end, and print how many kill points the run has
* ``op:N`` -- die just before the run's N-th durable journal write: an entry appended, a
  dispatch claimed, a claim marked unsent, or an ack settled
* ``send:N`` -- die just before the world applies its N-th mutation: the claim is on disk and
  the request never arrived -- the lost request
* ``mutation:N`` -- die just after the world has durably applied its N-th mutation, before the
  runtime has recorded that it did -- the request went out and the reply was lost

``changed-mind`` scripts a model that decides differently -- charges 30 rather than 25 -- which
is what a real model asked the same question twice may do. A resumed process is given it to
show that a decision already in the journal is not asked for again.

The process dies by ``os._exit``: no cleanup, no flush, no finally blocks, as if the machine
lost power, except that what the OS already holds survives. The world writes to a durable log,
so what the dead process sent is still visible to the process that resumes -- which is the whole
point: a resumed run that could not see the dead one's effects could not avoid re-sending them.

Kill points used to be delays in milliseconds, calibrated against a timed run. A delay is not a
point in the run: on a CI runner the same delay landed before the run had journaled anything on
one attempt and after it had finished on another, so the test failed on either side of a
window it could not see. Counting names each point at which the journal, or the effects in
the world, change on disk -- on every machine, every time.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.support_agent.agent import build

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.journal import journal as journal_module
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})
CHANGED_MIND = ("charge_card", {"customer_id": "cus-1", "amount": 30.0})

#: Every method that changes the journal database, all run on its writer thread.
DURABLE_WRITES = ("_append", "_claim", "_mark_not_sent", "_settle")


def seeded_world(directory: Path) -> World:
    world = World(log_path=directory / "world.jsonl")
    if not world.tables["customers"]:
        for index in range(1, 4):
            world.seed("customers", f"cus-{index}", name=f"Customer {index}", balance=100.0)
    return world


class Counter:
    """Counts one kind of event and dies at the chosen one."""

    def __init__(self, die_at: int | None) -> None:
        self.count = 0
        self.die_at = die_at
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self.count += 1
            if self.count == self.die_at:
                os._exit(9)


def count_journal_writes(counter: Counter) -> None:
    """Tick before each durable write, so ``op:N`` dies with writes 1..N-1 on disk."""
    writer = journal_module._JournalWriter
    for name in DURABLE_WRITES:
        original: Callable[..., Any] = getattr(writer, name)

        def before(self: object, *args: Any, _original: Callable[..., Any] = original) -> Any:
            counter.tick()
            return _original(self, *args)

        setattr(writer, name, before)


def count_world_mutations(world: World, sends: Counter, mutations: Counter) -> None:
    """Tick either side of the fsync to the world log: ``send:N`` before, ``mutation:N`` after.

    Dying before the log write loses the request -- the world's in-memory change dies with the
    process, and the world a resumed process opens never had it. Dying after loses the reply.
    """
    original = world._persist_mutation

    def around(*args: Any) -> None:
        sends.tick()
        original(*args)
        mutations.tick()

    world._persist_mutation = around  # type: ignore[method-assign]


async def main() -> None:
    directory = Path(sys.argv[1])
    run_id = sys.argv[2]
    kill = sys.argv[3]
    resuming = "resume" in sys.argv[4:]
    charge = CHANGED_MIND if "changed-mind" in sys.argv[4:] else CHARGE

    kind, _, at = kill.partition(":")
    writes = Counter(int(at) if kind == "op" else None)
    sends = Counter(int(at) if kind == "send" else None)
    mutations = Counter(int(at) if kind == "mutation" else None)
    count_journal_writes(writes)

    world = seeded_world(directory)
    count_world_mutations(world, sends, mutations)
    adapter, registry = build(world)
    journal = Journal(directory / "journal.db")
    model = ScriptedModel(turns=[tool_turn(charge, turn=0), tool_turn(charge, turn=1)])
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=1.0),  # type: ignore[arg-type]
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=False),
    )
    result = (
        await scheduler.resume(run_id)
        if resuming
        else await scheduler.run(run_id, {"customer_id": "cus-1"})
    )
    world.close()
    print(
        f"done ok={result.ok} rows={len(result.ledger.rows)} "
        f"ops={writes.count} mutations={mutations.count}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
