"""Subprocess helper: run the support agent, optionally dying part-way through.

``python _kill_agent.py <dir> <run_id> <delay_ms|-1> [resume]``

The world writes to a durable log, so what the killed process sent is still visible to the
process that resumes -- which is the whole point: a resumed run that could not see the dead
one's effects could not avoid re-sending them.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.support_agent.agent import build

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})


def seeded_world(directory: Path) -> World:
    world = World(log_path=directory / "world.jsonl")
    if not world.tables["customers"]:
        for index in range(1, 4):
            world.seed("customers", f"cus-{index}", name=f"Customer {index}", balance=100.0)
    return world


async def main() -> None:
    directory = Path(sys.argv[1])
    run_id = sys.argv[2]
    delay_ms = float(sys.argv[3])
    resuming = len(sys.argv) > 4 and sys.argv[4] == "resume"

    world = seeded_world(directory)
    adapter, registry = build(world)
    journal = Journal(directory / "journal.db")
    model = ScriptedModel(turns=[tool_turn(CHARGE, turn=0), tool_turn(CHARGE, turn=1)])
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=1.0),  # type: ignore[arg-type]
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=False),
    )
    # The killer starts here, not at process start: interpreter startup and imports dominate
    # the process lifetime, so a delay measured from launch almost never lands inside the run
    # -- and a kill test where nothing is killed is green and worthless.
    if delay_ms >= 0:

        def killer() -> None:
            time.sleep(delay_ms / 1000.0)
            os._exit(9)

        threading.Thread(target=killer, daemon=True).start()

    started = time.monotonic()
    result = (
        await scheduler.resume(run_id)
        if resuming
        else await scheduler.run(run_id, {"customer_id": "cus-1"})
    )
    work_ms = (time.monotonic() - started) * 1000.0
    world.close()
    print(f"done ok={result.ok} rows={len(result.ledger.rows)} work_ms={work_ms:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
