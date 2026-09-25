"""Subprocess helper for Demo 3: run the ops workload, optionally dying part-way through.

``python bench/_demo3_agent.py <dir> <run_id> <delay_ms|op:N|send:N|mutation:N|-1> [resume]``

A delay is what the demo itself uses: a SIGKILL-like death at a moment measured to land inside
the run's work. A named point (``bench._kill_points``) is what the test sweep uses, because a
delay can miss the run on a slow machine and a named point cannot.

A separate process because the demo's claim is about a process that *died*, and a process
cannot SIGKILL itself and then go on to prove anything. The world writes to a durable log, so
what the dead process sent is still visible to the process that resumes — which is the whole
point: a resumed run that could not see the dead one's effects could not avoid re-sending them.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench._kill_points import arm, is_kill_point, watch
from bench.demo import (
    BLOCK_MS,
    TURN_1,
    TURN_2,
    TURN_MS,
    PastWriteGraph,
    answered_turns,
    demo3_registry,
    demo3_world,
)
from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn


async def main() -> None:
    directory = Path(sys.argv[1])
    run_id = sys.argv[2]
    kill = sys.argv[3]
    delay_ms = -1.0 if is_kill_point(kill) else float(kill)
    resuming = len(sys.argv) > 4 and sys.argv[4] == "resume"

    points = arm(kill if is_kill_point(kill) else "-1")
    world = demo3_world(directory)
    watch(world, points)
    registry = demo3_registry(world)
    journal = Journal(directory / "journal.db")
    # A resumed run is a new process, so its script would otherwise start over while the run
    # does not; skip the turns the journal already answers for (bench.demo.answered_turns).
    already = answered_turns(journal, run_id) if resuming else 0
    model = ScriptedModel(
        turns=[tool_turn(*TURN_1, turn=0), tool_turn(*TURN_2, turn=1)],
        block_delay_ms=BLOCK_MS,
        complete_delay_ms=TURN_MS,
        consumed=already,
    )
    scheduler = Scheduler(
        graph=PastWriteGraph("specunode"),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=1.0),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=True),
    )

    # The killer starts here, not at process start: interpreter startup and imports dominate
    # the process lifetime, so a delay measured from launch almost never lands inside the run
    # -- and a kill demo where nothing is killed is green and worthless.
    if delay_ms >= 0:

        def killer() -> None:
            time.sleep(delay_ms / 1000.0)
            os._exit(9)

        threading.Thread(target=killer, daemon=True).start()

    started = time.monotonic()
    result = await scheduler.resume(run_id) if resuming else await scheduler.run(run_id, {})
    work_ms = (time.monotonic() - started) * 1000.0
    world.close()
    print(
        f"done ok={result.ok} rows={len(result.ledger.rows)} work_ms={work_ms:.3f} "
        f"{points.report()}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
