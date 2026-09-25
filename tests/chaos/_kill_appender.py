"""Subprocess helper: append entries, then die mid-append without cleanup.

Run as ``python _kill_appender.py <db> <run_id> <count> <kill_after> <fraction>``. Once
``kill_after`` appends have returned, a daemon thread is armed that calls ``os._exit`` after
``fraction`` of this process's own mean append time -- usually while the next append is inside
SQLite's write. ``os._exit`` skips atexit hooks, buffer flushes and destructors: the closest a
test can get to the power going out while SQLite is inside a write. ``kill_after`` of -1 never
arms it.

The kill used to be a delay from the start of the loop, calibrated against a timed run. On a
CI runner that put most delays past the end of the loop on one run and let every process finish
on another, so whether the property was exercised at all depended on the machine. Arming the
killer by count means every run is killed inside the loop, with hundreds of appends to spare.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from specunode.journal.journal import Journal


def die_after(seconds: float) -> None:
    time.sleep(seconds)
    os._exit(9)


def main() -> None:
    db, run_id, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    kill_after, fraction = int(sys.argv[4]), float(sys.argv[5])

    journal = Journal(db)
    started = time.monotonic()
    for index in range(count):
        if index == kill_after:
            mean_s = (time.monotonic() - started) / max(index, 1)
            threading.Thread(target=die_after, args=(fraction * mean_s,), daemon=True).start()
        journal.append(
            run_id, "policy_event", {"v": 1, "event": "tick", "reason": "kill-test", "i": index}
        )
    print(f"completed {count}", flush=True)


if __name__ == "__main__":
    main()
