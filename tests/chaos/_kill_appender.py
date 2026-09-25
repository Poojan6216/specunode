"""Subprocess helper: append entries, then die mid-append without cleanup.

Run as ``python _kill_appender.py <db> <run_id> <count> <delay_ms>``. A daemon thread calls
``os._exit`` after ``delay_ms``, which skips atexit hooks, buffer flushes and destructors --
the closest a test can get to the power going out while SQLite is inside a write.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from specunode.journal.journal import Journal


def main() -> None:
    db, run_id, count, delay_ms = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])

    def killer() -> None:
        time.sleep(delay_ms / 1000.0)
        os._exit(9)

    # Timed from where the killer's clock starts, so the test can calibrate its kill window
    # against the same span -- not against interpreter start-up, which the killer never sees.
    started = time.monotonic()
    if delay_ms >= 0:
        threading.Thread(target=killer, daemon=True).start()

    journal = Journal(db)
    for index in range(count):
        journal.append(
            run_id, "policy_event", {"v": 1, "event": "tick", "reason": "kill-test", "i": index}
        )
    work_ms = (time.monotonic() - started) * 1000.0
    print(f"completed {count} work_ms={work_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
