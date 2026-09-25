"""Kill the process mid-append and prove the journal survives it (spec task 0.3's Verify).

The property under test is the one every other guarantee stands on: **the last entry is
wholly present or wholly absent, never partial, and the chain still verifies.** If a torn
entry were possible, a resume could read a half-written model response and act on it, and
Hard Rule 5 would be decoration.

The mechanism that delivers it is that one entry is one row is one statement is one commit.
SQLite appends the row's frames plus a commit frame to the WAL; at ``synchronous=FULL`` the
fsync happens before ``execute()`` returns; and on the next open, WAL recovery discards any
trailing frames that have no valid commit frame. Because the unit of commit is exactly one
entry, "the last committed transaction" and "the last entry" are the same thing.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from specunode.journal.journal import Journal

HELPER = Path(__file__).with_name("_kill_appender.py")
RUN = "01KILLRUNAAAAAAAAAAAAAAAAA"
ENTRIES = 1000


def _run_appender(
    db: Path, delay_ms: float, count: int = ENTRIES
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), str(db), RUN, str(count), str(delay_ms)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def _kill_window_ms(tmp_path: Path) -> float:
    """How long the append loop takes on this machine: the span a kill must land inside.

    Measured by the helper from where its killer's clock starts, and taken from the fastest of
    three runs. Timing the whole subprocess counted interpreter start-up, which the killer never
    sees, and a single warm-up is the slowest run on a cold machine; on CI the two together put
    most delays past the end of the loop, and only 3 of 15 processes were killed.
    """
    windows = []
    for warm in range(3):
        result = _run_appender(tmp_path / f"warmup-{warm}.db", delay_ms=-1)
        assert result.returncode == 0, result.stderr[-600:]
        for token in result.stdout.split():
            if token.startswith("work_ms="):
                windows.append(float(token.split("=", 1)[1]))
    assert len(windows) == 3, "the helper did not report how long its loop took"
    return min(windows)


def _assert_intact(db: Path, *, expect_at_most: int = ENTRIES) -> int:
    """The chain verifies, offsets are dense, and every payload is complete."""
    journal = Journal(db)
    result = journal.verify_chain(RUN)
    assert result.ok, (
        f"chain broken at offset {result.first_bad_offset}: {result.reason} {result.detail}"
    )

    entries = list(journal.read(RUN))
    assert [e.offset for e in entries] == list(range(len(entries))), "offsets are not dense"
    assert len(entries) <= expect_at_most

    # A partial write would show up as a payload that does not parse, does not hash, or is
    # missing the counter. verify_chain already checks the first two; this checks the third,
    # and that no entry was skipped -- both of which a torn row would break.
    assert [e.payload["i"] for e in entries] == list(range(len(entries))), (
        "an entry is missing or out of order, which means a write was not atomic"
    )
    return len(entries)


def test_an_uninterrupted_run_writes_every_entry(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    assert _run_appender(db, delay_ms=-1).returncode == 0
    assert _assert_intact(db) == ENTRIES


@pytest.mark.slow
def test_killing_mid_append_never_leaves_a_partial_entry(tmp_path: Path) -> None:
    # Calibrate against this machine, so the kills land inside the write loop rather than
    # before it starts or after it finishes.
    window_ms = _kill_window_ms(tmp_path)
    assert window_ms > 20, "appends are too fast to land a kill inside one; raise ENTRIES"

    rng = random.Random(20260915)
    survivors: list[int] = []
    killed = 0
    for attempt in range(15):
        db = tmp_path / f"kill-{attempt}.db"
        delay_ms = rng.uniform(0.2, window_ms * 0.9)
        result = _run_appender(db, delay_ms=delay_ms)
        if result.returncode != 0:
            killed += 1
        survivors.append(_assert_intact(db))

    assert killed >= 5, (
        f"only {killed}/15 subprocesses were actually killed mid-run; the test would be "
        "green without exercising the property it exists to test"
    )
    assert any(0 < n < ENTRIES for n in survivors), (
        "no kill landed inside the append loop, so no partial-write window was exercised"
    )


@pytest.mark.slow
def test_a_killed_journal_can_be_reopened_and_appended_to(tmp_path: Path) -> None:
    """Recovery is not just readable: the chain must continue from where it stopped."""
    db = tmp_path / "journal.db"
    _run_appender(db, delay_ms=_kill_window_ms(tmp_path) * 0.4)  # kill roughly mid-loop
    before = _assert_intact(db)

    journal = Journal(db)
    for index in range(10):
        offset = journal.append(
            RUN,
            "policy_event",
            {"v": 1, "event": "tick", "reason": "after-kill", "i": before + index},
        )
        assert offset == before + index, "the offset counter did not resume from the journal"
    assert journal.verify_chain(RUN).ok
    assert _assert_intact(db, expect_at_most=before + 10) == before + 10
