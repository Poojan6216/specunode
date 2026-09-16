"""The chaos and concurrency matrices (spec tasks 5.2 and 5.3).

Both are run as benchmarks so a nightly job produces numbers, and asserted here so a regression
is caught on the change that caused it rather than the next morning.

The concurrency half asks two separate questions and must not conflate them. Many runs through
one *journal* is a question about whether interleaved writers corrupt each other -- their
ledgers must be identical to a solo run's. Many runs through one *world* is a question about
attribution, and there the ledgers are *supposed* to differ, because the runs genuinely interact
and each charge gets its own id. Comparing the second case against a solo run measures nothing
and reports a leak that is not one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.chaos.run_chaos import run_chaos, run_concurrency  # noqa: E402


@pytest.mark.slow
async def test_many_runs_through_one_journal_do_not_disturb_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard Rule 6 and the single-writer journal, under interleaving."""
    monkeypatch.chdir(tmp_path)
    report = await run_concurrency(runs=8, out=None)

    assert report.leaks == 0, "a branch reached the world without retiring in its own run"
    assert report.cross_branch_observations == 0, "two runs claimed the same branch identity"
    assert report.equivalence_failures == 0, (
        "sharing a journal changed what a run did; interleaved ledgers differ from a solo one"
    )
    assert report.duplicate_deliveries == 0
    assert not report.notes, report.notes


@pytest.mark.slow
async def test_faults_during_a_drain_never_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partitions, duplicate deliveries and slow reads, with the leak invariant after each."""
    monkeypatch.chdir(tmp_path)
    report = await run_chaos(rounds=6, out=None)

    assert report.leaks == 0
    assert report.equivalence_failures == 0
    assert not report.notes, report.notes
    # A partition mid-drain must dead-letter rather than half-apply, so seeing none at all
    # would mean the faults were never actually injected.
    assert report.dead_letters > 0, "no fault ever fired; the matrix exercised nothing"
