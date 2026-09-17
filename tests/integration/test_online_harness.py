"""The online latency harness, run without a credential (spec task 6.4).

This runner did not exist for most of the build while the Final Report described it as written
and blocked only on an API key -- in a project whose stated purpose is to make exactly that
class of claim impossible. It exists now, and this test runs the whole of it: three workloads,
three arms, the timing, the spend accounting, the budget gate and the bootstrap.

What it deliberately does not do is measure latency. ``--model scripted`` is a deterministic
stand-in, so these numbers are not figures anyone should quote and the runner's own output says
so on every report it writes. The point is narrower and was learned from the MCP proxy, which
shipped unable to start because only its rules were tested: a benchmark nobody can execute
without a credit card is a benchmark nobody has executed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "bench" / "online" / "run_latency.py"

pytestmark = pytest.mark.slow


def run(tmp_path: Path, *extra: str) -> dict[str, object]:
    out = tmp_path / "latency.json"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--model",
            "scripted",
            "--tasks",
            "2",
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_runner_exists_where_every_document_says_it_does() -> None:
    """RESULTS.md, bench/report.py and the nightly workflow all name this path."""
    assert RUNNER.is_file(), f"{RUNNER} is referenced by three documents and does not exist"


def test_the_harness_runs_every_workload_under_every_arm(tmp_path: Path) -> None:
    report = run(tmp_path)
    from bench.online.run_latency import ARMS
    from bench.workloads import WORKLOADS

    workloads = report["workloads"]
    assert isinstance(workloads, dict)
    assert set(workloads) == {w.name for w in WORKLOADS}
    for name, entry in workloads.items():
        assert set(entry["arms"]) == set(ARMS), f"{name} did not run every arm"
        for arm, stats in entry["arms"].items():
            assert stats["n"] == 2, f"{name}/{arm} ran {stats['n']} tasks"
            low, high = stats["wall_ms_ci95"]
            assert low <= stats["wall_ms_mean"] <= high, "the interval does not contain the mean"


def test_a_scripted_run_is_labelled_as_not_a_measurement(tmp_path: Path) -> None:
    """The one property that keeps this from becoming the claim it replaced."""
    report = run(tmp_path)
    assert report["is_real_model"] is False
    assert report["model"] == "scripted"


def test_no_arm_leaks(tmp_path: Path) -> None:
    """Hard Rule 3 across the whole matrix. Section 6.4 says this must be zero."""
    report = run(tmp_path)
    assert report["total_leaks"] == 0
    for entry in report["workloads"].values():  # type: ignore[union-attr]
        for stats in entry["arms"].values():
            assert stats["leaks"] == 0


def test_every_arm_reaches_the_same_world(tmp_path: Path) -> None:
    """Hard Rule 9 in the shape this bench can see it: the effect count must not move."""
    report = run(tmp_path)
    for name, entry in report["workloads"].items():  # type: ignore[union-attr]
        counts = {arm: tuple(stats["effects"]) for arm, stats in entry["arms"].items()}
        assert len(set(counts.values())) == 1, f"{name}: arms disagree about effects: {counts}"


def test_the_budget_gate_halts_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision Gate D2: report the reduced n, never a raised cap.

    Driven through the real gate rather than a unit test of the arithmetic, because the thing
    worth knowing is that the runner stops and says where -- not that a float comparison works.
    """
    out = tmp_path / "capped.json"
    env = dict(os.environ)
    env["SPECUNODE_BENCH_BUDGET_USD"] = "0.0000001"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--model", "scripted", "--tasks", "5", "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["halted_at"], "the cap was far below one call's cost and nothing halted"
    assert report["tasks_completed"] < 5 * 3 * 3
    assert "Decision Gate D2" in result.stdout
