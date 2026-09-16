"""Demo 1 is an acceptance test, not marketing (spec section 2).

The spec is explicit that the demos have to work from the published package, and that every
number they print comes from the run rather than from a table someone wrote. So the demo is
exercised here the same way a reader would exercise it, and the column that carries the claim
is asserted rather than eyeballed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run_demo() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "demo.py"), "--demo", "leak", "--json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def arm(report: dict[str, object], name: str) -> dict[str, int]:
    arms = report["arms"]
    assert isinstance(arms, list)
    for entry in arms:
        if entry["runtime"] == name:
            return entry
    raise AssertionError(f"no {name} arm in the demo output")


def test_the_naive_runtime_leaks_and_the_demo_says_so() -> None:
    """Without a store buffer, a wrong guess has already charged the card.

    This arm exists to fail, and if it ever stops failing the demo has stopped demonstrating
    anything -- so a leak count of zero here is itself a failure.
    """
    naive = arm(run_demo(), "naive-parallel")
    assert naive["effects_from_squashed_branches"] > 0
    assert naive["effects_reaching_world"] > naive["runs"]


def test_specunode_reaches_the_world_exactly_as_the_sequential_run_does() -> None:
    """The second and third rows agreeing is the claim."""
    report = run_demo()
    spec = arm(report, "specunode")
    sequential = arm(report, "sequential")
    assert spec["effects_from_squashed_branches"] == 0
    assert spec["effects_reaching_world"] == sequential["effects_reaching_world"]


def test_the_speculative_arm_really_mispredicted(tmp_path: Path) -> None:
    """Otherwise it reached the sequential row by never having speculated at all."""
    spec = arm(run_demo(), "specunode")
    assert spec["mispredictions"] > 0
    assert spec["staged_and_discarded"] == spec["mispredictions"]


def test_the_committed_results_match_a_fresh_run() -> None:
    """Hard Rule 12: the numbers in the repo are the ones the command produces today."""
    committed = json.loads((REPO / "bench" / "results" / "demo_leak.json").read_text())
    assert committed == run_demo()
