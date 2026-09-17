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

import pytest

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


# -- Demo 2 ------------------------------------------------------------------------------------
#
# Nothing here asserts a wall-clock figure. Demo 2's timings are measured on the machine that
# runs it and vary between runs, so pinning one would either be flaky or be a number nobody
# measured. What is asserted is everything that is *not* a timing: the three arms agree about
# what reached the world, the specunode arm really overlapped the read, and the two baselines
# really did not.


def run_past_write() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "demo.py"), "--demo", "past-write", "--json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_all_three_arms_change_the_world_identically() -> None:
    """The claim the whole design rests on: faster, and identical in effect."""
    report = run_past_write()
    assert report["worlds_identical"] is True
    digests = {a["world_digest"] for a in report["arms"]}  # type: ignore[union-attr]
    assert len(digests) == 1
    for entry in report["arms"]:  # type: ignore[union-attr]
        assert entry["effects_reaching_world"] == 2


def test_the_specunode_arm_really_overlapped_the_read() -> None:
    """Without this, an arm that simply ran faster by accident would satisfy the table."""
    report = run_past_write()
    assert isinstance(report["read_overlapped_ms"], (int, float))
    assert report["read_overlapped_ms"] > 0, (
        "no part of the independent read ran while the write was staged, so the demo's "
        "explanation of where its saving comes from is not what happened"
    )


def test_the_readonly_baseline_did_not_run_ahead() -> None:
    """PASTE's rule, exercised rather than asserted in prose.

    Turn 1 emits the write first, so a runtime that stops at the first tool with side effects
    has nothing before it to run ahead into. If the read ever starts before the write finishes
    in that arm, the baseline is not implementing the rule it is named for.
    """
    report = run_past_write()
    arms = {a["runtime"]: a for a in report["arms"]}  # type: ignore[union-attr]
    calls = {c["name"]: c for c in arms["readonly-spec"]["calls"]}
    assert calls["fetch_runbook"]["started_ms"] >= calls["restart_job"]["finished_ms"]


def test_the_demo_reports_its_injected_latencies() -> None:
    """A timeline whose inputs are hidden is a drawing rather than a measurement."""
    injected = run_past_write()["injected_latency_ms"]
    assert isinstance(injected, dict)
    assert set(injected) == {"read", "write", "stream_block", "model_turn"}
    assert all(value > 0 for value in injected.values())


# -- Demo 3 ------------------------------------------------------------------------------------
#
# Demo 3 kills a real subprocess, so it is marked slow. What is asserted is the set of claims
# the demo makes in prose: the kill landed, the resume neither duplicated nor invented, the
# journal still verifies, replay refuses a changed prompt *with the diff*, and replay with
# speculation off completes.


def run_replay_demo() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "demo.py"), "--demo", "replay", "--json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


@pytest.mark.slow
def test_the_demo_actually_kills_the_process() -> None:
    """A kill demo that killed nothing would print every claim below and mean none of them."""
    report = run_replay_demo()
    assert report["process_was_killed"] is True
    delay, work = report["kill_delay_ms"], report["work_ms"]
    assert isinstance(delay, (int, float)) and isinstance(work, (int, float))
    assert 0 < delay < work, "the kill was not scheduled inside the run's own work"


@pytest.mark.slow
def test_the_resume_neither_duplicates_nor_invents() -> None:
    """The two properties that matter, stated the way the chaos suite states them."""
    report = run_replay_demo()
    assert report["duplicate_deliveries"] == 0
    assert report["resumed_is_prefix_of_clean"] is True
    assert report["journal_chain_verifies_after_kill"] is True


@pytest.mark.slow
def test_replay_refuses_a_changed_prompt_and_says_what_changed() -> None:
    """Refusing is half of it. The spec asks for the step index and the diff, so check both.

    A refusal that only said "node act failed" would satisfy the boolean and be useless to the
    operator it exists for -- which is exactly what this reported before the scheduler was
    changed to carry the branch's failure reason out of the drive loop.
    """
    report = run_replay_demo()
    assert report["replay_with_changed_prompt_refused"] is True
    refusal = report["replay_refusal"]
    assert isinstance(refusal, str)
    assert "diverged at step" in refusal
    assert "system:" in refusal, "the refusal did not name the field that changed"
    assert "cautious operator" in refusal, "the refusal did not show the new value"


@pytest.mark.slow
def test_replay_with_speculation_off_completes() -> None:
    report = run_replay_demo()
    assert report["replay_with_speculation_off_ok"] is True
    assert report["replay_ledger_digest"]


def test_demo_ones_specunode_row_is_produced_by_the_real_runtime() -> None:
    """The front page's evidence table has to exercise the thing it is evidence for.

    This row used to be a conditional in ``bench/baselines.py``, which by design cannot import
    ``specunode.buffer`` -- the naive-parallel baseline has to be *able* to leak or the demo
    measures nothing. That isolation is right for the baselines and meant SpecuNode's own row
    was a description of a store buffer rather than a run of one. The numbers did not change
    when it was rewired, which is the point: they were not wrong, they were just not
    measurements of this software.

    Asserted structurally rather than by eye, because "did this number come from the product?"
    is exactly the question a table cannot answer about itself.
    """
    import bench.real_arm as real_arm

    source = (REPO / "bench" / "real_arm.py").read_text(encoding="utf-8")
    for required in (
        "from specunode.core.scheduler import Scheduler",
        "from specunode.buffer.store_buffer import StoreBuffer",
        "from specunode.buffer.dispatcher import Dispatcher",
    ):
        assert required in source, f"the specunode arm no longer uses the real runtime: {required}"
    assert hasattr(real_arm, "run_specunode_arm")

    # And the counts it reports come from the runtime's own counters, not from the arm's
    # arithmetic: a discarded effect is one the store buffer actually held back.
    spec = arm(run_demo(), "specunode")
    assert spec["staged_and_discarded"] == spec["mispredictions"] > 0
    assert spec["effects_from_squashed_branches"] == 0
