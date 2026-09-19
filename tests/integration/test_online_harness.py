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

import asyncio
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


def test_every_workload_sends_the_model_the_bench_was_told_to_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real run must not send ``"model": "scripted"`` to the Messages API.

    All four sample apps hard-coded the stand-in's id, and nothing rewrites an envelope -- Hard
    Rule 13 makes the request the unit of identity, so the runtime must not. The first real call
    of task 6.4 would therefore have been rejected by the provider, after the credential was
    obtained and the money spent.

    Asserted on the envelopes each workload actually sends, not on the helper they are supposed
    to call: a first version of this test imported ``target_model`` and called it, and passed
    just as happily with every ``model=`` argument back to its hard-coded string.
    """
    import asyncio as _asyncio

    from bench.online.run_latency import ARMS, MeteredModel, Spend, run_one
    from bench.online.scripted_target import ScriptedTarget
    from bench.workloads import WORKLOADS

    from specunode.core.model import RequestEnvelope

    sentinel = "claude-model-under-test"
    monkeypatch.setenv("SPECUNODE_MODEL", sentinel)
    seen: list[str] = []

    class Recording(ScriptedTarget):  # type: ignore[misc]
        async def complete(self, envelope: RequestEnvelope) -> object:
            seen.append(envelope.model)
            return await super().complete(envelope)

        def stream(self, envelope: RequestEnvelope) -> object:
            seen.append(envelope.model)
            return super().stream(envelope)

    target = MeteredModel(inner=Recording(), spend=Spend(cap_usd=1.0))
    for workload in WORKLOADS:
        _asyncio.run(run_one(workload, ARMS[0], 0, target, tmp_path))

    assert len(seen) >= len(WORKLOADS), f"only {len(seen)} request(s) recorded"
    assert set(seen) == {sentinel}, (
        f"a workload asked for {sorted(set(seen) - {sentinel})} instead of the model the bench "
        "was told to use"
    )


def test_a_real_run_refuses_the_stand_ins_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mistake above, made impossible to repeat by accident.

    The key is set so the refusal cannot be the credential check standing in for the one under
    test -- an earlier version of this assertion was satisfied by the "ANTHROPIC_API_KEY is not
    set" message, which happens to contain the word "scripted".
    """
    from bench.online.run_latency import main as latency_main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    with pytest.raises(SystemExit) as raised:
        asyncio.run(latency_main(["--model", "anthropic", "--target-model", "scripted"]))
    assert "Messages API" in str(raised.value), str(raised.value)


class _AnswersInProse:
    """A model that replies in prose and never calls a tool -- what a real one does sometimes."""

    def _response(self, envelope: object) -> object:
        from specunode.core.model import ModelResponse, TextBlock, Usage

        return ModelResponse(
            model=getattr(envelope, "model", "scripted"),
            content=(TextBlock(text="I'd need more detail before charging anything."),),
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=20),
        )

    async def complete(self, envelope: object) -> object:
        return self._response(envelope)

    async def stream(self, envelope: object) -> object:  # pragma: no cover - not used here
        from specunode.core.model import TurnComplete

        yield TurnComplete(response=self._response(envelope))


@pytest.mark.parametrize("name", ["support_agent", "research_agent"])
def test_a_model_that_answers_in_prose_ends_the_run_and_sends_nothing(
    tmp_path: Path, name: str
) -> None:
    """Two defects in one, and the second is the dangerous one.

    A turn with no ``tool_use`` block is a ``FreeText`` barrier. The sample apps recorded a
    decision only when it was a ``ToolCall``, and their routers keyed on that state, so a
    prose answer sent the run back to the same node forever -- one model call per lap, which
    against a real provider is an unbounded bill rather than a hang. It is not hypothetical:
    the first real call this project ever made came back as prose, because a model that thinks
    by default spent its whole ``max_tokens`` budget before reaching a tool call.

    The obvious repair -- record ``None`` and carry on -- was worse: the next node fell back to
    its default arguments and charged a card the model had never asked to charge. So the
    router stops instead, and the assertion here is that the world stays empty.
    """
    from bench.online.run_latency import ARMS, MeteredModel, Spend, run_one
    from bench.workloads import WORKLOADS

    workload = next(w for w in WORKLOADS if w.name == name)
    target = MeteredModel(inner=_AnswersInProse(), spend=Spend(cap_usd=1.0))
    result = asyncio.run(run_one(workload, ARMS[0], 0, target, tmp_path))

    assert result.effects == 0, f"{name} put {result.effects} effect(s) in the world uninvited"
    assert target.calls == 1, f"the model was asked {target.calls} times for one declined turn"


def _tier2(tmp_path: Path, *extra: str) -> dict[str, object]:
    out = tmp_path / "tier2.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "bench" / "online" / "run_tier2_acceptance.py"),
            "--model",
            "scripted",
            "--sample",
            "40",
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


@pytest.mark.skipif(
    not (REPO / "bench" / "corpus" / "values.json").is_file(),
    reason="no values sidecar; run bench/corpus/fetch.py --values",
)
def test_the_tier_2_harness_runs_without_a_credential(tmp_path: Path) -> None:
    """The lesson the MCP proxy taught: a benchmark nobody can run is a benchmark that does not
    work. Everything but the provider's own HTTP call is exercised here."""
    report = _tier2(tmp_path)
    assert report["is_real_model"] is False
    assert report["draft_model"] == "scripted"
    result = report["result"]
    assert isinstance(result, dict)
    assert result["steps_graded"] == 40
    assert 0.0 <= float(result["acceptance_rate"]) <= 1.0
    # The right tool with the wrong arguments is the gap the whole measurement is about, so it
    # must be reported, not folded into a single number.
    assert float(result["right_tool"]) >= float(result["acceptance_rate"])


@pytest.mark.skipif(
    not (REPO / "bench" / "corpus" / "values.json").is_file(),
    reason="no values sidecar; run bench/corpus/fetch.py --values",
)
def test_the_tier_2_sample_is_reproducible_and_the_seed_moves_it(tmp_path: Path) -> None:
    """A sample nobody can redraw is a number nobody can check."""
    from bench.offline.run_acceptance import load_joined
    from bench.online.run_tier2_acceptance import SEED, sample_positions

    trajectories, _ = load_joined(
        REPO / "bench" / "corpus" / "traces.json", REPO / "bench" / "corpus" / "values.json"
    )
    first = sample_positions(trajectories, 40, SEED)
    assert first == sample_positions(trajectories, 40, SEED)
    assert first != sample_positions(trajectories, 40, SEED + 1)
    assert len(first) == 40
    for index, position in first:
        assert 0 <= position < len(trajectories[index]) - 1, "a sampled step has no successor"


def test_the_sweep_gives_every_arm_the_same_tool_latency(tmp_path: Path) -> None:
    """A sweep that slowed only reads would flatter the store buffer, whose claim is writes."""
    from bench.online.run_latency_sweep import with_latency
    from bench.workloads import WORKLOADS

    from specunode.testing.world import standard_world

    workload = WORKLOADS[0]
    assert with_latency(workload, 0).prepare is None, "zero latency should leave the world alone"

    slowed = with_latency(workload, 250)
    world = standard_world()
    slowed.make(world)
    assert world.faults.slow_read_ms == 250
    assert world.faults.slow_write_ms == 250
    # The original is untouched: a sweep that mutated its own input would carry one rung's
    # latency into the next.
    untouched = standard_world()
    workload.make(untouched)
    assert untouched.faults.slow_read_ms == 0


def test_the_sweep_runs_every_rung_and_reports_a_difference(tmp_path: Path) -> None:
    """The crossing point is the answer, so every rung has to carry an interval, not a point."""
    out = tmp_path / "sweep.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "bench" / "online" / "run_latency_sweep.py"),
            "--model",
            "scripted",
            "--tasks",
            "2",
            "--ladder",
            "0,25",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["is_real_model"] is False
    assert report["halted_at"] is None, report["halted_at"]
    assert sorted(report["points"]) == ["0", "25"]
    for rung in report["points"].values():
        for entry in rung.values():
            assert entry["leaks"] == 0
            band = entry["saving_vs_seq_ci95"]["B_specunode"]
            assert band["ci95_low"] <= band["mean"] <= band["ci95_high"]
    # The slower rung must actually be slower, or the latency never reached the tools.
    fast = report["points"]["0"]["support_agent"]["wall_ms_mean"]["B_seq"]
    slow = report["points"]["25"]["support_agent"]["wall_ms_mean"]["B_seq"]
    assert slow > fast, f"latency had no effect: {fast}ms then {slow}ms"
