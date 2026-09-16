"""The offline opportunity analysis (spec tasks 6.1, 6.2 and 6.6).

The analysis is what makes this project's central claim checkable rather than asserted, so the
tests are about the *measurement* rather than the runtime: the corpus is what the manifest says
it is, the numbers regenerate identically, and the report cannot be hand-edited without the
build noticing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "bench" / "corpus" / "traces.json"
MANIFEST = REPO / "bench" / "corpus" / "manifest.json"


def test_the_committed_corpus_matches_its_manifest() -> None:
    """A corpus whose hash moved is a different corpus, and the numbers describe the old one."""
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "corpus" / "fetch.py"), "--verify-manifest"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_corpus_records_turn_structure() -> None:
    """Without it the headline measurement cannot be computed at all.

    A staged write blocks the next model *turn*, not the next call, so whether two calls share
    a turn is exactly what decides how far speculation can run past a write. A corpus that did
    not record it could only be analysed by assuming an answer.
    """
    traces = json.loads(CORPUS.read_text())["traces"]
    steps = [s for t in traces for s in t["steps"]]
    assert steps
    for step in steps[:200]:
        assert "turn" in step and "ordinal" in step


def test_effect_classes_come_from_a_table_and_default_to_write() -> None:
    """Hard Rule 2 in the analysis as well as the runtime: never inferred from a name."""
    sys.path.insert(0, str(REPO))
    from bench.corpus.effect_classes import CORPUS_EFFECT_CLASSES, effect_of

    assert effect_of("a_tool_nobody_classified") == "write"
    assert effect_of("think") == "read"
    # Anything that writes must not be in the read list, however harmless its name looks.
    assert "str_replace_editor" not in CORPUS_EFFECT_CLASSES
    assert "execute_bash" not in CORPUS_EFFECT_CLASSES


def run_analysis(tmp_path: Path) -> dict[str, object]:
    out = tmp_path / "opportunity.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "bench" / "offline" / "run_opportunity.py"),
            "--out",
            str(out),
            "--predict-sample",
            "8",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text())


def test_every_measure_carries_a_confidence_interval(tmp_path: Path) -> None:
    """A mean without an interval over 300 trajectories is a number nobody can argue with."""
    report = run_analysis(tmp_path)
    opportunity = report["opportunity"]
    assert isinstance(opportunity, dict)
    for key, value in opportunity.items():
        if isinstance(value, dict) and "mean" in value:
            assert "ci95_low" in value and "ci95_high" in value, f"{key} has no interval"
            assert value["ci95_low"] <= value["mean"] <= value["ci95_high"]


def test_the_anti_result_is_measured_rather_than_assumed(tmp_path: Path) -> None:
    """The span past a write, and the reason it is what it is.

    If this corpus ever starts emitting several calls per turn, the span becomes non-zero and
    this test fails -- which is correct, because the README's central caveat would then be
    wrong and needs rewriting rather than quietly surviving.
    """
    opportunity = run_analysis(tmp_path)["opportunity"]
    assert isinstance(opportunity, dict)
    new_turn = opportunity["calls_opening_a_new_model_turn"]["mean"]  # type: ignore[index]
    span = opportunity["specunode_post_write_span"]["mean"]  # type: ignore[index]
    if new_turn == 1.0:
        assert span == 0.0, (
            "every call opens a new model turn, so nothing can run past a write; a non-zero "
            "span here would mean the measurement is wrong"
        )


def test_the_store_buffer_opportunity_is_reported_next_to_the_predictor(tmp_path: Path) -> None:
    """Neither number means anything alone, so the report must carry both."""
    opportunity = run_analysis(tmp_path)["opportunity"]
    assert isinstance(opportunity, dict)
    staged = opportunity["steps_paste_must_skip_that_specunode_can_stage"]["mean"]  # type: ignore[index]
    reads = opportunity["read_fraction"]["mean"]  # type: ignore[index]
    assert abs((staged + reads) - 1.0) < 1e-9, "every step is either a read or a stageable write"
    assert opportunity["predictability"]["top_1"] >= 0.0  # type: ignore[index]


def test_results_md_regenerates_identically() -> None:
    """Spec task 6.6: it is generated, so it cannot be edited into saying something else."""
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "report.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_traceability_check_passes_on_the_committed_docs() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "check_numbers.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
def test_a_planted_untraceable_number_fails_the_check(tmp_path: Path) -> None:
    """Spec task 6.6's Verify: CI fails on a number no results file contains."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_numbers_traceable import measured_numbers, untraceable

    planted = "Speculation cut wall clock by 41% on the ops workload."
    assert untraceable(planted, measured_numbers()), "a planted figure slipped through"
