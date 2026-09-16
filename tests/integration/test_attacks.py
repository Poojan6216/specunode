"""The adversarial suite is part of the build, not a document (spec Phase 7).

The README's "What beats it" section is generated from these strategies rather than written
from memory, so the properties asserted here are about the *suite* rather than about the
runtime: every strategy runs, every one reports a measured rate, and the ones that are supposed
to beat the runtime still do.

That last clause is the one that matters. If a strategy silently stops defeating the runtime --
because a detector improved, or because the attack stopped exercising what it claims to -- the
honest response is to find out and say so, not to keep publishing a hole that is no longer
there or a clean sheet that was never earned.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "bench" / "adversarial" / "run_attacks.py"


def run_suite(tmp_path: Path) -> dict[str, dict[str, object]]:
    out = tmp_path / "attacks.json"
    result = subprocess.run(
        [sys.executable, str(SUITE), "--all", "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=180,
    )
    assert result.returncode == 0, f"a strategy failed to run:\n{result.stdout[-3000:]}"
    payload = json.loads(out.read_text())
    return {entry["id"]: entry for entry in payload["attacks"]}


def test_every_strategy_runs_or_says_why_not(tmp_path: Path) -> None:
    """A suite that drops the attacks it cannot execute reports a clean sheet for a bad reason."""
    attacks = run_suite(tmp_path)
    assert len(attacks) >= 8
    for entry in attacks.values():
        assert entry["error"] is None, f"{entry['id']} failed: {entry['error']}"
        assert entry["measured"], f"{entry['id']} produced no measurement"
        assert entry["claim"], f"{entry['id']} states no claim"


@pytest.mark.parametrize(
    "attack_id",
    ["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.10"],
)
def test_the_known_holes_are_still_holes(tmp_path: Path, attack_id: str) -> None:
    """Each of these defeats the runtime, and the README says so.

    If one stops defeating it, this fails -- which is the right outcome: either the runtime
    genuinely improved and the documented limitation should be removed, or the attack stopped
    testing what it claims to. Both need a human to look.
    """
    entry = run_suite(tmp_path)[attack_id]
    assert entry["defeated_runtime"], (
        f"{attack_id} no longer defeats the runtime. Either a real hole was closed and "
        f"docs/limitations.md should drop it, or the attack stopped exercising it."
    )


def test_a_misdeclared_read_leaks_from_a_squashed_branch(tmp_path: Path) -> None:
    """7.1, the trust boundary, stated as a number rather than as a caveat."""
    measured = run_suite(tmp_path)["7.1"]["measured"]
    assert measured["effects_from_squashed_branch"] >= 1  # type: ignore[index]


def test_the_undetectable_stale_read_fraction_is_reported(tmp_path: Path) -> None:
    """7.3's honest number: not the stale rate, but the share nothing could have caught."""
    measured = run_suite(tmp_path)["7.3"]["measured"]
    assert measured["unwitnessed_and_therefore_undetectable"] >= 1  # type: ignore[index]
    assert 0.0 <= float(measured["undetectable_fraction"]) <= 1.0  # type: ignore[arg-type]


def test_the_laundering_matrix_reports_both_halves(tmp_path: Path) -> None:
    """Some transforms are caught and some are not; publishing only one half is not a result."""
    measured = run_suite(tmp_path)["7.5"]["measured"]
    assert measured["caught"] >= 1 and measured["missed"] >= 1  # type: ignore[operator]
    assert "base64" in str(measured["missed_cases"])


def test_the_docs_and_the_suite_agree_about_what_is_missed(tmp_path: Path) -> None:
    """A hazard table written from memory drifts from the detector it describes."""
    measured = run_suite(tmp_path)["7.5"]["measured"]
    hazards = (REPO / "docs" / "hazards.md").read_text(encoding="utf-8")
    missed = str(measured["missed_cases"]).lower()
    if "base64" in missed:
        assert "base64-encoded | **no**" in hazards
    if "hex" in missed:
        assert "hex- or percent-encoded | **no**" in hazards


def test_the_committed_results_match_a_fresh_run(tmp_path: Path) -> None:
    """Hard Rule 12: the numbers in the repo are the ones the command produces today."""
    committed = json.loads((REPO / "bench" / "results" / "attacks.json").read_text())
    fresh = {entry["id"]: entry for entry in committed["attacks"]}
    assert fresh.keys() == run_suite(tmp_path).keys()
    for attack_id, entry in run_suite(tmp_path).items():
        assert entry["measured"] == fresh[attack_id]["measured"], (
            f"{attack_id}'s committed numbers differ from a fresh run"
        )
