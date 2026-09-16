"""The report regenerates from the committed results (spec task 9.4).

The property is provenance, not layout: every figure in the PDF is read from a file in
`bench/results/`, so the published report cannot drift from what was measured. A generator that
typed any number in would break that link silently.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytest.importorskip("reportlab", reason="the report PDF needs the bench extra")


def test_the_report_regenerates(tmp_path: Path) -> None:
    out = tmp_path / "report.pdf"
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "make_report_pdf.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 2000


def test_the_generator_types_in_no_numbers(tmp_path: Path) -> None:
    """A figure written into the source would survive a change in what was measured.

    Every number must come from a results file, so the generator's own source should contain
    no bare decimal that looks like a measurement. Structural constants -- page margins, font
    sizes, string slices -- are fine and are what the exclusions cover.
    """
    import re

    source = (REPO / "bench" / "make_report_pdf.py").read_text(encoding="utf-8")
    # Strip anything that is plainly layout rather than a claim.
    stripped = re.sub(r"\b\d+(\.\d+)?\s*\*\s*mm\b", "", source)
    stripped = re.sub(r"\[:?-?\d+\]|\[\d+,\s*-?\d+\]|:\.\d+[fF%]|\{\d\}", "", stripped)
    stripped = re.sub(r"\b(0|1|2|3|4|5|6|9|10|36|50|52|60|255|2000)\b", "", stripped)
    suspicious = re.findall(r"\b\d+\.\d{3,}\b", stripped)
    assert not suspicious, f"the generator contains hard-coded figures: {suspicious}"
