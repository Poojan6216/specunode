"""Hard Rule 12, first half: never report a number you did not measure.

Every figure in ``README.md`` and ``RESULTS.md`` must trace to a value in a committed
``bench/results/*.json``, which in turn was written by a committed command. The proposal this
project was derived from contained a latency reduction, a state-divergence rate and an
adherence gain that nobody had measured; this test is the mechanism that stops that class of
number from reappearing.

Numbers that are *not* claims are excluded, and the exclusion list is closed and explicit:

* figures from published work, which must carry a ``[cited]`` marker on the same line
* version numbers, Python versions, years, dates, arXiv identifiers, URLs, RFC numbers
* ordered-list markers and Markdown heading numbers
* anything on a line preceded by ``<!-- numbers-ok: reason -->``

:mod:`bench.check_numbers` shares this module's logic so that CI and the test agree; this
file is the definition and ``bench/check_numbers.py`` is the command-line front end.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "bench" / "results"
CLAIM_FILES = ("README.md", "RESULTS.md")

# Matches a number a document could be asserting. Comma-grouped forms ("1,842") count as
# one number. The lookbehind and lookahead keep digits that live inside an identifier
# ("blake2b", "test_1_2", "Ed25519") from being read as claims.
_NUMBER = re.compile(
    r"(?<![\w.,$])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:_\d+)*(?:\.\d+)?)(?![\d_,.])"
)
_NUMBERS_OK = re.compile(r"numbers-ok:\s*\S")
_CITED = re.compile(r"\[cited\]", re.IGNORECASE)

#: Spans that are never claims. Order matters: the date rule must precede the year rule, or
#: the year rule eats "2026" out of "2026-09-15" and leaves "09" and "15" looking like
#: measurements. The version rule requires real version context -- a leading "v", three
#: components, or the word "version" -- because a bare "5.6" is a measurement, not a version.
_NOT_A_CLAIM = re.compile(
    r"""
      https?://\S+
    | `[^`]*`
    | arxiv[:\s]*\d{4}\.\d{4,5}
    | \b\d{4}-\d{2}-\d{2}\b
    | \bv\d+(?:\.\d+){1,3}\b
    | \b\d+\.\d+\.\d+\b
    | \b(?:version|release|tag|schema_version)\s*:?\s*\d+(?:\.\d+)*
    | \bpython\s*3\.\d+
    | \b(?:19|20)\d{2}\b
    | \bRFC\s*\d+ | \bPEP\s*\d+
    | ^\s{0,3}(?:\d+[.)]|[-*+])\s
    | ^\s{0,3}\#{1,6}\s.*$
    | \bsection\s+\d+(?:\.\d+)* | \bphase\s+\d+ | \brule\s+\d+ | \btask\s+\d+(?:\.\d+)*
    | \btier\s*\d | \border-\d | \bblake2b-\d+ | \bed25519 | \bbase32 | \bsqlite\s*3
    | \b\d+(?:\.\d+)*\s*(?:st|nd|rd|th)\b
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

#: A bare "3.11" is a Python version when the line is talking about interpreters, and a
#: measurement otherwise. Context decides, rather than a blanket rule that would hide real
#: numbers whose value happens to look like a version.
_INTERPRETER_CONTEXT = re.compile(r"python|interpreter|ci matrix|classifier", re.IGNORECASE)
_INTERPRETER_VERSION = re.compile(r"\b3\.\d{1,2}\b")


def _numeric_strings(value: object, out: set[str]) -> None:
    """Every number reachable in a results document, in the spellings a doc might use."""
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        out.add(str(value))
        out.add(f"{value:,}")
    elif isinstance(value, float):
        out.add(repr(value))
        decimal = Decimal(str(value))
        out.add(str(decimal))
        for places in range(0, 5):
            out.add(f"{value:.{places}f}")
            out.add(f"{value * 100:.{places}f}")  # a rate written as a percentage
        out.add(f"{value:,}")
    elif isinstance(value, str):
        for match in _NUMBER.finditer(value):
            out.add(match.group(1))
    elif isinstance(value, dict):
        for item in value.values():
            _numeric_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _numeric_strings(item, out)


def measured_numbers(results_dir: Path = RESULTS_DIR) -> set[str]:
    """Every number that appears in a committed results file."""
    found: set[str] = set()
    if not results_dir.is_dir():
        return found
    for path in sorted(results_dir.glob("*.json")):
        try:
            _numeric_strings(json.loads(path.read_text(encoding="utf-8")), found)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise AssertionError(f"{path} is not valid JSON: {exc}") from exc
    return {value.lstrip("0") or "0" for value in found} | found


def claimed_numbers(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_number, number, line)`` for every number a document asserts."""
    claims: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        previous = lines[index - 2] if index >= 2 else ""
        if _NUMBERS_OK.search(previous) or _NUMBERS_OK.search(line) or _CITED.search(line):
            continue
        scrubbed = _NOT_A_CLAIM.sub(" ", line)
        if _INTERPRETER_CONTEXT.search(line):
            scrubbed = _INTERPRETER_VERSION.sub(" ", scrubbed)
        for match in _NUMBER.finditer(scrubbed):
            number = match.group(1)
            if number in {"0", "1"}:
                continue  # "one code path, not two"; "0 leaks" is an absence, not a measurement
            claims.append((index, number, line.strip()))
    return claims


def untraceable(text: str, measured: set[str]) -> list[tuple[int, str, str]]:
    normalised = {m.replace(",", "").replace("_", "") for m in measured} | measured
    return [
        claim
        for claim in claimed_numbers(text)
        if claim[1] not in normalised and claim[1].replace("_", "") not in normalised
    ]


def test_every_number_in_a_claim_file_traces_to_a_results_file() -> None:
    measured = measured_numbers()
    problems: list[str] = []
    for name in CLAIM_FILES:
        path = REPO / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, number, line in untraceable(text, measured):
            problems.append(
                f"{name}:{line_number}: {number!r} appears in no results file\n    {line}"
            )
    assert not problems, (
        "Hard Rule 12: never report a number you did not measure. Each number below appears "
        "in no bench/results/*.json. Run the bench that produces it and commit the JSON, mark "
        "the line [cited] if it comes from published work, or add "
        "`<!-- numbers-ok: reason -->` above it.\n\n" + "\n".join(problems)
    )


def test_results_md_is_generated_and_never_hand_edited() -> None:
    """RESULTS.md carries a generation banner; a hand-written one has no provenance."""
    path = REPO / "RESULTS.md"
    if not path.exists():
        pytest.skip("RESULTS.md is generated by bench/report.py and does not exist yet")
    head = path.read_text(encoding="utf-8")[:400].lower()
    assert "generated by" in head and "bench/report.py" in head, (
        "RESULTS.md must start with the banner bench/report.py writes; it is generated, "
        "never hand-edited"
    )


# --------------------------------------------------------------------------------------
# Planted-bug proofs (spec task 6.6: "CI fails on a planted untraceable number").
# --------------------------------------------------------------------------------------

_MEASURED = {"48.5", "0.7", "1842", "1,842", "12"}


@pytest.mark.parametrize(
    "planted",
    [
        "Speculation cut wall clock by 37%.",
        "We observed 2.4x tool throughput.",
        "Mean speculable run length past a write was 5.6 calls.",
    ],
)
def test_the_detector_fires_on_an_untraceable_number(planted: str) -> None:
    assert untraceable(planted, _MEASURED), f"detector missed a planted number: {planted!r}"


@pytest.mark.parametrize(
    "legitimate",
    [
        "PASTE reports a 48.5% reduction in average task completion time [cited].",
        "Rolling alpha over the window was 0.7 on the ops workload.",
        "Wasted tokens: 1,842.",
        "Requires Python 3.11 or newer; the CI matrix covers 3.11, 3.12 and 3.13.",
        "See arXiv 2603.18897 and RFC 6902.",
        "Released as v0.1.0 on 2026-09-15.",
        "Hard Rule 13 is checked by tests/test_context_equivalence.py.",
        "<!-- numbers-ok: illustrative shape, not a measurement -->\nwall clock 7.6s",
        "Set `max_wasted_tokens: 20000` in specunode.yaml.",
    ],
)
def test_the_detector_does_not_fire_on_non_claims(legitimate: str) -> None:
    assert not untraceable(legitimate, _MEASURED), f"false positive on: {legitimate!r}"
