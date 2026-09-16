"""Fail the build if a number in a claim file traces to no committed measurement.

Hard Rule 12's enforcement point on the command line. The logic lives in
``tests/test_numbers_traceable.py`` so that CI and the test suite cannot drift apart; this is
the front end a nightly job calls.

``python bench/check_numbers.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_numbers_traceable import CLAIM_FILES, measured_numbers, untraceable  # noqa: E402


def main() -> int:
    measured = measured_numbers()
    problems: list[str] = []
    for name in CLAIM_FILES:
        path = REPO / name
        if not path.is_file():
            continue
        for line_number, number, line in untraceable(path.read_text(encoding="utf-8"), measured):
            problems.append(
                f"{name}:{line_number}: {number!r} appears in no results file\n    {line}"
            )

    if problems:
        print("Hard Rule 12: never report a number you did not measure.\n", file=sys.stderr)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"every number in {', '.join(CLAIM_FILES)} traces to bench/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
