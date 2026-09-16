"""Hard Rule 12, second half: no absolute claim without the condition that makes it true.

A runtime that says "guaranteed" is making a promise its user will rely on. Every term in
:data:`FORBIDDEN` has been used, somewhere in this field, to describe a property the system
did not have. They are not banned outright -- they are banned *unqualified*.

A forbidden term is accepted in exactly three situations:

1. It is **mentioned, not used** -- wrapped in backticks or double quotes. Writing
   ``never say "exactly-once" for a tool that has not declared itself idempotent`` is the
   honest sentence, not the dishonest one.
2. Its sentence contains a **qualifier from a closed list** that names the condition.
   The list is deliberately short; "no", "not" and "never" alone are *not* on it, because
   "guaranteed no leaks" would otherwise pass.
3. The line is preceded by an explicit ``<!-- vocab-ok: reason -->`` marker, which forces
   whoever wants the exemption to write down why.

``BUILD_SPEC.md`` is excluded: it is the input specification rather than something this
project claims, and it necessarily quotes the whole forbidden vocabulary in order to forbid it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Terms that state an absolute, and the reason each is dangerous here.
FORBIDDEN = {
    "acid": "the store buffer is not a transaction manager and relaxes durability boundaries",
    "exactly-once": "dispatch is at-least-once with idempotent dedupe; see Hard Rule 8",
    "exactly once": "dispatch is at-least-once with idempotent dedupe; see Hard Rule 8",
    "guaranteed": "names a promise without naming the tool classes it holds for",
    "guarantees": "names a promise without naming the tool classes it holds for",
    "zero-latency": "speculation hides tool latency, not model latency",
    "zero latency": "speculation hides tool latency, not model latency",
    "eliminates": "hazards stall rather than disappear",
    "100% safe": "a misdeclared tool defeats the store buffer entirely (attack 7.1)",
    "context rot": "vocabulary from a claim this project does not measure",
}

#: Condition-naming connectives. Short on purpose: a term is only excused by a phrase that
#: actually introduces the condition under which the absolute holds.
QUALIFIERS = (
    "unless",
    "only when",
    "only if",
    "only for",
    "only on",
    "only after",
    "except when",
    "except for",
    "provided that",
    "subject to",
    "conditional on",
    "assuming",
    "relaxes",
    "does not",
    "do not",
    "is not",
    "are not",
    "never say",
    "never claim",
    "must not",
    "cannot",
    "rather than",
    "instead of",
)

EXCLUDED_FILES = {"BUILD_SPEC.md"}
EXCLUDED_DIRS = {".git", ".venv", "node_modules", "dist", "build", ".mypy_cache", ".ruff_cache"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n")
# A mention is a term wrapped in backticks or quotes. Single quotes only count when they
# are not apostrophes -- "the checker's own proof" must not open a quoted span that swallows
# everything up to the next apostrophe and exposes whatever follows.
_MENTION = re.compile(r"`[^`]*`|\"[^\"]*\"|(?<![A-Za-z])'[^']*'(?![A-Za-z])")
_VOCAB_OK = re.compile(r"vocab-ok:\s*\S")


def _scanned_files() -> list[Path]:
    """Every documentation and source file this project publishes."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
        candidates = [REPO / name for name in tracked]
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        candidates = [p for p in REPO.rglob("*") if not EXCLUDED_DIRS & set(p.parts)]
    return [
        path
        for path in candidates
        if path.suffix in {".md", ".py", ".yaml", ".yml", ".txt"}
        and path.name not in EXCLUDED_FILES
        and path.is_file()
        and not EXCLUDED_DIRS & set(path.relative_to(REPO).parts)
    ]


def violations_in(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_number, term, sentence)`` for every unqualified forbidden term."""
    found: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        previous = lines[index - 2] if index >= 2 else ""
        if _VOCAB_OK.search(previous) or _VOCAB_OK.search(line):
            continue
        # Strip mentions (backticked or quoted spans): a term inside them is being named,
        # not asserted.
        used = _MENTION.sub(" ", line)
        for sentence in _SENTENCE_SPLIT.split(used):
            lowered = sentence.lower()
            for term in FORBIDDEN:
                if term in lowered and not any(q in lowered for q in QUALIFIERS):
                    found.append((index, term, sentence.strip()))
    return found


def test_no_unqualified_absolute_claims_anywhere_in_the_repo() -> None:
    problems: list[str] = []
    for path in _scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - binary file
            continue
        for line_number, term, sentence in violations_in(text):
            rel = path.relative_to(REPO)
            problems.append(
                f"{rel}:{line_number}: {term!r} without a qualifier naming the condition "
                f"({FORBIDDEN[term]})\n    {sentence}"
            )
    assert not problems, (
        "Hard Rule 12: an absolute claim appears without the condition that makes it true.\n"
        "Qualify it, quote it as a term, or add `<!-- vocab-ok: reason -->` on the line "
        "above with a reason.\n\n" + "\n".join(problems)
    )


# --------------------------------------------------------------------------------------
# The checker's own planted-bug proof. Spec task 0.5: "planting the word 'guaranteed' in
# README.md fails the vocabulary test". These assert the detector actually fires, so the
# test above cannot rot into one that passes because it checks nothing.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "planted",
    [
        "SpecuNode guaranteed no leaks.",
        "Delivery is exactly-once.",
        "The store buffer eliminates duplicate charges.",
        "Speculation is 100% safe.",
        "Full ACID semantics across tools.",
        "This removes context rot from long runs.",
        "Tool dispatch is zero-latency.",
    ],
)
def test_the_detector_fires_on_planted_claims(planted: str) -> None:
    assert violations_in(planted), f"detector missed a planted claim: {planted!r}"


@pytest.mark.parametrize(
    "honest",
    [
        'The docs say at-least-once with idempotent dedupe and never say "exactly-once".',
        "Nothing here is guaranteed unless the tool declared its effect class correctly.",
        "It does not eliminate hazards; a hazard stalls the branch.",
        "SagaLLM relaxes ACID and ensures workflow-wide recoverability.",
        "Speculation hides tool latency rather than model latency.",
        "<!-- vocab-ok: quoting a reviewer -->\nThey called it guaranteed.",
        "The term `exactly-once` describes something this dispatcher does not provide.",
    ],
)
def test_the_detector_does_not_fire_on_qualified_prose(honest: str) -> None:
    assert not violations_in(honest), f"detector false-positived on: {honest!r}"
