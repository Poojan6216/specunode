"""The acceptance rate is a measurement, not a bound (the half of spec task 6.2 that was missing).

Every document in this repository said the acceptance rate was not measured anywhere and that
signature accuracy was an upper bound on it. These tests are about the measurement that
replaced that sentence: the leave-one-out is exact, the grader is the runtime's own gate, a
right signature with a wrong value is a miss, a guess at a turn boundary is a miss under the
policy the runtime actually runs, the drafter sees only what the runtime hands it, and the
committed numbers regenerate identically from the committed sidecar.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.corpus.fetch import VALUE_INLINE_BYTES, reduce_value  # noqa: E402
from bench.offline.run_acceptance import (  # noqa: E402
    Call,
    grade,
    load_joined,
    measure,
    without,
)

from specunode.canonical import canonical  # noqa: E402
from specunode.core.decision import ToolCall  # noqa: E402
from specunode.drafters.t1_pattern import PatternIndex  # noqa: E402

VALUES = REPO / "bench" / "corpus" / "values.json"
COMMITTED = REPO / "bench" / "results" / "acceptance.json"


def _traces() -> list[list[ToolCall]]:
    """Six small trajectories with enough variety that leaving one out changes the counts."""
    return [
        [ToolCall("f", {"x": 1}), ToolCall("g", {"x": 1}), ToolCall("h", {"y": 2})],
        [ToolCall("f", {"x": 2}), ToolCall("g", {"x": 2})],
        [ToolCall("f", {"x": 3}), ToolCall("h", {"y": 3}), ToolCall("g", {"x": 3})],
        [ToolCall("g", {"x": 4}), ToolCall("g", {"x": 4})],
        [ToolCall("f", {"x": 5})],
        [
            ToolCall("h", {"y": 6}),
            ToolCall("f", {"x": 6}),
            ToolCall("g", {"x": 6}),
            ToolCall("h", {"y": 6}),
        ],
    ]


def _trained(traces: list[list[ToolCall]]) -> PatternIndex:
    index = PatternIndex(order=2)
    index.train(traces)
    return index


def test_leave_one_out_by_subtraction_equals_retraining() -> None:
    """Exact, not approximate: the same counts, the same ranking, for every held-out trace."""
    traces = _traces()
    full = _trained(traces)
    for held_out, trace in enumerate(traces):
        retrained = _trained([t for i, t in enumerate(traces) if i != held_out])
        subtracted = without(full, trace)
        assert subtracted.transitions == retrained.transitions, f"trajectory {held_out}"
        assert subtracted.trained_on == retrained.trained_on
        for history in ([], trace[:1], trace[:2], [ToolCall("h", {"y": 0})]):
            assert subtracted.rank(history) == retrained.rank(history)
    # The full index is untouched by the subtraction.
    assert full.transitions == _trained(traces).transitions


def test_the_right_signature_with_the_wrong_value_is_a_miss() -> None:
    """The distinction every document draws between signature accuracy and acceptance."""
    index = _trained([[ToolCall("f", {"x": 1}), ToolCall("g", {"x": 1})]] * 3)
    known = frozenset({"f", "g"})
    wrong = [
        Call(tool="f", args={"x": 2}, turn=0, ordinal=0, refs_prior_output=False),
        Call(tool="g", args={"x": 3}, turn=0, ordinal=1, refs_prior_output=False),
    ]
    right = [
        Call(tool="f", args={"x": 2}, turn=0, ordinal=0, refs_prior_output=False),
        Call(tool="g", args={"x": 2}, turn=0, ordinal=1, refs_prior_output=False),
    ]
    [miss] = asyncio.run(grade(wrong, index, across_turns=True, known=known))
    [hit] = asyncio.run(grade(right, index, across_turns=True, known=known))
    assert miss.offered and miss.signature_hit and not miss.accepted
    assert hit.offered and hit.signature_hit and hit.accepted


def test_a_guess_at_a_turn_boundary_is_a_miss_under_the_runtimes_policy() -> None:
    """The runtime squashes an open guess when the turn ends, whatever it predicted."""
    index = _trained([[ToolCall("f", {"x": 1}), ToolCall("g", {"x": 1})]] * 3)
    known = frozenset({"f", "g"})
    # The same two calls as a confirmed guess would need, but ``g`` opens a new model turn.
    calls = [
        Call(tool="f", args={"x": 2}, turn=0, ordinal=0, refs_prior_output=False),
        Call(tool="g", args={"x": 2}, turn=1, ordinal=0, refs_prior_output=False),
    ]
    [within] = asyncio.run(grade(calls, index, across_turns=False, known=known))
    [across] = asyncio.run(grade(calls, index, across_turns=True, known=known))
    assert within.offered and not within.accepted and within.squashed_at_turn_end
    assert across.offered and across.accepted and not across.squashed_at_turn_end


def test_within_a_turn_the_drafter_sees_only_that_turns_calls() -> None:
    """The history the runtime hands the drafter is the turn's, not the run's.

    An order-2 index whose two-call context says one thing and whose one-call context says
    another tells the two apart: with the whole run in view the guess is right, with only the
    current turn in view it is wrong.
    """
    index = _trained(
        [[ToolCall("f", {"x": 1}), ToolCall("g", {"x": 1}), ToolCall("h", {"x": 1})]] * 3
        + [[ToolCall("g", {"x": 1}), ToolCall("k", {"x": 1})]] * 5
    )
    known = frozenset({"f", "g", "h", "k"})
    calls = [
        Call(tool="f", args={"x": 1}, turn=0, ordinal=0, refs_prior_output=False),
        Call(tool="g", args={"x": 1}, turn=1, ordinal=0, refs_prior_output=False),
        Call(tool="h", args={"x": 1}, turn=1, ordinal=1, refs_prior_output=False),
    ]
    within = asyncio.run(grade(calls, index, across_turns=False, known=known))
    across = asyncio.run(grade(calls, index, across_turns=True, known=known))
    assert [g.accepted for g in across] == [True, True]
    # After ``g`` the turn-local history is ``[g]`` alone, and ``g`` alone predicts ``k``.
    assert [g.accepted for g in within] == [False, False]
    assert [g.squashed_at_turn_end for g in within] == [True, False]
    assert within[1].offered and not within[1].signature_hit


def test_the_sidecar_join_drops_what_does_not_line_up(tmp_path: Path) -> None:
    """A trajectory whose values do not match its shape is dropped and counted, never guessed."""

    def shape(tool: str, keys: list[str], turn: int, ordinal: int, refs: bool) -> dict[str, object]:
        return {
            "tool": tool,
            "arg_keys": keys,
            "turn": turn,
            "ordinal": ordinal,
            "refs_prior_output": refs,
        }

    corpus = {
        "traces": [
            {
                "trajectory_id": "t1",
                "repo": "r",
                "steps": [shape("f", ["x"], 0, 0, False), shape("g", ["x"], 1, 0, True)],
            },
            {"trajectory_id": "t2", "repo": "r", "steps": [shape("f", ["x"], 0, 0, False)]},
            {"trajectory_id": "t3", "repo": "r", "steps": [shape("f", ["x"], 0, 0, False)]},
        ]
    }
    values = {
        "trajectories": [
            {"trajectory_id": "t1", "steps": [{"x": 1}, {"x": 2}]},
            {"trajectory_id": "t3", "steps": [{"y": 1}]},  # keys disagree with the corpus
            {"trajectory_id": "t3", "steps": [{"x": 1}]},  # a duplicate id is ignored
        ]
    }
    (tmp_path / "traces.json").write_text(json.dumps(corpus))
    (tmp_path / "values.json").write_text(json.dumps(values))

    joined, report = load_joined(tmp_path / "traces.json", tmp_path / "values.json")
    assert [[c.tool for c in calls] for calls in joined] == [["f", "g"]]
    assert joined[0][1].refs_prior_output and joined[0][1].turn == 1
    assert joined[0][1].args == {"x": 2}
    assert report == {
        "trajectories_in_corpus": 3,
        "joined": 1,
        "dropped_missing_values": 1,
        "dropped_misaligned": 1,
        "duplicate_ids_in_sidecar": 1,
    }


def test_a_digested_value_equals_itself_and_only_itself() -> None:
    """The sidecar keeps equality, which is all the gate ever asks of a value, and nothing else."""
    long_a, long_b = "x" * 200, "x" * 199 + "y"
    assert reduce_value("short") == "short"
    assert reduce_value(7) == 7
    assert reduce_value([1, 2]) == [1, 2]
    assert reduce_value(long_a) == reduce_value(long_a)
    assert reduce_value(long_a) != reduce_value(long_b)
    assert reduce_value(long_a) != long_a
    assert len(canonical(reduce_value(long_a))) < VALUE_INLINE_BYTES + 32


def test_acceptance_never_exceeds_signature_accuracy() -> None:
    """A confirmed guess has the right signature by construction, so the two numbers nest."""
    trajectories = [
        [
            Call(tool=c.name, args=c.args, turn=i, ordinal=0, refs_prior_output=False)
            for i, c in enumerate(trace)
        ]
        for trace in _traces()
    ]
    report = asyncio.run(measure(trajectories, order=2))
    for mode in ("within_turn", "across_turns"):
        assert report[mode]["accepted"] <= report[mode]["steps"]
        assert report[mode]["acceptance_rate"] <= report[mode]["signature_top1"]
    assert report["across_turns"]["accepted"] > 0, "the fixture should yield some hits"
    # Every call opens a turn here, so the runtime's own policy can never resolve a guess.
    assert report["within_turn"]["accepted"] == 0
    assert report["within_turn"]["squashed_at_turn_end"] == report["within_turn"]["offered"]


@pytest.mark.skipif(
    not VALUES.is_file(), reason="no values sidecar; run bench/corpus/fetch.py --values"
)
def test_the_committed_acceptance_numbers_regenerate_identically(tmp_path: Path) -> None:
    """Hard Rule 12: the numbers in the repo are the ones the command produces today."""
    assert COMMITTED.is_file(), "the sidecar is committed but the measurement is not"
    out = tmp_path / "acceptance.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "offline" / "run_acceptance.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(out.read_text()) == json.loads(COMMITTED.read_text())


def test_the_copying_ceiling_is_every_value_seen_before_not_the_whole_call() -> None:
    """A tier-1 guess can be right for a call nobody made yet, if its values were all seen."""
    from bench.offline.run_acceptance import copying_ceiling

    trace = [
        ToolCall("f", {"x": 1}),
        ToolCall("g", {"x": 1}),  # a new call, assembled from a value already seen
        ToolCall("f", {"x": 1}),  # repeats an earlier call, not the previous one
        ToolCall("f", {"x": 1}),  # repeats the previous call
        ToolCall("f", {"x": 2}),  # same signature, a value never seen: not assemblable
    ]
    ceiling = copying_ceiling([trace])
    assert ceiling["steps"] == 4
    assert ceiling["every_argument_value_seen_before"] == 3
    assert ceiling["whole_call_seen_before"] == 2
    assert ceiling["whole_call_is_the_previous_call"] == 1


def test_acceptance_never_exceeds_the_copying_ceiling() -> None:
    """The measurement asserts the nesting itself; this is the fixture that exercises it."""
    trajectories = [
        [
            Call(tool=c.name, args=c.args, turn=0, ordinal=i, refs_prior_output=False)
            for i, c in enumerate(trace)
        ]
        for trace in _traces()
    ]
    report = asyncio.run(measure(trajectories, order=2))
    ceiling = report["copying_ceiling"]["every_argument_value_seen_before"]
    assert 0 < report["across_turns"]["accepted"] <= ceiling
    assert report["within_turn"]["accepted"] <= ceiling
