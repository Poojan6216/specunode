"""The acceptance rate, measured (the half of spec task 6.2 that was missing).

``run_opportunity.py`` reports how often an order-2 index ranks the right *signature* -- the
tool name and its sorted argument keys -- and says on every page that this is an upper bound on
the acceptance rate, because the gate that releases a write compares argument *values* and the
signature measure never sees one. RESULTS.md and the Final Report said the acceptance rate
itself was not measured anywhere in this repository. This measures it.

The predictor is the real one, ``PatternDrafter`` over ``PatternIndex``. The grader is the real
one, ``resolve_decision``, which is what the scheduler calls when the model's block arrives. The
corpus is the same 300 trajectories, joined to the argument values the committed corpus drops
(``bench/corpus/values.json``). Leave-one-trajectory-out over every trajectory, by subtracting
the held-out trajectory's transitions from an index trained on all of them -- exact, and linear
rather than quadratic, so the whole corpus is graded rather than a sample of it.

Two policies are graded, because they answer different questions:

**within_turn** is what the runtime does today. A ``SpeculativeTurn`` asks the drafter after
every block with the history of *this turn*, and squashes any open guess when the turn ends. On
a corpus where nearly every call opens a new turn, almost every guess is squashed at the turn
boundary whatever it predicted, and the number says so.

**across_turns** is what carrying a guess into the next turn would be worth: the history is the
whole trajectory so far and the target is the next call wherever it lands. The runtime does not
do this. The number is published so the decision to build it, or not, rests on a measurement.

The drafter is given no tool results. The corpus keeps none, and at run time a result is only
in hand if its read completed before the next block parsed. Steps whose arguments reference a
prior result are tallied separately so the reader can see how much that half could matter.

``python bench/offline/run_acceptance.py --out bench/results/acceptance.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.corpus.effect_classes import effect_of
from bench.offline.run_opportunity import bootstrap_ci

from specunode.canonical import JsonValue, canonical
from specunode.core.branch import BranchStatus
from specunode.core.decision import ToolCall
from specunode.drafters.base import DraftContext
from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
from specunode.verify.gate import resolve_decision

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "traces.json"
VALUES = CORPUS.parent / "values.json"
MODES = ("within_turn", "across_turns")
#: The policy the scheduler implements. ``SpeculativeTurn`` owns the history and squashes at
#: ``TurnComplete``; nothing carries a guess into the next turn.
RUNTIME_POLICY = "within_turn"


@dataclass(frozen=True)
class Call:
    """One corpus step with its argument values joined back on."""

    tool: str
    args: Mapping[str, JsonValue]
    turn: int
    ordinal: int
    refs_prior_output: bool

    @property
    def signature(self) -> str:
        return f"{self.tool}({','.join(sorted(self.args))})"

    def decision(self) -> ToolCall:
        return ToolCall(name=self.tool, args=dict(self.args))


@dataclass(frozen=True)
class Grade:
    """What happened to the guess made after one call, about the next."""

    tool: str
    effect: str
    refs_prior_output: bool
    #: The drafter offered a prediction at all. It withholds one it cannot fill.
    offered: bool
    #: The gate would have confirmed it: exact canonical equality with the next call.
    accepted: bool
    #: The index's top-ranked signature was the next call's, from the history this policy hands
    #: it. Measures the index alone: a guess the turn boundary squashes can still be a hit here.
    signature_hit: bool
    #: Offered, and the next call opened a new turn, so the runtime squashed it unresolved.
    squashed_at_turn_end: bool


# -- corpus -----------------------------------------------------------------------------------


def load_joined(corpus: Path, values: Path) -> tuple[list[list[Call]], dict[str, Any]]:
    """The committed corpus with its values sidecar joined on, per trajectory and per step.

    Joined by trajectory id and checked step for step: the same number of calls, and each
    call's sorted argument keys equal to the corpus's ``arg_keys``. A trajectory that fails
    either check is dropped and counted, never silently included.
    """
    traces = json.loads(corpus.read_text(encoding="utf-8"))["traces"]
    sidecar = json.loads(values.read_text(encoding="utf-8"))["trajectories"]
    by_id: dict[str, list[dict[str, JsonValue]]] = {}
    duplicates = 0
    for entry in sidecar:
        # One spelling of the key for both the membership test and the store. Testing the raw
        # value and storing the string let a non-string duplicate id go uncounted and silently
        # replace the entry before it, which the docstring above promises never happens.
        key = str(entry["trajectory_id"])
        if key in by_id:
            duplicates += 1
            continue
        by_id[key] = list(entry["steps"])

    joined: list[list[Call]] = []
    missing = misaligned = 0
    for trace in traces:
        steps = by_id.get(str(trace["trajectory_id"]))
        if steps is None:
            missing += 1
            continue
        shapes = trace["steps"]
        if len(steps) != len(shapes) or any(
            sorted(args) != list(shape.get("arg_keys", []))
            for args, shape in zip(steps, shapes, strict=True)
        ):
            misaligned += 1
            continue
        joined.append(
            [
                Call(
                    tool=str(shape["tool"]),
                    args=args,
                    turn=int(shape.get("turn", 0)),
                    ordinal=int(shape.get("ordinal", 0)),
                    refs_prior_output=bool(shape.get("refs_prior_output")),
                )
                for args, shape in zip(steps, shapes, strict=True)
            ]
        )
    report = {
        "trajectories_in_corpus": len(traces),
        "joined": len(joined),
        "dropped_missing_values": missing,
        "dropped_misaligned": misaligned,
        "duplicate_ids_in_sidecar": duplicates,
    }
    return joined, report


# -- leave-one-out ----------------------------------------------------------------------------


def without(index: PatternIndex, trace: Sequence[ToolCall]) -> PatternIndex:
    """The index minus one trajectory: leave-one-out by subtraction, exact and linear.

    ``PatternIndex.train`` is additive -- a trajectory adds its transition counts and nothing
    else -- so an index trained on one trajectory is exactly what that trajectory contributed,
    and subtracting it from the full index is the index trained on all the others. Contexts
    and candidates that reach zero are removed, so ``rank`` never sees a candidate with no
    support. A test asserts this equals retraining.
    """
    held = PatternIndex(order=index.order)
    held.train([trace])
    out = PatternIndex(
        order=index.order,
        transitions={key: dict(counts) for key, counts in index.transitions.items()},
        arg_keys=dict(index.arg_keys),
        tool_of=dict(index.tool_of),
        trained_on=index.trained_on - 1,
    )
    for key, counts in held.transitions.items():
        remaining = out.transitions[key]
        for signature, count in counts.items():
            remaining[signature] -= count
            if remaining[signature] <= 0:
                del remaining[signature]
        if not remaining:
            del out.transitions[key]
    return out


# -- grading ----------------------------------------------------------------------------------


async def grade(
    calls: Sequence[Call], index: PatternIndex, *, across_turns: bool, known: frozenset[str]
) -> list[Grade]:
    """Grade every position at which the runtime would ask the drafter and a next call exists.

    The guess made after the final call has nothing to be graded against in either policy and
    is not counted; at run time it is squashed when the turn, and the run, ends.
    """
    drafter = PatternDrafter(index=index, top_k=1)
    decisions = [call.decision() for call in calls]
    grades: list[Grade] = []
    turn_start = 0
    for position in range(len(calls) - 1):
        if calls[position].ordinal == 0:
            turn_start = position
        target = calls[position + 1]
        if across_turns:
            history = decisions[: position + 1]
            resolvable = True
        else:
            history = decisions[turn_start : position + 1]
            resolvable = target.ordinal != 0
        context = DraftContext(
            run_id="offline",
            branch_id="offline",
            step_index=position + 1,
            node_id="offline",
            history=tuple(history),
            results={},
            known_tools=known,
        )
        candidates = await drafter.predict(context)
        offered = bool(candidates)
        ranked = index.rank(history)
        # Deliberately not gated on ``resolvable``. This measures the *index* -- did it rank the
        # right signature from the history the policy gives it -- and the gate's verdict is
        # ``accepted`` below. Gated, the within-turn column read 0.0000 by construction on a
        # corpus where every call opens a turn, which looked like a measured collapse in the
        # predictor and was a restatement of ``squashed_at_turn_end``.
        signature_hit = bool(ranked) and ranked[0][0] == target.signature
        accepted = (
            offered
            and resolvable
            and resolve_decision(candidates[0].decision, decisions[position + 1])
            is BranchStatus.CONFIRMED
        )
        grades.append(
            Grade(
                tool=target.tool,
                effect=effect_of(target.tool),
                refs_prior_output=target.refs_prior_output,
                offered=offered,
                accepted=accepted,
                signature_hit=signature_hit,
                squashed_at_turn_end=offered and not resolvable,
            )
        )
    return grades


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _bucket(grades: Sequence[Grade], of_total: int = 0) -> dict[str, Any]:
    steps = len(grades)
    offered = sum(1 for g in grades if g.offered)
    accepted = sum(1 for g in grades if g.accepted)
    bucket = {
        "steps": steps,
        "offered": offered,
        "accepted": accepted,
        "acceptance_rate": _rate(accepted, steps),
        "acceptance_given_offered": _rate(accepted, offered),
    }
    if of_total:
        # The bucket's share of all graded steps. Carried here rather than divided in prose,
        # because a number a document computes for itself traces to no measurement (Rule 12).
        bucket["share_of_graded_steps"] = _rate(steps, of_total)
    return bucket


def tally(per_trajectory: Sequence[Sequence[Grade]]) -> dict[str, Any]:
    flat = [g for grades in per_trajectory for g in grades]
    steps = len(flat)
    offered = sum(1 for g in flat if g.offered)
    accepted = sum(1 for g in flat if g.accepted)
    signature = sum(1 for g in flat if g.signature_hit)
    # A confirmed guess has the right signature by construction: the gate demands exact
    # equality, keys included, and the drafter offers only the top-ranked signature.
    assert accepted <= signature, "accepted guesses exceed signature hits"
    mean, low, high = bootstrap_ci(
        [sum(1 for g in grades if g.accepted) / len(grades) for grades in per_trajectory if grades]
    )
    return {
        "steps": steps,
        "offered": offered,
        "offered_rate": _rate(offered, steps),
        "accepted": accepted,
        "acceptance_rate": _rate(accepted, steps),
        "acceptance_rate_by_trajectory": {
            "mean": round(mean, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
        },
        "acceptance_given_offered": _rate(accepted, offered),
        "signature_top1": _rate(signature, steps),
        "squashed_at_turn_end": sum(1 for g in flat if g.squashed_at_turn_end),
        "by_effect": {
            label: _bucket([g for g in flat if g.effect == label]) for label in ("read", "write")
        },
        "by_argument_provenance": {
            "references_a_prior_result": _bucket(
                [g for g in flat if g.refs_prior_output], of_total=steps
            ),
            "does_not": _bucket([g for g in flat if not g.refs_prior_output], of_total=steps),
        },
        "by_tool": {
            tool: _bucket([g for g in flat if g.tool == tool])
            for tool in sorted({g.tool for g in flat})
        },
        # The realisable half of the opportunity analysis: steps at which a predicted WRITE
        # would have been staged and then retired. PASTE's figure for this is zero by policy.
        "realisable_write_speculation": _rate(
            sum(1 for g in flat if g.accepted and g.effect == "write"), steps
        ),
    }


def copying_ceiling(trajectories: Sequence[Sequence[ToolCall]]) -> dict[str, Any]:
    """How often the next call could be assembled from earlier *calls*: this grading's ceiling.

    ``PatternDrafter`` fills a guess by copying values out of earlier calls **and out of tool
    results**. This corpus keeps no results -- they are free text, and ``fetch.py`` stores only
    argument keys and a flag for whether a value appeared in a prior result -- so the drafter is
    graded here with ``results={}``, and under that grading it can be exactly right only where
    every argument value of the next call has already occurred as an argument value of an
    earlier one. That is what this rate bounds: tier 1 *as graded here*, not tier 1 with results
    in hand, and not every conceivable copying predictor. The repository's own provenance
    measure says an argument of 20.5% of graded steps did appear in a prior result, so the bound
    with results available is unknown and higher. Whole-call repeats are reported alongside as
    the stricter, more intuitive figure. The denominator is the graded positions: calls with a
    predecessor.
    """
    steps = assemblable = whole = previous = 0
    for trace in trajectories:
        seen_values: set[bytes] = set()
        seen_calls: set[bytes] = set()
        last: bytes | None = None
        for position, call in enumerate(trace):
            key = canonical({"name": call.name, "args": dict(call.args)})
            values = {canonical(value) for value in call.args.values()}
            if position > 0:
                steps += 1
                assemblable += values <= seen_values
                whole += key in seen_calls
                previous += key == last
            seen_values |= values
            seen_calls.add(key)
            last = key
    return {
        "steps": steps,
        "every_argument_value_seen_before": assemblable,
        "rate": _rate(assemblable, steps),
        "whole_call_seen_before": whole,
        "rate_whole_call": _rate(whole, steps),
        "whole_call_is_the_previous_call": previous,
        "rate_previous_call": _rate(previous, steps),
        "relation": (
            "a predictor that copies argument values out of earlier CALLS only -- which is "
            "tier 1 as graded here, with no tool results -- cannot be exactly right more often "
            "than rate; with results in hand the bound is higher and is not measured"
        ),
    }


async def measure(trajectories: Sequence[Sequence[Call]], *, order: int = 2) -> dict[str, Any]:
    full = PatternIndex(order=order)
    decisions = [[call.decision() for call in calls] for calls in trajectories]
    full.train(decisions)
    known = frozenset(call.tool for calls in trajectories for call in calls)

    graded: dict[str, list[list[Grade]]] = {mode: [] for mode in MODES}
    for calls, trace in zip(trajectories, decisions, strict=True):
        index = without(full, trace)
        for mode in MODES:
            graded[mode].append(
                await grade(calls, index, across_turns=(mode == "across_turns"), known=known)
            )
    report: dict[str, Any] = {
        "trajectories": len(trajectories),
        "predictor": {
            "index": f"order-{order} PatternIndex, leave-one-trajectory-out over every trajectory",
            "drafter": "PatternDrafter top_k=1, require_complete_args=True, no tool results",
            "grader": "specunode.verify.gate.resolve_decision",
        },
        "runtime_policy": RUNTIME_POLICY,
        "copying_ceiling": copying_ceiling(decisions),
    }
    for mode in MODES:
        report[mode] = tally(graded[mode])
        ceiling = report["copying_ceiling"]["every_argument_value_seen_before"]
        assert report[mode]["accepted"] <= ceiling, "accepted guesses exceed the copying ceiling"
    return report


# -- cli --------------------------------------------------------------------------------------


def _print(report: dict[str, Any]) -> None:
    print()
    print("ACCEPTANCE RATE (the gate's own verdict, not the signature bound)")
    print(f"  {report['trajectories']} trajectories; runtime policy: {report['runtime_policy']}")
    print()
    print(f"  {'':34} {'within_turn':>12} {'across_turns':>13}")
    rows = (
        ("steps graded", "steps", "d"),
        ("guesses offered", "offered_rate", "f"),
        ("acceptance rate (pooled)", "acceptance_rate", "f"),
        ("acceptance given offered", "acceptance_given_offered", "f"),
        ("signature top-1, same steps", "signature_top1", "f"),
        ("squashed at a turn boundary", "squashed_at_turn_end", "d"),
        ("realisable write speculation", "realisable_write_speculation", "f"),
    )
    for label, key, kind in rows:
        cells = [report[mode][key] for mode in MODES]
        text = [f"{c:>12d}" if kind == "d" else f"{c:>12.4f}" for c in cells]
        print(f"  {label:<34} {text[0]} {text[1]:>13}")
    for mode in MODES:
        ci = report[mode]["acceptance_rate_by_trajectory"]
        print(
            f"  {mode + ', by trajectory':<34} {ci['mean']:.4f} "
            f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"
        )
    ceiling = report["copying_ceiling"]
    print()
    print(f"  {'every argument value seen before':<34} {ceiling['rate']:>12.4f}   (tier-1 ceiling)")
    print(f"  {'whole call seen before':<34} {ceiling['rate_whole_call']:>12.4f}")
    print(f"  {'whole call is the previous call':<34} {ceiling['rate_previous_call']:>12.4f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the tier-1 acceptance rate")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--values", type=Path, default=VALUES)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="trajectories, for a quick run")
    args = parser.parse_args(argv)

    if not args.corpus.is_file():
        print(f"no corpus at {args.corpus}; run bench/corpus/fetch.py first", file=sys.stderr)
        return 1
    if not args.values.is_file():
        print(
            f"no values sidecar at {args.values}; run bench/corpus/fetch.py --values first",
            file=sys.stderr,
        )
        return 1

    trajectories, alignment = load_joined(args.corpus, args.values)
    if args.limit is not None:
        trajectories = trajectories[: args.limit]
    if not trajectories:
        print("no trajectory in the corpus lines up with the values sidecar", file=sys.stderr)
        return 1

    manifest = json.loads((args.corpus.parent / "manifest.json").read_text(encoding="utf-8"))
    values_manifest_path = args.values.parent / "values_manifest.json"
    values_manifest = (
        json.loads(values_manifest_path.read_text(encoding="utf-8"))
        if values_manifest_path.is_file()
        else None
    )
    report = {
        "corpus": manifest,
        "values": values_manifest,
        "alignment": alignment,
        "acceptance": asyncio.run(measure(trajectories, order=args.order)),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    _print(report["acceptance"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
