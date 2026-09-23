"""The tier-2 acceptance rate: does a draft model clear the ceiling tier 1 cannot? (task 6.8)

``bench/offline/run_acceptance.py`` measured the tier-1 predictor with the runtime's own gate
and found 0.0002 across turns and 0.0000 under the policy the runtime runs. It also measured
why: tier 1 fills a guess by copying argument values it has already seen, and every value of
the next call has appeared before at only 9.8% of steps. Anything above that has to come from
a predictor that *generates* values, which is what tier 2 is -- a small model asked what comes
next. That number is the one the case for this whole design now rests on, and it is the only
measurement here that needs a credential.

Same corpus, same join, same grader. The drafter is the real ``ModelDrafter`` and the verdict
is the real ``resolve_decision``: exact canonical equality of the tool name and every argument
value. The comparison to tier 1 is therefore like for like, on the same steps, with one
difference stated plainly: the draft model may invent a value it has never seen, and the
pattern index may not.

**The draft model is shown real argument values.** The committed sidecar digests any value
over 64 bytes, which is all a *gate* needs -- it compares values for equality -- but not what a
model needs to predict one. Measured against the digested sidecar the draft model said so
itself: "I cannot determine the next tool call because the hashes in the previous calls
obscure" what it needed. This runner defaults to ``values_full.json``, the undigested sidecar
that ``fetch.py --values --full`` rebuilds and that is too large to commit.

**Two policies, and only one is interesting here.** Under the policy the runtime actually runs
-- a guess is squashed when the turn ends -- acceptance on this corpus is zero for *any*
predictor, because every call in it opens a new model turn. That is a property of the corpus,
not of the drafter, and the offline measurement already reports it. So this runner grades the
generous policy: the whole trajectory so far as history, the next call wherever it lands. A
near-zero result here is decisive; a good one is an argument for carrying guesses across turns,
which the runtime does not do today.

**It is runnable without an API key.** ``--model scripted`` drives the whole harness against a
deterministic stand-in that predicts the most recent call again, so the sampling, grading,
accounting and budget gate are exercised in CI where the credential is not available.

``ANTHROPIC_API_KEY=... python bench/online/run_tier2_acceptance.py --out bench/results/tier2.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.corpus.effect_classes import effect_of
from bench.offline.run_acceptance import Call, load_joined
from bench.offline.run_opportunity import bootstrap_ci
from bench.online.run_latency import Spend, prices_for
from specunode.canonical import JsonValue
from specunode.core.branch import BranchStatus
from specunode.core.decision import ToolCall
from specunode.core.model import ModelResponse, RequestEnvelope, TextBlock, ToolDef, Usage
from specunode.drafters.base import DraftContext
from specunode.drafters.t2_model import DEFAULT_DRAFT_MODEL, ModelDrafter
from specunode.verify.gate import resolve_decision

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "traces.json"
VALUES = CORPUS.parent / "values_full.json"
#: Deterministic, so two runs of this command grade the same steps and the numbers can be
#: compared. Recorded in the output beside the sample size.
SEED = 20260918
DEFAULT_SAMPLE = 500
#: How many earlier calls the draft model is shown. The runtime hands its drafter the calls of
#: the *current turn*, not the whole run, so an unbounded history is both unlike production and
#: needlessly expensive -- a position 50 calls into a trajectory was costing 2,400 input tokens.
#: Tier 1's index looks at the last two signatures, so this window is more context than the
#: predictor it is being compared against ever gets.
DEFAULT_HISTORY = 12
DEFAULT_BUDGET_USD = 5.0
#: Haiku-class prices, per million tokens. Declared here and printed, like the latency bench's.
PRICE_PER_MTOK = {"input": 1.0, "output": 5.0}


@dataclass
class Graded:
    """One position: what the drafter said, and what the gate made of it."""

    tool: str
    effect: str
    refs_prior_output: bool
    offered: bool
    accepted: bool
    tool_only: bool
    predicted: str | None = None


class _RepeatsTheLastCall:
    """The stand-in: predicts the most recent call again. Cheap, deterministic, usually wrong.

    Not a strawman for the real thing -- it exists so the harness runs in CI. Its own number is
    reported with ``is_real_model: false``, exactly as the latency bench does.
    """

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        history = envelope.messages[0].content[0]
        text = getattr(history, "text", "")
        last = [line for line in text.splitlines() if line and line[0].isdigit()]
        answer = "UNKNOWN"
        if last:
            _, _, rest = last[-1].partition(". ")
            name, _, args = rest.partition(" ")
            answer = json.dumps({"tool": name, "args": json.loads(args or "{}")})
        return ModelResponse(
            model=envelope.model,
            content=(TextBlock(text=answer),),
            stop_reason="end_turn",
            usage=Usage(input_tokens=160, output_tokens=40),
        )

    async def stream(self, envelope: RequestEnvelope) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


def tools_from_corpus(trajectories: Sequence[Sequence[Call]]) -> tuple[ToolDef, ...]:
    """Tool definitions built from the corpus's own argument keys, never invented.

    The draft model is offered the same tools the runtime would know about. Their schemas come
    from the keys each tool is observed with, so the model is told what a call looks like
    without anyone deciding what it *should* look like.
    """
    keys: dict[str, set[str]] = {}
    for calls in trajectories:
        for call in calls:
            keys.setdefault(call.tool, set()).update(call.args)
    return tuple(
        ToolDef(
            name=tool,
            description=f"{tool} (arguments observed in this corpus)",
            input_schema={
                "type": "object",
                "properties": {key: {"type": "string"} for key in sorted(observed)},
            },
        )
        for tool, observed in sorted(keys.items())
    )


def sample_positions(
    trajectories: Sequence[Sequence[Call]], count: int, seed: int
) -> list[tuple[int, int]]:
    """``count`` (trajectory, position) pairs, drawn once and reproducibly.

    Every position with a predecessor and a successor is eligible, and the draw is over all of
    them rather than per trajectory: a long trajectory contributes more positions because it
    *has* more, which is what a per-step rate is a rate over.
    """
    eligible = [
        (index, position)
        for index, calls in enumerate(trajectories)
        for position in range(len(calls) - 1)
    ]
    rng = random.Random(seed)
    if count >= len(eligible):
        return eligible
    return sorted(rng.sample(eligible, count))


async def grade_one(
    drafter: ModelDrafter,
    calls: Sequence[Call],
    position: int,
    known: frozenset[str],
    window: int = DEFAULT_HISTORY,
) -> Graded:
    """Ask the draft model for the call after ``position``, and put it to the gate."""
    target = calls[position + 1]
    start = max(0, position + 1 - window) if window > 0 else 0
    history = tuple(call.decision() for call in calls[start : position + 1])
    candidates = await drafter.predict(
        DraftContext(
            run_id="tier2",
            branch_id="tier2",
            step_index=position + 1,
            node_id="tier2",
            history=history,
            # The corpus keeps no result text, so the draft model works from the calls alone --
            # the same handicap the tier-1 measurement reports, and the same one the runtime
            # imposes on a drafter asked before a read has returned.
            results={},
            known_tools=known,
        )
    )
    if not candidates:
        return Graded(
            tool=target.tool,
            effect=effect_of(target.tool),
            refs_prior_output=target.refs_prior_output,
            offered=False,
            accepted=False,
            tool_only=False,
        )
    guess = candidates[0].decision
    accepted = resolve_decision(guess, target.decision()) is BranchStatus.CONFIRMED
    name = guess.name if isinstance(guess, ToolCall) else None
    return Graded(
        tool=target.tool,
        effect=effect_of(target.tool),
        refs_prior_output=target.refs_prior_output,
        offered=True,
        accepted=accepted,
        tool_only=name == target.tool,
        predicted=name,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def tally(graded: Sequence[Graded], drafter: ModelDrafter) -> dict[str, Any]:
    steps = len(graded)
    offered = sum(1 for g in graded if g.offered)
    accepted = sum(1 for g in graded if g.accepted)
    tool_only = sum(1 for g in graded if g.tool_only)
    mean, low, high = bootstrap_ci([1.0 if g.accepted else 0.0 for g in graded])
    return {
        "steps_graded": steps,
        "offered": offered,
        "offered_rate": _rate(offered, steps),
        "unparsable": drafter.unparsable,
        "accepted": accepted,
        "acceptance_rate": _rate(accepted, steps),
        "acceptance_rate_ci95": {
            "mean": round(mean, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
        },
        "acceptance_given_offered": _rate(accepted, offered),
        # The right tool, whatever the arguments: the same relation the offline measurement
        # calls signature accuracy, minus the argument *names*. Reported because the gap
        # between it and the acceptance rate is the whole finding.
        "right_tool_wrong_arguments": _rate(tool_only - accepted, steps),
        "right_tool": _rate(tool_only, steps),
        "by_effect": {
            label: {
                "steps": len([g for g in graded if g.effect == label]),
                "accepted": len([g for g in graded if g.effect == label and g.accepted]),
            }
            for label in ("read", "write")
        },
    }


async def measure(
    trajectories: Sequence[Sequence[Call]],
    positions: Sequence[tuple[int, int]],
    drafter: ModelDrafter,
    spend: Spend,
    window: int = DEFAULT_HISTORY,
) -> tuple[list[Graded], bool]:
    known = frozenset(call.tool for calls in trajectories for call in calls)
    graded: list[Graded] = []
    for index, position in positions:
        if not spend.may_continue():
            return graded, True
        graded.append(await grade_one(drafter, trajectories[index], position, known, window))
    return graded, False


@dataclass
class _Metered:
    """Counts the draft model's spend the way the latency bench counts the target's."""

    inner: Any
    spend: Spend
    calls: int = field(default=0)

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.calls += 1
        response = await self.inner.complete(envelope)
        self.spend.record(response)
        return response


def build_client(kind: str, model: str) -> Any:
    if kind == "scripted":
        return _RepeatsTheLastCall()
    if kind == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Run with --model scripted to exercise the "
                "harness without a credential; what it produces is not a measurement of a "
                "draft model and the output says so."
            )
        if model == "scripted":
            raise SystemExit("--model anthropic with --draft-model scripted is not a model id.")
        from specunode.integrations.anthropic import AnthropicModel

        return AnthropicModel()
    raise SystemExit(f"unknown --model {kind!r}; use 'anthropic' or 'scripted'")


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench/results/tier2.json"))
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--values", type=Path, default=VALUES)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--model", default="anthropic", help="anthropic|scripted")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not args.corpus.is_file() or not args.values.is_file():
        print("no corpus or values sidecar; run bench/corpus/fetch.py --values", file=sys.stderr)
        return 1

    trajectories, alignment = load_joined(args.corpus, args.values)
    positions = sample_positions(trajectories, args.sample, args.seed)
    cap = float(os.environ.get("SPECUNODE_BENCH_BUDGET_USD", DEFAULT_BUDGET_USD))
    # Haiku-class prices, because that is what this benchmark bills against. The class
    # defaults to the target model's table, which is three times these.
    spend_prices = (
        prices_for(args.draft_model) if args.model == "anthropic" else dict(PRICE_PER_MTOK)
    )
    spend = Spend(cap_usd=cap, prices_per_mtok=spend_prices)

    client = _Metered(inner=build_client(args.model, args.draft_model), spend=spend)
    drafter = ModelDrafter(
        client=client,  # type: ignore[arg-type]
        model=args.draft_model if args.model == "anthropic" else "scripted",
        tools=tools_from_corpus(trajectories),
    )

    started = time.monotonic()
    graded, halted = await measure(trajectories, positions, drafter, spend, args.history)
    elapsed = time.monotonic() - started

    report: dict[str, JsonValue] = {
        "bench": "tier2_acceptance",
        "is_real_model": args.model == "anthropic",
        "draft_model": drafter.model,
        "policy": "across_turns",
        "policy_note": (
            "Under the policy the runtime runs -- an open guess is squashed when the turn "
            "ends -- acceptance on this corpus is zero for any predictor, because every call "
            "in it opens a new model turn. This grades the generous policy instead."
        ),
        "grader": "specunode.verify.gate.resolve_decision",
        "seed": args.seed,
        "sample_requested": args.sample,
        "history_window": args.history,
        "values": str(args.values.name),
        "eligible_positions": len(positions),
        "alignment": alignment,
        "halted_on_budget": halted,
        "budget_usd_cap": cap,
        "estimated_spend_usd": round(spend.usd, 4),
        "prices_per_mtok_usd": spend_prices,
        "wall_seconds": round(elapsed, 1),
        "draft_calls": client.calls,
        "result": tally(graded, drafter),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    r = report["result"]
    assert isinstance(r, dict)
    print()
    print(f"TIER-2 ACCEPTANCE ({drafter.model}, real: {report['is_real_model']})")
    print(f"  steps graded            {r['steps_graded']}")
    print(f"  guesses offered         {r['offered_rate']:.4f}  ({r['unparsable']} unparsable)")
    print(f"  ACCEPTANCE RATE         {r['acceptance_rate']:.4f}")
    ci = r["acceptance_rate_ci95"]
    print(f"    95% interval          [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]")
    print(f"  right tool              {r['right_tool']:.4f}")
    print(f"  right tool, wrong args  {r['right_tool_wrong_arguments']:.4f}")
    print(f"  spend                   ${report['estimated_spend_usd']} of ${cap:.2f}")
    if halted:
        print("  HALTED on the spend cap; the sample is smaller than requested.")
    if not report["is_real_model"]:
        print("  NOT a measurement of a draft model. The harness ran; no model did.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
