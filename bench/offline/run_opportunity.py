"""How much speculation a corpus actually exposes (spec task 6.2). No model calls.

This is the analysis the project exists to be honest about. It measures, per trace, how far a
runtime could speculate before something stops it, under two policies:

**PASTE-style** — a tool with side effects is not speculated at all, so a write is a hard
barrier. This is what the prior work does, and it is safe.

**SpecuNode** — a write is staged rather than executed, so it is not a barrier *by itself*. But
a model turn after a staged write is, because the next turn would have to contain a placeholder
where the real result should be, and a model conditioned on a placeholder is deciding on a
different premise.

That second clause is what makes this measurement worth running rather than assuming. If a
corpus emits one tool call per model turn, then the call after a write is always in a new turn,
and staging the write buys exactly nothing. Publishing that when it happens is the point.

``python bench/offline/run_opportunity.py --out bench/results/opportunity.json``
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.corpus.effect_classes import effect_of

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "traces.json"
BOOTSTRAP = 2000


@dataclass(frozen=True)
class Step:
    tool: str
    arg_keys: tuple[str, ...]
    refs_prior_output: bool
    turn: int
    ordinal: int

    @property
    def is_read(self) -> bool:
        return effect_of(self.tool) == "read"

    @property
    def signature(self) -> str:
        return f"{self.tool}({','.join(self.arg_keys)})"


def load(path: Path) -> list[list[Step]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces: list[list[Step]] = []
    for trace in payload["traces"]:
        steps = [
            Step(
                tool=s["tool"],
                arg_keys=tuple(s.get("arg_keys", [])),
                refs_prior_output=bool(s.get("refs_prior_output")),
                turn=int(s.get("turn", 0)),
                ordinal=int(s.get("ordinal", 0)),
            )
            for s in trace["steps"]
        ]
        if steps:
            traces.append(steps)
    return traces


# -- per-trace measurements -------------------------------------------------------------------


def read_fraction(trace: Sequence[Step]) -> float:
    return sum(1 for s in trace if s.is_read) / len(trace)


def new_turn_fraction(trace: Sequence[Step]) -> float:
    """Calls that open a model turn rather than continuing one.

    The headline denominator. A call that opens a turn had to wait for the model to see the
    previous results, so nothing can speculate past a write into it.
    """
    return sum(1 for s in trace if s.ordinal == 0) / len(trace)


def refs_prior_fraction(trace: Sequence[Step]) -> float:
    return sum(1 for s in trace if s.refs_prior_output) / len(trace)


def paste_span(trace: Sequence[Step]) -> float:
    """Mean run length speculable under PASTE's policy: reads only, a write stops it."""
    spans: list[int] = []
    run = 0
    for step in trace:
        if step.is_read:
            run += 1
        else:
            spans.append(run)
            run = 0
    spans.append(run)
    return statistics.fmean(spans) if spans else 0.0


def specunode_post_write_span(trace: Sequence[Step]) -> float:
    """Mean number of calls speculable *after* a staged write, before a model turn stops it.

    A staged write does not block the next tool call; it blocks the next model *turn*. So the
    span past a write is the number of further calls in the same turn. On a corpus that emits
    one call per turn, that is zero by construction, and saying so is the honest result.
    """
    spans: list[int] = []
    for index, step in enumerate(trace):
        if step.is_read:
            continue
        span = 0
        for later in trace[index + 1 :]:
            if later.ordinal == 0:  # a new model turn: the barrier
                break
            span += 1
        spans.append(span)
    return statistics.fmean(spans) if spans else 0.0


def model_turn_consumes_write(trace: Sequence[Step]) -> float:
    """Steps that are a model turn immediately consuming a write's result.

    The shape that can gain nothing from past-write speculation, by design. Spec section 1 says
    the bench must report how much of each workload has it.
    """
    hits = 0
    for index in range(1, len(trace)):
        previous, current = trace[index - 1], trace[index]
        if not previous.is_read and current.ordinal == 0:
            hits += 1
    return hits / max(len(trace) - 1, 1)


def staged_not_skipped(trace: Sequence[Step]) -> float:
    """Steps PASTE must refuse to speculate on, that SpecuNode can stage instead.

    This is the opportunity the store buffer unlocks, and it is a different quantity from the
    within-turn span above. PASTE excludes a tool with side effects from speculation entirely;
    SpecuNode stages it, so a *predicted* write can be run ahead like any other call and
    discarded if the model decides otherwise. On a corpus that is almost all writes, this is
    almost all steps -- but it is only realisable to the extent the predictor is right, which
    is why the two numbers are reported together and neither is quoted alone.
    """
    return sum(1 for s in trace if not s.is_read) / len(trace)


def predictability(traces: Sequence[Sequence[Step]], top_k: int) -> float:
    """Leave-one-trace-out top-k accuracy of an order-2 index over tool signatures."""
    from specunode.core.decision import ToolCall
    from specunode.drafters.t1_pattern import PatternIndex

    def as_calls(trace: Sequence[Step]) -> list[ToolCall]:
        return [ToolCall(s.tool, dict.fromkeys(s.arg_keys, "")) for s in trace]

    hits = 0
    total = 0
    for held_out in range(len(traces)):
        index = PatternIndex(order=2)
        index.train([as_calls(t) for i, t in enumerate(traces) if i != held_out])
        calls = as_calls(traces[held_out])
        for position in range(1, len(calls)):
            ranked = [sig for sig, _ in index.rank(calls[:position])[:top_k]]
            actual = f"{calls[position].name}({','.join(sorted(calls[position].args))})"
            total += 1
            if actual in ranked:
                hits += 1
    return hits / total if total else 0.0


# -- statistics -------------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], rounds: int = BOOTSTRAP, seed: int = 20260916
) -> tuple[float, float, float]:
    """Mean and a 95% percentile bootstrap interval over traces."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.fmean(sample))
    means.sort()
    lower = means[int(0.025 * rounds)]
    upper = means[min(int(0.975 * rounds), rounds - 1)]
    return statistics.fmean(values), lower, upper


def measure(traces: Sequence[Sequence[Step]], predict_sample: int) -> dict[str, Any]:
    def stat(fn: Any) -> dict[str, float]:
        mean, low, high = bootstrap_ci([fn(t) for t in traces])
        return {"mean": round(mean, 4), "ci95_low": round(low, 4), "ci95_high": round(high, 4)}

    sample = list(traces[:predict_sample])
    return {
        "trajectories": len(traces),
        "steps": sum(len(t) for t in traces),
        "read_fraction": stat(read_fraction),
        "calls_opening_a_new_model_turn": stat(new_turn_fraction),
        "args_referencing_a_prior_result": stat(refs_prior_fraction),
        "paste_speculable_span": stat(paste_span),
        "steps_paste_must_skip_that_specunode_can_stage": stat(staged_not_skipped),
        "specunode_post_write_span": stat(specunode_post_write_span),
        "model_turn_consuming_a_write_result": stat(model_turn_consumes_write),
        "predictability": {
            "sampled_trajectories": len(sample),
            "top_1": round(predictability(sample, 1), 4),
            "top_3": round(predictability(sample, 3), 4),
        },
    }


# -- overhead (spec task 6.5) ------------------------------------------------------------------


async def measure_overhead(rounds: int) -> dict[str, Any]:
    """What journaling and classification cost, against the same graph running bare.

    The same LangGraph app, the same scripted model, the same fake world -- once through
    ``ainvoke`` with nothing wrapping it, and once through the runtime. The difference is the
    journal's fsyncs, the effect classification, the staging and the ledger.

    Reported per *step* as well as in total, because a per-run figure says more about how many
    steps the workload happens to have than about what the runtime costs.
    """
    import statistics as stats
    import tempfile
    import time

    from examples.support_agent.langgraph_agent import build_graph, build_registry

    from specunode.buffer.dispatcher import Dispatcher
    from specunode.core.model import JournaledModel
    from specunode.integrations.langgraph import wrap
    from specunode.journal.journal import Journal
    from specunode.testing.models import ScriptedModel, tool_turn
    from specunode.testing.world import standard_world

    charge = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})

    def scripted() -> ScriptedModel:
        return ScriptedModel(turns=[tool_turn(charge, turn=0)])

    bare: list[float] = []
    wrapped: list[float] = []
    steps = 0

    for _ in range(rounds):
        world = standard_world()
        graph = build_graph(world, scripted())
        started = time.perf_counter()
        await graph.ainvoke({"customer_id": "cus-1"})
        bare.append((time.perf_counter() - started) * 1000.0)

    tmp = Path(tempfile.mkdtemp())
    for index in range(rounds):
        world = standard_world()
        journal = Journal(tmp / f"overhead-{index}.db")
        registry = build_registry(world)
        runtime = wrap(
            build_graph(world, scripted()),
            registry=registry,
            journal=journal,
            target=JournaledModel(scripted(), journal, provider="scripted"),
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        )
        started = time.perf_counter()
        result = await runtime.run({"customer_id": "cus-1"})
        wrapped.append((time.perf_counter() - started) * 1000.0)
        steps = max(steps, result.steps)

    bare_mean = stats.fmean(bare)
    wrapped_mean = stats.fmean(wrapped)
    overhead = wrapped_mean - bare_mean
    return {
        "rounds": rounds,
        "steps_per_run": steps,
        "bare_ms": {"mean": round(bare_mean, 3), "median": round(stats.median(bare), 3)},
        "wrapped_ms": {"mean": round(wrapped_mean, 3), "median": round(stats.median(wrapped), 3)},
        "overhead_ms_per_run": round(overhead, 3),
        "overhead_ms_per_step": round(overhead / steps, 3) if steps else 0.0,
        "overhead_fraction_of_wall_clock": (
            round(overhead / wrapped_mean, 4) if wrapped_mean else 0.0
        ),
        "note": (
            "A scripted model answers instantly, so this is the worst case for the ratio: "
            "against a real multi-second turn the same absolute cost is a far smaller "
            "fraction. The absolute per-step figure is the one to compare."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline opportunity analysis")
    parser.add_argument("--overhead", action="store_true", help="measure runtime overhead (6.5)")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--predict-sample",
        type=int,
        default=25,
        help="trajectories used for leave-one-out predictability (it is quadratic)",
    )
    args = parser.parse_args(argv)

    if args.overhead:
        import asyncio

        report = {"overhead": asyncio.run(measure_overhead(args.rounds))}
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        o = report["overhead"]
        print()
        print("OVERHEAD (journaling + classification)")
        print(f"  bare graph:        {o['bare_ms']['mean']:.3f} ms/run")
        print(f"  under the runtime: {o['wrapped_ms']['mean']:.3f} ms/run")
        print(
            f"  overhead:          {o['overhead_ms_per_run']:.3f} ms/run "
            f"({o['overhead_ms_per_step']:.3f} ms/step)"
        )
        print(f"  as a fraction:     {o['overhead_fraction_of_wall_clock']:.1%} of wall clock")
        print(f"  {o['note']}")
        print()
        return 0

    if not args.corpus.is_file():
        print(f"no corpus at {args.corpus}; run bench/corpus/fetch.py first", file=sys.stderr)
        return 1

    traces = load(args.corpus)
    manifest = json.loads((args.corpus.parent / "manifest.json").read_text())
    report = {"corpus": manifest, "opportunity": measure(traces, args.predict_sample)}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    o = report["opportunity"]
    print()
    print("OPPORTUNITY ANALYSIS")
    print(f"  corpus: {manifest['source']} — {o['trajectories']} trajectories, {o['steps']} steps")
    print()
    for label, key in (
        ("reads (speculable under any policy)", "read_fraction"),
        ("calls that open a new model turn", "calls_opening_a_new_model_turn"),
        ("args referencing a prior result", "args_referencing_a_prior_result"),
        ("PASTE speculable span (calls)", "paste_speculable_span"),
        ("steps PASTE skips, SpecuNode stages", "steps_paste_must_skip_that_specunode_can_stage"),
        ("SpecuNode span past a write (calls)", "specunode_post_write_span"),
        ("model turn consuming a write result", "model_turn_consuming_a_write_result"),
    ):
        s = o[key]
        print(f"  {label:<38} {s['mean']:>8.4f}  [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]")
    p = o["predictability"]
    print(f"  {'T1 top-1 / top-3':<38} {p['top_1']:>8.4f}  / {p['top_3']:.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
