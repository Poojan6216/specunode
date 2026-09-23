"""Wall-clock latency against a real target model (spec task 6.4).

``python bench/online/run_latency.py --out bench/results/latency.json``

Every other benchmark here runs against a scripted model, so every latency claim this project
makes is currently an argument rather than a measurement. This is the one that is not -- and for
most of this build it did not exist, while the Final Report described it as written and blocked
only on a credential. That is the exact class of claim this project exists to make impossible,
so the runner now exists and says clearly which parts of it have run.

**It is runnable without an API key.** ``--model scripted`` drives the whole harness -- the three
arms, the timing, the accounting, the budget gate, the bootstrap -- against a deterministic
stand-in. That is what CI runs, so the code path is exercised even where the credential is not
available, and the only untested thing is the provider's own HTTP call. The lesson this project
learned the hard way from its MCP proxy is that a component nobody ever starts is a component
that does not work; a benchmark nobody can run without a credit card is the same thing.

**The budget is a gate, not a suggestion.** Spend is estimated from the tokens each turn
reports, checked before every model call, and the run stops at the cap. Decision Gate D2 says to
report the reduced *n* and its wider interval rather than raising the cap, so the output records
how many tasks actually completed and the report renders that rather than quietly averaging
fewer samples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.offline.run_opportunity import bootstrap_ci
from bench.workloads import WORKLOADS, Workload
from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import (
    JournaledModel,
    ModelClient,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TurnComplete,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
from specunode.ids import new_ulid
from specunode.journal.journal import Journal, close_all_writers
from specunode.testing.world import standard_world

#: The three arms section 6.4 names, plus the baseline they were all missing.
#:
#: ``B_strict_seq`` is what "no speculation" actually means: wait for the turn, then make the
#: calls in order. Every arm here used to run with tier-0 early issue on, ``B_seq`` included,
#: so the control group contained the treatment and the benchmark could not see what tier 0
#: was worth -- which is most of what this runtime does on a workload that never predicts.
ARMS = ("B_strict_seq", "B_seq", "B_readonly_spec", "B_specunode")

#: Default spend cap in USD. Overridden by SPECUNODE_BENCH_BUDGET_USD.
DEFAULT_BUDGET_USD = 25.0

#: The target model a real run asks for. The sample apps read it from ``SPECUNODE_MODEL`` at
#: call time and default to the deterministic stand-in, so a run that forgot to set it would
#: have sent ``"model": "scripted"`` to the Messages API and failed on the first call -- which
#: is what this runner did until someone was about to pay for it.
DEFAULT_TARGET_MODEL = "claude-sonnet-5"

#: Per-million-token list prices, per model, used to estimate spend. Declared here and printed
#: in the output, because an estimate whose inputs are hidden is not an estimate anyone can
#: check. They are *prices*, not measurements, and the output labels them that way.
#:
#: Keyed by model because one flat table was wrong twice: $3/$15 is the previous Sonnet's
#: price, so every run against Claude Sonnet 5 ($2/$10) overstated its bill by half again,
#: and the tier-2 benchmark was billed at those rates for a Haiku model at a third of them.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}
#: For a model not in the table: the highest price listed, so an estimate errs towards the
#: cap tripping early rather than late.
PRICE_PER_MTOK = {"input": 5.0, "output": 25.0}


def prices_for(model: str) -> dict[str, float]:
    return dict(PRICES_PER_MTOK.get(model, PRICE_PER_MTOK))


@dataclass
class Spend:
    """What the run has cost so far, and whether it may continue.

    The price table is a field rather than the module constant it used to read. A second
    benchmark metered a Haiku-class draft model through this class and was charged at the
    target model's rates -- three times over -- so it halted on a cap it had not reached and
    published a spend figure that was wrong by that factor.
    """

    cap_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    halted: bool = False
    prices_per_mtok: Mapping[str, float] = field(default_factory=lambda: dict(PRICE_PER_MTOK))

    @property
    def usd(self) -> float:
        # A cache write costs 1.25x the input price and a cache read 0.1x (the published
        # multipliers for the default five-minute entry).
        return (
            self.input_tokens * self.prices_per_mtok["input"]
            + self.cache_write_tokens * self.prices_per_mtok["input"] * 1.25
            + self.cache_read_tokens * self.prices_per_mtok["input"] * 0.1
            + self.output_tokens * self.prices_per_mtok["output"]
        ) / 1_000_000

    #: Cache writes and reads, which the API reports *outside* ``input_tokens``. With caching
    #: on, ``input_tokens`` is only the uncached remainder, so an estimate that read nothing
    #: else would under-count every request that wrote a cache entry and over-count none.
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def record(self, response: ModelResponse) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cache_write_tokens += response.usage.cache_creation_tokens
        self.cache_read_tokens += response.usage.cache_read_tokens

    def may_continue(self) -> bool:
        if self.usd >= self.cap_usd:
            self.halted = True
        return not self.halted


class BudgetExceeded(RuntimeError):
    """The spend cap was reached. Decision Gate D2: report fewer tasks, do not raise the cap."""


@dataclass
class MeteredModel:
    """Wraps the target, counts what it costs, and refuses once the cap is reached.

    In front of the client rather than inside it, so the same gate applies to any provider and
    so a scripted run exercises the identical accounting path.
    """

    inner: ModelClient
    spend: Spend
    calls: int = 0

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        if not self.spend.may_continue():
            raise BudgetExceeded(f"spend cap of ${self.spend.cap_usd:.2f} reached")
        self.calls += 1
        response = await self.inner.complete(envelope)
        self.spend.record(response)
        return response

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        """An async generator, not a coroutine that returns one.

        Returning ``self.inner.stream(envelope)`` from an ``async def`` hands the caller a
        coroutine, and ``async for`` over it fails. Being a generator is also what makes the
        spend accounting correct: a streamed turn reports its usage on ``TurnComplete``, so a
        wrapper that only forwarded the iterator would meter ``complete`` calls and let every
        streamed turn -- which is every turn the runtime itself drives -- go uncounted.
        """
        if not self.spend.may_continue():
            raise BudgetExceeded(f"spend cap of ${self.spend.cap_usd:.2f} reached")
        self.calls += 1
        async for event in self.inner.stream(envelope):
            if isinstance(event, TurnComplete):
                self.spend.record(event.response)
            yield event


@dataclass
class TaskResult:
    """One (workload, arm, task) run."""

    workload: str
    arm: str
    task: int
    wall_ms: float
    effects: int
    leaks: int
    stalls: dict[str, int] = field(default_factory=dict)
    wasted_tokens: int = 0
    accepted: int = 0
    offered: int = 0


def policy_for(arm: str) -> Policy:
    """The arms differ only in policy and predictor, so any difference is the runtime's.

    ``B_readonly_spec`` is PASTE's rule: the same predictor as ``B_specunode``, but a predicted
    write is a barrier rather than something to stage (``speculate_writes=False``). It used to
    be ``speculation=True`` with no predictor at all, justified as what "never speculate on a
    write" reduces to -- which forks nothing, is ``B_seq`` under another name, and made the
    comparison this project's thesis rests on a comparison against nothing.
    """
    if arm == "B_strict_seq":
        return Policy(speculation=False, early_issue=False)
    if arm == "B_seq":
        return Policy(speculation=False)
    if arm == "B_readonly_spec":
        return Policy(speculation=True, speculate_writes=False)
    return Policy(speculation=True)


def drafter_for(arm: str, workload: Workload) -> PatternDrafter | None:
    # Both speculative arms get the same predictor; only what they may do with a guess differs.
    if arm not in ("B_specunode", "B_readonly_spec"):
        return None
    index = PatternIndex(order=2)
    index.train([workload.decisions()])
    return PatternDrafter(index=index)


async def run_one(
    workload: Workload,
    arm: str,
    task: int,
    target: MeteredModel,
    root: Path,
    *,
    policy: Policy | None = None,
    predictor: object | None = None,
) -> TaskResult:
    """One run of one workload under one arm. ``policy``/``predictor`` override the arm's."""
    world = standard_world()
    adapter, registry = workload.make(world)
    journal = Journal(root / f"{workload.name}-{arm}-{task}.db")
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(target, journal, provider="metered"),
        policy=policy if policy is not None else policy_for(arm),
        predictor=predictor if predictor is not None else drafter_for(arm, workload),  # type: ignore[arg-type]
    )
    run_id = new_ulid()
    started = time.monotonic()
    result = await scheduler.run(run_id, dict(workload.seed))
    wall_ms = (time.monotonic() - started) * 1000.0
    if not result.ok:
        # The cap is reached *inside* a node, so the scheduler catches it and reports a node
        # fault -- it does not know a spend gate from any other tool failure. Recognising it
        # here is what turns "the benchmark crashed" into "the benchmark stopped where it said
        # it would", which is the whole of Decision Gate D2's behaviour.
        if target.spend.halted or "BudgetExceeded" in str(result.error):
            raise BudgetExceeded(str(result.error))
        raise SystemExit(f"{workload.name}/{arm}/task {task} did not complete: {result.error}")

    retired = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    leaks = len(world.mutating_branches() - retired)
    return TaskResult(
        workload=workload.name,
        arm=arm,
        task=task,
        wall_ms=wall_ms,
        effects=len(world.mutations),
        leaks=leaks,
        stalls=dict(scheduler.counters.stalls_by_hazard),
        wasted_tokens=scheduler.counters.wasted_tokens,
        # Guesses, counted where guesses are made. This used to be ``branches_forked -
        # branches_squashed``, but every node visit forks a canonical branch as well, so
        # ordinary steps were counted as correct guesses: a workload whose drafter never
        # offered anything reported an acceptance rate of 1.0, and that figure was quoted as
        # "a perfect predictor, and still slower". There had been no predictions at all.
        accepted=sum(turn.confirmed for turn in scheduler._turns),
        offered=sum(turn.confirmed + turn.squashed + turn.stalled for turn in scheduler._turns),
    )


def difference_ci(
    treatment: Sequence[float], control: Sequence[float], rounds: int = 2000, seed: int = 20260918
) -> dict[str, float]:
    """Bootstrap interval for the *difference* of two arms' means, as a share of the control.

    Two overlapping intervals do not mean two means are indistinguishable, and two separated
    ones are not a test either -- the comparison has to be made on the difference itself. This
    resamples both arms independently, because the arms are repeats of the same inputs rather
    than paired observations, and reports the saving `1 - treatment/control` with an interval.
    An interval that spans zero is the honest way to say "no difference was resolved here".
    """
    import random as _random
    import statistics as _stats

    if not treatment or not control:
        return {}
    rng = _random.Random(seed)
    savings: list[float] = []
    for _ in range(rounds):
        a = _stats.fmean(treatment[rng.randrange(len(treatment))] for _ in treatment)
        b = _stats.fmean(control[rng.randrange(len(control))] for _ in control)
        if b > 0:
            savings.append(1.0 - a / b)
    if not savings:
        return {}
    savings.sort()
    return {
        "mean": round(_stats.fmean(savings), 4),
        "ci95_low": round(savings[int(0.025 * len(savings))], 4),
        "ci95_high": round(savings[min(int(0.975 * len(savings)), len(savings) - 1)], 4),
    }


def summarise(rows: Sequence[TaskResult]) -> dict[str, object]:
    """Per (workload, arm): mean wall clock with a bootstrap interval over tasks."""
    out: dict[str, object] = {}
    workloads = sorted({row.workload for row in rows})
    for name in workloads:
        per_arm: dict[str, object] = {}
        for arm in ARMS:
            values = [row.wall_ms for row in rows if row.workload == name and row.arm == arm]
            if not values:
                continue
            mean, low, high = bootstrap_ci(values)
            per_arm[arm] = {
                "n": len(values),
                "wall_ms_mean": round(mean, 3),
                "wall_ms_ci95": [round(low, 3), round(high, 3)],
                "effects": sorted({r.effects for r in rows if r.workload == name and r.arm == arm}),
                "leaks": sum(r.leaks for r in rows if r.workload == name and r.arm == arm),
                # Every run's wall clock, so the numbers can be re-analysed without paying for
                # the benchmark again. An aggregate nobody can reopen is a dead end.
                "wall_ms": [round(v, 3) for v in values],
            }
        base = per_arm.get("B_strict_seq")
        seq = per_arm.get("B_seq")
        spec = per_arm.get("B_specunode")
        reduction = None
        if isinstance(seq, dict) and isinstance(spec, dict):
            base = float(seq["wall_ms_mean"])  # type: ignore[arg-type]
            if base > 0:
                reduction = round(1.0 - float(spec["wall_ms_mean"]) / base, 4)  # type: ignore[arg-type]

        def arm_values(arm: str, workload: str = name) -> list[float]:
            return [r.wall_ms for r in rows if r.workload == workload and r.arm == arm]

        accepted = [r.accepted for r in rows if r.workload == name and r.arm == "B_specunode"]
        offered = [r.offered for r in rows if r.workload == name and r.arm == "B_specunode"]
        alpha = round(sum(accepted) / sum(offered), 4) if offered and sum(offered) > 0 else None
        strict_reduction = None
        if isinstance(base, dict) and isinstance(spec, dict):
            floor = float(base["wall_ms_mean"])  # type: ignore[arg-type]
            if floor > 0:
                strict_reduction = round(1.0 - float(spec["wall_ms_mean"]) / floor, 4)  # type: ignore[arg-type]
        out[name] = {
            "arms": per_arm,
            "wall_clock_reduction_vs_seq": reduction,
            "wall_clock_reduction_vs_strict_seq": strict_reduction,
            # The difference with an interval, which is the comparison the point estimate above
            # only gestures at. Reported for both speculative arms against the sequential one.
            "saving_vs_seq_ci95": {
                "B_specunode": difference_ci(arm_values("B_specunode"), arm_values("B_seq")),
                "B_readonly_spec": difference_ci(
                    arm_values("B_readonly_spec"), arm_values("B_seq")
                ),
            },
            # Against the arm that does none of it. ``B_seq`` answers "what does *branch*
            # speculation add on top of early issue?"; this answers "what does the runtime buy
            # at all?", which is the question a reader thinks the table is answering.
            "saving_vs_strict_seq_ci95": {
                arm: difference_ci(arm_values(arm), arm_values("B_strict_seq"))
                for arm in ("B_seq", "B_readonly_spec", "B_specunode")
            },
            "alpha_observed": alpha,
            # The break-even alpha needs a sweep, which needs a budget this runner does not
            # assume it has. Reported as null rather than guessed; see the Final Report.
            "alpha_break_even": None,
        }
    return out


def build_target(kind: str) -> ModelClient:
    """The real provider, or a deterministic stand-in that exercises the same harness."""
    if kind == "scripted":
        from bench.online.scripted_target import ScriptedTarget

        return ScriptedTarget()
    if kind == "anthropic":
        from specunode.integrations.anthropic import AnthropicModel

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Run with --model scripted to exercise the "
                "harness without a credential; the numbers it produces are not latency "
                "measurements of a real model and the output says so."
            )
        return AnthropicModel()
    raise SystemExit(f"unknown --model {kind!r}; use 'anthropic' or 'scripted'")


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench/results/latency.json"))
    parser.add_argument("--tasks", type=int, default=30, help="seeded tasks per workload per arm")
    parser.add_argument("--model", default="anthropic", help="anthropic|scripted")
    parser.add_argument(
        "--target-model",
        default=os.environ.get("SPECUNODE_MODEL", DEFAULT_TARGET_MODEL),
        help=f"model id the workloads ask for (default {DEFAULT_TARGET_MODEL})",
    )
    args = parser.parse_args(argv)

    if args.model == "anthropic":
        if args.target_model == "scripted":
            raise SystemExit(
                "--model anthropic with --target-model scripted would send "
                '"model": "scripted" to the Messages API. Pass a real model id.'
            )
        # The workloads read this when they build their envelope, so it has to be set before
        # any of them runs -- and read at call time, not at import, for the same reason.
        os.environ["SPECUNODE_MODEL"] = args.target_model

    cap = float(os.environ.get("SPECUNODE_BENCH_BUDGET_USD", DEFAULT_BUDGET_USD))
    spend = Spend(cap_usd=cap, prices_per_mtok=prices_for(args.target_model))
    target = MeteredModel(inner=build_target(args.model), spend=spend)

    rows: list[TaskResult] = []
    halted_at: str | None = None
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for workload in WORKLOADS:
            for arm in ARMS:
                for task in range(args.tasks):
                    try:
                        rows.append(await run_one(workload, arm, task, target, root))
                    except BudgetExceeded:
                        halted_at = f"{workload.name}/{arm}/task {task}"
                        break
                if halted_at:
                    break
            if halted_at:
                break
    close_all_writers()

    report = {
        "bench": "latency",
        "model": args.model,
        "is_real_model": args.model != "scripted",
        "target_model": args.target_model if args.model == "anthropic" else "scripted",
        "tasks_requested": args.tasks,
        "tasks_completed": len(rows),
        "budget_usd_cap": cap,
        "estimated_spend_usd": round(spend.usd, 4),
        "prices_per_mtok_usd": dict(spend.prices_per_mtok),
        "halted_at": halted_at,
        "total_leaks": sum(row.leaks for row in rows),
        "workloads": summarise(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"  model: {args.model} (real: {report['is_real_model']})")
    print(f"  tasks completed: {len(rows)} of {args.tasks * len(WORKLOADS) * len(ARMS)}")
    print(f"  estimated spend: ${spend.usd:.4f} of ${cap:.2f}")
    if halted_at:
        print(f"  HALTED at {halted_at}: the spend cap was reached.")
        print("  Decision Gate D2: report the reduced n and its wider interval, not a raised cap.")
    if report["total_leaks"]:
        print(f"  *** {report['total_leaks']} LEAKS -- this must be zero ***")
        return 1
    if not report["is_real_model"]:
        print("  These are NOT latency measurements of a real model. The harness ran; the")
        print("  provider did not. Run with an API key for figures anyone should quote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
