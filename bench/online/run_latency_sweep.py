"""At what tool latency does running ahead start to pay? (spec task 6.9)

``run_latency.py`` measured three sample apps against a real model and found no wall-clock
saving -- and on one workload a saving of -15.5% [-31.9%, -2.5%], a loss the interval does not
let anyone call noise. The detail that made that result worth chasing was the acceptance rate
beside it: **1.0**. Every guess on that workload was correct and speculation was still slower,
so the cost was never misprediction. It was the machinery, against tools that return in under
a millisecond because they are a dictionary in memory.

That is not the case speculation was built for. Running ahead hides tool latency; a tool with
no latency has nothing to hide. So the honest question is not "does it help?" but "how slow
does a tool have to be before it helps?", and that is a number rather than an opinion.

This sweeps the sample apps' tool latency -- every arm pays it, identically -- and reports the
wall-clock saving against the sequential arm at each point, with a bootstrap interval on the
*difference*. The crossing point, if there is one, is the answer. A sweep that never crosses is
also an answer, and a more damning one.

``ANTHROPIC_API_KEY=... python bench/online/run_latency_sweep.py --out bench/results/sweep.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.online.run_latency import (
    ARMS,
    DEFAULT_TARGET_MODEL,
    MeteredModel,
    Spend,
    TaskResult,
    build_target,
    difference_ci,
    prices_for,
    run_one,
)
from bench.workloads import WORKLOADS, Workload
from specunode.journal.journal import close_all_writers
from specunode.testing.world import World

#: Milliseconds of latency given to every tool. 0 is the world as the other benchmarks see it;
#: the rest bracket what a real tool costs -- a local database, a warm HTTP API, a cold one.
DEFAULT_LADDER = (0, 200, 500, 1000)
DEFAULT_TASKS = 8
DEFAULT_BUDGET_USD = 8.0


def with_latency(workload: Workload, ms: int) -> Workload:
    """The same workload whose tools all take ``ms`` to answer.

    Reads and writes alike: a sweep that slowed only reads would flatter the store buffer,
    whose whole claim is about writes.
    """

    def prepare(world: World, latency: int = ms) -> None:
        world.faults.slow("read", latency)
        world.faults.slow("write", latency)

    return replace(workload, prepare=prepare if ms else None)


async def sweep(
    workloads: Sequence[Workload],
    ladder: Sequence[int],
    tasks: int,
    target: MeteredModel,
    spend: Spend,
    root: Path,
) -> tuple[dict[str, Any], str | None]:
    out: dict[str, Any] = {}
    halted: str | None = None
    for ms in ladder:
        point: dict[str, Any] = {}
        for workload in workloads:
            slowed = with_latency(workload, ms)
            rows: list[TaskResult] = []
            for arm in ARMS:
                for task in range(tasks):
                    if not spend.may_continue():
                        halted = halted or f"{workload.name}@{ms}ms/{arm}/task {task}"
                        break
                    try:
                        rows.append(await run_one(slowed, arm, task, target, root))
                    except Exception as exc:  # a failed point is reported, never dropped
                        halted = halted or f"{workload.name}@{ms}ms/{arm}: {exc}"
                        break
            if not rows:
                continue

            def wall(arm: str, runs: list[TaskResult] = rows) -> list[float]:
                return [r.wall_ms for r in runs if r.arm == arm]

            means = {arm: round(sum(wall(arm)) / len(wall(arm)), 1) for arm in ARMS if wall(arm)}
            point[workload.name] = {
                "n_per_arm": tasks,
                "wall_ms_mean": means,
                "saving_vs_seq_ci95": {
                    arm: difference_ci(wall(arm), wall("B_seq"))
                    for arm in ("B_specunode", "B_readonly_spec")
                    if wall(arm) and wall("B_seq")
                },
                "saving_vs_strict_seq_ci95": {
                    arm: difference_ci(wall(arm), wall("B_strict_seq"))
                    for arm in ("B_seq", "B_readonly_spec", "B_specunode")
                    if wall(arm) and wall("B_strict_seq")
                },
                "leaks": sum(r.leaks for r in rows),
                "alpha": (
                    round(sum(r.accepted for r in rows) / sum(r.offered for r in rows), 4)
                    if sum(r.offered for r in rows)
                    else None
                ),
            }
        out[str(ms)] = point
        if halted:
            break
    return out, halted


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench/results/sweep.json"))
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS)
    parser.add_argument("--model", default="anthropic", help="anthropic|scripted")
    parser.add_argument(
        "--target-model", default=os.environ.get("SPECUNODE_MODEL", DEFAULT_TARGET_MODEL)
    )
    parser.add_argument(
        "--ladder",
        default=",".join(str(ms) for ms in DEFAULT_LADDER),
        help="tool latencies in milliseconds, comma separated",
    )
    args = parser.parse_args(argv)

    if args.model == "anthropic":
        if args.target_model == "scripted":
            raise SystemExit("--model anthropic needs a real --target-model")
        os.environ["SPECUNODE_MODEL"] = args.target_model

    ladder = [int(value) for value in str(args.ladder).split(",") if value.strip()]
    cap = float(os.environ.get("SPECUNODE_BENCH_BUDGET_USD", DEFAULT_BUDGET_USD))
    spend = Spend(cap_usd=cap, prices_per_mtok=prices_for(args.target_model))
    target = MeteredModel(inner=build_target(args.model), spend=spend)

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        points, halted = await sweep(WORKLOADS, ladder, args.tasks, target, spend, Path(directory))
    close_all_writers()

    report = {
        "bench": "latency_sweep",
        "is_real_model": args.model == "anthropic",
        "target_model": args.target_model if args.model == "anthropic" else "scripted",
        "ladder_ms": ladder,
        "tasks_per_arm": args.tasks,
        "halted_at": halted,
        "budget_usd_cap": cap,
        "estimated_spend_usd": round(spend.usd, 4),
        "prices_per_mtok_usd": dict(spend.prices_per_mtok),
        "wall_seconds": round(time.monotonic() - started, 1),
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"TOOL-LATENCY SWEEP ({report['target_model']}, real: {report['is_real_model']})")
    print(
        f"  {'tool latency':>13}  {'workload':<16} {'strict_seq':>10} {'specunode':>10}  "
        "saving vs strict_seq"
    )
    for ms, point in points.items():
        for name, entry in sorted(point.items()):
            means = entry["wall_ms_mean"]
            ci = entry["saving_vs_strict_seq_ci95"].get("B_specunode") or {}
            band = (
                f"{ci['mean']:+.1%} [{ci['ci95_low']:+.1%}, {ci['ci95_high']:+.1%}]" if ci else "-"
            )
            print(
                f"  {ms + 'ms':>13}  {name:<16} {means.get('B_strict_seq', 0):10.0f} "
                f"{means.get('B_specunode', 0):10.0f}  {band}"
            )
    print(
        f"\n  spend: ${report['estimated_spend_usd']} of ${cap:.2f}   "
        f"wall: {report['wall_seconds']}s"
    )
    if halted:
        print(f"  HALTED at {halted}")
    if not report["is_real_model"]:
        print("  NOT a measurement of a real model. The harness ran; no model did.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
