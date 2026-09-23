"""How accurate must a guesser be before guessing pays, and what does guessing a write add?

Spec task 6.4 asks for "the break-even alpha per workload" and every report so far has printed
it as not measured, because finding it needs predictors of known, controlled accuracy rather
than whatever alpha a real one happens to deliver. This supplies them: an oracle that knows the
turn the model will emit and names the right next call with probability ``alpha``, and
otherwise names a *valid* wrong one -- the same tool with a different real argument, so a miss
runs and is squashed exactly as a real miss is.

Three questions, each against the same baseline -- tier-0 early issue on, no guessing -- so
the answer is what branch speculation adds *on top of* what the runtime already does:

**What is a hit worth?** A guess can run at most one block ahead of the model's stream, since
the drafter is asked after each block and the next block resolves it. So a correct guess
saves at most one block of streaming time, and only when the call is slow enough that early
issue could not already have hidden it.

**What does a miss cost?** A wrong guess is squashed when the real block arrives, and the real
call is issued as it always was. The measurement is whether that is free in wall clock -- the
bill arrives elsewhere, as upstream reads and, for a draft model, tokens.

**What does speculating on a write add?** ``speculate_writes=False`` is PASTE's rule. The
store buffer is what makes the other setting safe, and this measures whether it is also
faster. A staged write cannot be dispatched before its branch retires, and the drafter does not
chain past one, so the prediction is that it adds nothing -- and a prediction is not a result.

No network and no model: the scripted target streams at ``--think-ms`` per block, calibrated
from the real sweep's measured turn times, so the numbers are reproducible on any machine.

``python bench/offline/run_break_even.py --out bench/results/break_even.json``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.online.run_latency import MeteredModel, Spend, difference_ci, run_one
from bench.online.run_latency_sweep import with_latency
from bench.online.scripted_target import ScriptedTarget
from bench.workloads import WORKLOADS, Workload

from specunode.canonical import JsonValue
from specunode.core.decision import ToolCall
from specunode.core.policy import Policy
from specunode.drafters.base import DraftContext, Prediction
from specunode.journal.journal import close_all_writers

#: The only sample app that hands its model turn to the runtime, and so the only one on which a
#: drafter is ever consulted. The other two call the model directly and issue each tool
#: themselves; guessing cannot touch them, whatever its accuracy.
WORKLOAD = "ops_agent"
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_LATENCIES = (500, 2000)
DEFAULT_TASKS = 8
#: Per-block streaming time for the stand-in model. The real sweep measured this workload's
#: strictly sequential run at ~1.7 s with instant tools, for a turn of three blocks.
DEFAULT_THINK_MS = 500.0
SEED = 20260923

#: A valid wrong answer for each call the turn makes: the same tool, a different real row.
#: A miss that named a row which does not exist would fail rather than be squashed, and that
#: is not what a wrong guess does.
NEAR_MISS: Mapping[str, tuple[str, JsonValue]] = {
    "get_pipeline_status": ("pipeline_id", "etl-4"),
    "fetch_runbook": ("section", "escalate"),
    "restart_job": ("job_id", "etl-4"),
}


class OracleDrafter:
    """Names the model's real next call with probability ``alpha``, deterministically.

    The draw for each position is a hash of the seed, the task and the position, so a run is
    reproducible and two arms given the same seed make the same guesses. It knows the turn in
    advance, which no real drafter does -- that is the point: it isolates what a guess of a given
    accuracy is worth from the question of how to build one.
    """

    tier = 1

    def __init__(self, truth: Sequence[ToolCall], alpha: float, seed: int, task: int) -> None:
        self._truth = list(truth)
        self._alpha = alpha
        self._seed = seed
        self._task = task
        self.offered = 0

    def _hit(self, position: int) -> bool:
        digest = hashlib.blake2b(
            f"{self._seed}:{self._task}:{position}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") / 2**64 < self._alpha

    async def predict(self, ctx: DraftContext) -> list[Prediction]:
        position = len(ctx.history)
        if position >= len(self._truth):
            return []
        true = self._truth[position]
        guess = true
        if not self._hit(position):
            key, alternative = NEAR_MISS[true.name]
            guess = ToolCall(true.name, {**dict(true.args), key: alternative})
        self.offered += 1
        return [Prediction(decision=guess, tier=1, score=1.0)]


async def _run(
    workload: Workload,
    policy: Policy,
    predictor: OracleDrafter | None,
    think_ms: float,
    root: Path,
    task: int,
) -> tuple[float, int, int]:
    target = MeteredModel(inner=ScriptedTarget(think_ms=think_ms), spend=Spend(cap_usd=1e9))
    row = await run_one(workload, "B_seq", task, target, root, policy=policy, predictor=predictor)
    return row.wall_ms, row.accepted, row.offered


async def measure(
    alphas: Sequence[float], latencies: Sequence[int], tasks: int, think_ms: float
) -> dict[str, Any]:
    base = next(w for w in WORKLOADS if w.name == WORKLOAD)
    truth = base.decisions()
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for ms in latencies:
            workload = with_latency(base, ms)
            baseline = [
                (await _run(workload, Policy(speculation=False), None, think_ms, root, t))[0]
                for t in range(tasks)
            ]
            point: dict[str, Any] = {
                "early_issue_only_ms": round(sum(baseline) / len(baseline), 1),
                "by_alpha": {},
            }
            for alpha in alphas:
                arms: dict[str, Any] = {}
                for label, policy in (
                    ("reads_and_writes", Policy(speculation=True)),
                    ("reads_only", Policy(speculation=True, speculate_writes=False)),
                ):
                    walls: list[float] = []
                    accepted = offered = 0
                    for t in range(tasks):
                        oracle = OracleDrafter(truth, alpha, SEED, t)
                        wall, hit, made = await _run(workload, policy, oracle, think_ms, root, t)
                        walls.append(wall)
                        accepted += hit
                        offered += made
                    arms[label] = {
                        "wall_ms_mean": round(sum(walls) / len(walls), 1),
                        "saving_vs_early_issue_ci95": difference_ci(walls, baseline),
                        "guesses_offered": offered,
                        "guesses_accepted": accepted,
                        "wall_ms": [round(w, 3) for w in walls],
                    }
                point["by_alpha"][f"{alpha:.2f}"] = arms
            point["early_issue_only_wall_ms"] = [round(w, 3) for w in baseline]
            out[str(ms)] = point
    close_all_writers()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench/results/break_even.json"))
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS)
    parser.add_argument("--think-ms", type=float, default=DEFAULT_THINK_MS)
    parser.add_argument("--alphas", default=",".join(f"{a}" for a in DEFAULT_ALPHAS))
    parser.add_argument("--latencies", default=",".join(str(m) for m in DEFAULT_LATENCIES))
    args = parser.parse_args(argv)

    alphas = [float(a) for a in str(args.alphas).split(",") if a.strip()]
    latencies = [int(m) for m in str(args.latencies).split(",") if m.strip()]
    started = time.monotonic()
    points = asyncio.run(measure(alphas, latencies, args.tasks, args.think_ms))
    report = {
        "bench": "break_even",
        "workload": WORKLOAD,
        "model": "scripted",
        "think_ms_per_block": args.think_ms,
        "tasks_per_point": args.tasks,
        "seed": SEED,
        "baseline": "tier-0 early issue on, no guessing (Policy(speculation=False))",
        "wall_seconds": round(time.monotonic() - started, 1),
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"BREAK-EVEN ({WORKLOAD}, model streams {args.think_ms:.0f} ms/block)")
    print("  saving over early issue alone, 95% interval on the difference")
    print(f"  {'tools':>7} {'alpha':>6}  {'guess reads and writes':>28}  {'guess reads only':>28}")
    for ms, point in points.items():
        for alpha, arms in point["by_alpha"].items():
            cells = []
            for label in ("reads_and_writes", "reads_only"):
                ci = arms[label]["saving_vs_early_issue_ci95"]
                cells.append(f"{ci['mean']:+.1%} [{ci['ci95_low']:+.1%}, {ci['ci95_high']:+.1%}]")
            print(f"  {ms + 'ms':>7} {alpha:>6}  {cells[0]:>28}  {cells[1]:>28}")
    print(f"\n  {report['wall_seconds']}s of wall clock, no model, no network.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
