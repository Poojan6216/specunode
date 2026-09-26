"""Fewer replies, prompt caching and parallel nodes, against a real model.

``bench/offline/run_model_bound.py`` measures what the runtime does with each shape of reply,
against a stand-in whose timings are settings. This measures what a real model does:

**Does it ask for several calls at once when invited to?** ``examples/incident_agent`` under
three prompts: ``one_call`` (how every turn in the corpus behaves -- enforced with the API's
``disable_parallel_tool_use``, because a first attempt at this benchmark found that asking in
the prompt did not stop the model batching), ``default`` (says nothing either way, so it
measures the model's own habit) and ``parallel`` (Anthropic's guidance for independent calls).
Counted: replies, calls per reply, and whether the run changed the world exactly as a correct
run does.

**What does prompt caching save?** The same agent with ``cache`` off and on -- its wall clock,
and its bill from the usage the API reports. Runs of one configuration follow each other well
inside the cache's five-minute lifetime, so from the second round on a run starts with its
system prompt already cached: the steady state of an agent that works more than one alert.

**What do parallel nodes save?** ``examples/fanout_agent`` with ``parallel_nodes`` off and on.

Cells run round-robin, one run of each per round, so drift in the API's latency lands on all of
them alike and a budget halt leaves every cell with nearly the same n. The first round is the
smoke test: any run that fails -- rather than completing and getting something wrong, which is
a result -- stops the benchmark there, before it has spent more than a round.

``ANTHROPIC_API_KEY=... python bench/online/run_model_bound.py``
``--out bench/results/model_bound_online.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.fanout_agent.agent import PIPELINES, KeyedScriptedModel
from examples.fanout_agent.agent import build as build_fanout
from examples.incident_agent.agent import build as build_incident
from examples.incident_agent.agent import expected_world_changes, scripted_replies

from bench.online.run_latency import (
    DEFAULT_TARGET_MODEL,
    BudgetExceeded,
    MeteredModel,
    Spend,
    difference_ci,
    prices_for,
)
from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import JournaledModel, ModelClient
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal, close_all_writers
from specunode.testing.models import ScriptedModel
from specunode.testing.world import standard_world

DEFAULT_RUNS = 15
DEFAULT_BUDGET_USD = 5.0


@dataclass(frozen=True)
class Cell:
    """One configuration: an agent, and the one or two settings that vary."""

    name: str
    agent: str  # "incident" | "fanout"
    style: str = "parallel"
    cache: bool = True
    parallel_nodes: bool = True


CELLS = (
    # Before: one call per reply, nothing cached.
    Cell("one_call/no_cache", "incident", style="one_call", cache=False),
    Cell("one_call/cache", "incident", style="one_call", cache=True),
    Cell("parallel/no_cache", "incident", style="parallel", cache=False),
    # After: both levers.
    Cell("parallel/cache", "incident", style="parallel", cache=True),
    Cell("default/cache", "incident", style="default", cache=True),
    Cell("nodes/one_at_a_time", "fanout", parallel_nodes=False),
    Cell("nodes/side_by_side", "fanout", parallel_nodes=True),
)

#: (label, control, treatment) -- each lever alone, and the whole before-and-after.
COMPARISONS = (
    ("caching", "one_call/no_cache", "one_call/cache"),
    ("multi_call_replies", "one_call/cache", "parallel/cache"),
    ("before_to_after", "one_call/no_cache", "parallel/cache"),
    ("parallel_nodes", "nodes/one_at_a_time", "nodes/side_by_side"),
)


class RunFailed(RuntimeError):
    """A run that did not complete. Not a wrong answer -- that is a result -- a broken setup."""


def _target(kind: str, cell: Cell) -> ModelClient:
    if kind == "anthropic":
        from specunode.integrations.anthropic import AnthropicModel

        return AnthropicModel(cache=cell.cache)
    if cell.agent == "fanout":
        return KeyedScriptedModel(think_ms=20.0)
    # The stand-in plays what a model following the style would say; "default" has no script
    # of its own, so it plays the one-call replies. A dry run checks the harness, not a model.
    style = cell.style if cell.style != "default" else "one_call"
    return ScriptedModel(turns=scripted_replies(style), reply_delay_ms=20.0)


async def run_cell(cell: Cell, kind: str, spend: Spend) -> dict[str, Any]:
    world = standard_world()
    if cell.agent == "incident":
        adapter, registry = build_incident(world, cell.style)
    else:
        adapter, registry = build_fanout(world)
    before = (spend.input_tokens, spend.output_tokens, spend.cache_write_tokens,
              spend.cache_read_tokens, spend.usd)  # fmt: skip
    model = MeteredModel(inner=_target(kind, cell), spend=spend)
    with tempfile.TemporaryDirectory() as directory:
        journal = Journal(Path(directory) / "run.db")
        scheduler = Scheduler(
            graph=adapter,
            registry=registry,  # type: ignore[arg-type]
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),  # type: ignore[arg-type]
            target=JournaledModel(model, journal, provider=kind),
            policy=Policy(speculation=False, parallel_nodes=cell.parallel_nodes),
        )
        run_id = new_ulid()
        started = time.monotonic()
        result = await scheduler.run(run_id, {})
        wall_ms = (time.monotonic() - started) * 1000.0
        if not result.ok:
            if spend.halted or "BudgetExceeded" in str(result.error):
                raise BudgetExceeded(str(result.error))
            raise RunFailed(f"{cell.name}: {result.error}")
        retired = {
            str(entry.payload["branch_id"])
            for entry in journal.read(run_id, kinds=["branch_resolved"])
            if entry.payload.get("status") == "retired"
        }
        leaks = len(world.mutating_branches() - retired)
        chain_ok = journal.verify_chain(run_id).ok
        close_all_writers()

    row: dict[str, Any] = {
        "wall_ms": round(wall_ms, 1),
        "leaks": leaks,
        "input_tokens": spend.input_tokens - before[0],
        "output_tokens": spend.output_tokens - before[1],
        "cache_write_tokens": spend.cache_write_tokens - before[2],
        "cache_read_tokens": spend.cache_read_tokens - before[3],
        "usd": round(spend.usd - before[4], 5),
    }
    if cell.agent == "incident":
        changes = {
            "restarted": sorted(m.row_id for m in world.mutations if m.tool == "restart_job"),
            "summaries": sum(1 for m in world.mutations if m.tool == "post_summary"),
        }
        row.update(
            replies=result.state["turns"],
            calls=result.state["calls"],
            stopped=result.state["stopped"],
            correct=changes == expected_world_changes() and chain_ok,
            world_changes=changes,
        )
    else:
        posts = [str(r["text"]) for r in world.tables["messages"].values()]
        row.update(
            model_calls=model.calls,
            correct=len(posts) == 1
            and all(f"{p}: unknown" not in posts[0] for p in PIPELINES)
            and chain_ok,
        )
    return row


def summarise(cell: Cell, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    walls = [row["wall_ms"] for row in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "wall_ms": walls,
        "wall_ms_mean": round(statistics.fmean(walls), 1) if walls else None,
        "correct": sum(1 for row in rows if row["correct"]),
        "leaks": sum(row["leaks"] for row in rows),
        "usd_per_run": round(statistics.fmean(r["usd"] for r in rows), 5) if rows else None,
    }
    if cell.agent == "incident" and rows:
        replies = [row["replies"] for row in rows]
        calls = [row["calls"] for row in rows]
        # Every stop but the turn cap ends on a reply that asked for no tools.
        tool_replies = [r["replies"] - (0 if r["stopped"] == "max_turns" else 1) for r in rows]
        cached = sum(r["cache_read_tokens"] for r in rows)
        total_in = sum(r["input_tokens"] + r["cache_write_tokens"] + r["cache_read_tokens"]
                       for r in rows)  # fmt: skip
        out.update(
            replies_mean=round(statistics.fmean(replies), 2),
            replies=replies,
            calls_per_reply=round(sum(calls) / max(1, sum(tool_replies)), 2),
            cache_read_share=round(cached / total_in, 4) if total_in else 0.0,
            stopped_at_max_turns=sum(1 for r in rows if r["stopped"] == "max_turns"),
        )
    return out


async def measure(kind: str, runs: int, spend: Spend, *, progress: bool = False) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {cell.name: [] for cell in CELLS}
    halted_at: str | None = None
    failure: str | None = None
    for round_index in range(runs):
        for cell in CELLS:
            try:
                row = await run_cell(cell, kind, spend)
            except BudgetExceeded:
                halted_at = f"round {round_index + 1}, {cell.name}"
                break
            except RunFailed as exc:
                failure = f"round {round_index + 1}: {exc}"
                if progress:
                    print(f"  FAILED {failure}", flush=True)
                break
            rows[cell.name].append(row)
            if progress:
                # A paid run takes minutes; the first round is the one worth watching.
                print(
                    f"  r{round_index + 1:02d} {cell.name:20s} {row['wall_ms']:8.0f} ms  "
                    f"replies={row.get('replies', '-')} calls={row.get('calls', '-')} "
                    f"correct={row['correct']} ${row['usd']:.4f}  total ${spend.usd:.3f}",
                    flush=True,
                )
        if halted_at or failure:
            break
    cells = {cell.name: summarise(cell, rows[cell.name]) for cell in CELLS}
    comparisons = {
        label: {
            "wall_saving_ci95": difference_ci(
                [r["wall_ms"] for r in rows[treated]], [r["wall_ms"] for r in rows[control]]
            ),
            "cost_saving_ci95": difference_ci(
                [r["usd"] for r in rows[treated]], [r["usd"] for r in rows[control]]
            ),
        }
        for label, control, treated in COMPARISONS
    }
    return {
        "cells": cells,
        "comparisons": comparisons,
        "halted_at": halted_at,
        "failure": failure,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("bench/results/model_bound_online.json"))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="rounds; one run per cell")
    parser.add_argument("--model", default="anthropic", help="anthropic|scripted")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument(
        "--budget",
        type=float,
        default=float(os.environ.get("SPECUNODE_BENCH_BUDGET_USD", DEFAULT_BUDGET_USD)),
    )
    args = parser.parse_args(argv)

    if args.model == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set; use --model scripted for a dry run")
        if args.target_model == "scripted":
            raise SystemExit("--model anthropic needs a real --target-model")
        os.environ["SPECUNODE_MODEL"] = args.target_model
    elif args.model != "scripted":
        raise SystemExit(f"unknown --model {args.model!r}; use 'anthropic' or 'scripted'")

    spend = Spend(cap_usd=args.budget, prices_per_mtok=prices_for(args.target_model))
    started = time.monotonic()
    measured = asyncio.run(measure(args.model, args.runs, spend, progress=True))
    report = {
        "bench": "model_bound_online",
        "is_real_model": args.model == "anthropic",
        "target_model": args.target_model if args.model == "anthropic" else "scripted",
        "runs_requested": args.runs,
        "budget_usd_cap": args.budget,
        "estimated_spend_usd": round(spend.usd, 4),
        "prices_per_mtok_usd": dict(spend.prices_per_mtok),
        "policy": {"speculation": False, "early_issue": True},
        **measured,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nMODEL-BOUND, {report['target_model']} (real: {report['is_real_model']})")
    for name, cell in report["cells"].items():
        extra = (
            f"replies {cell.get('replies_mean')}  calls/reply {cell.get('calls_per_reply')}  "
            f"cache-read share {cell.get('cache_read_share')}"
            if "replies_mean" in cell
            else ""
        )
        print(
            f"  {name:20s} n={cell['n']:2d}  {cell['wall_ms_mean'] or 0:8.0f} ms  "
            f"correct {cell['correct']}/{cell['n']}  leaks {cell['leaks']}  "
            f"${cell['usd_per_run'] or 0:.4f}/run  {extra}"
        )
    for label, comparison in report["comparisons"].items():
        print(f"  {label:20s} time {comparison['wall_saving_ci95']}")
        print(f"  {'':20s} cost {comparison['cost_saving_ci95']}")
    print(f"  spend ${report['estimated_spend_usd']} of ${args.budget:.2f}")
    if report["halted_at"]:
        print(f"  HALTED at the cap: {report['halted_at']}")
    if report["failure"]:
        print(f"  STOPPED on a failed run: {report['failure']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
