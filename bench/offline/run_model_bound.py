"""Fewer model replies, and model replies side by side: what the runtime does with each.

Two changes aimed at runs where the model, not the tools, is what everything waits on. When a
reply takes seconds, infrastructure has two levers left: how many replies a run needs, and how
many of them it waits on at once.

**Multi-call replies** (``examples/incident_agent``). One alert, worked by a model that asks for
one tool call per reply -- how every turn in the 300 real trajectories in ``bench/corpus``
behaves -- and by one that asks for every independent call at once. The first needs nine
replies, the second four, for the same eight calls. The runtime's part is to run a reply's reads
together as they parse, hold its writes until the reply is durable, and hand every result back
in one message.

**Parallel nodes** (``examples/fanout_agent``). Three independent checks, each one model reply,
run one after another (``parallel_nodes=False``) and side by side.

No model and no network. The stand-in waits ``--reply-ms`` before a reply's first block --
reading the prompt and thinking, the part of a reply that does not grow with what it says --
and ``--block-ms`` before each block and once more before the reply ends. Those are settings,
not measurements, and the results file records them beside what they produced. Two counts do
not depend on them at all: replies per run, and model calls in flight at once. Whether a real
model actually asks for several calls at once when told it may is a question only a real
model can answer, and this does not try to.

``python bench/offline/run_model_bound.py --out bench/results/model_bound.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.online.run_latency import difference_ci
from examples.fanout_agent.agent import PIPELINES, KeyedScriptedModel
from examples.fanout_agent.agent import build as build_fanout
from examples.incident_agent.agent import build as build_incident
from examples.incident_agent.agent import expected_world_changes, scripted_replies

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import JournaledModel, ModelClient
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal, close_all_writers
from specunode.testing.models import ScriptedModel
from specunode.testing.world import World, standard_world

REPLY_MS = 1500.0
BLOCK_MS = 300.0
TOOL_LATENCIES_MS = (0, 300)
RUNS = 5
STYLES = ("one_call", "parallel")


async def _run(
    world: World, adapter: object, registry: object, model: ModelClient, policy: Policy
) -> tuple[RunResult, float, int, bool]:
    """One run; its result, wall clock, leak count, and whether its journal chain verifies."""
    with tempfile.TemporaryDirectory() as directory:
        journal = Journal(Path(directory) / "run.db")
        scheduler = Scheduler(
            graph=adapter,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),  # type: ignore[arg-type]
            target=JournaledModel(model, journal, provider="scripted"),
            policy=policy,
        )
        run_id = new_ulid()
        started = time.monotonic()
        result = await scheduler.run(run_id, {})
        wall_ms = (time.monotonic() - started) * 1000.0
        if not result.ok:
            raise SystemExit(f"a run did not complete: {result.error}")
        retired = {
            str(entry.payload["branch_id"])
            for entry in journal.read(run_id, kinds=["branch_resolved"])
            if entry.payload.get("status") == "retired"
        }
        leaks = len(world.mutating_branches() - retired)
        chain_ok = journal.verify_chain(run_id).ok
        close_all_writers()
    return result, wall_ms, leaks, chain_ok


def _slowed(tool_ms: int) -> World:
    world = standard_world()
    if tool_ms:
        # Reads and writes alike, as in the latency sweep.
        world.slow("read", tool_ms)
        world.slow("write", tool_ms)
    return world


async def replies(style: str, tool_ms: int, reply_ms: float, block_ms: float) -> dict[str, Any]:
    world = _slowed(tool_ms)
    adapter, registry = build_incident(world, style)
    model = ScriptedModel(
        turns=scripted_replies(style), reply_delay_ms=reply_ms, block_delay_ms=block_ms
    )
    result, wall_ms, leaks, chain_ok = await _run(
        world, adapter, registry, model, Policy(speculation=False)
    )
    changes = {
        "restarted": sorted(m.row_id for m in world.mutations if m.tool == "restart_job"),
        "summaries": sum(1 for m in world.mutations if m.tool == "post_summary"),
    }
    return {
        "wall_ms": wall_ms,
        "replies": result.state["turns"],
        "calls": result.state["calls"],
        "correct": changes == expected_world_changes() and chain_ok,
        "leaks": leaks,
    }


async def branches(
    parallel: bool, tool_ms: int, reply_ms: float, block_ms: float
) -> dict[str, Any]:
    world = _slowed(tool_ms)
    adapter, registry = build_fanout(world)
    # A check's reply is one tool call: the start of the reply, the block, and its end.
    model = KeyedScriptedModel(think_ms=reply_ms + 2 * block_ms)
    _result, wall_ms, leaks, chain_ok = await _run(
        world, adapter, registry, model, Policy(speculation=False, parallel_nodes=parallel)
    )
    posts = [row["text"] for row in world.tables["messages"].values()]
    return {
        "wall_ms": wall_ms,
        "in_flight_max": model.high_water,
        "model_calls": len(model.calls),
        "correct": len(posts) == 1
        and all(f"{p}: unknown" not in str(posts[0]) for p in PIPELINES)
        and chain_ok,
        "leaks": leaks,
        "summary": posts[0] if posts else None,
    }


async def handwritten_replies(tool_ms: int, reply_ms: float, block_ms: float) -> dict[str, Any]:
    """The fastest loop a developer would write by hand, and what the runtime's safety is priced
    against: stream the reply, then run every call it asked for at once. No journal, no store
    buffer, no order among the writes, and nothing a crash could be resumed from."""
    from dataclasses import replace

    from examples.incident_agent.agent import MAX_TURNS, build_tools, envelope

    from specunode.core.model import Message, ToolResultBlock, TurnComplete

    world = _slowed(tool_ms)
    tools = {fn.__name__: fn for fn in build_tools(world)}  # type: ignore[attr-defined]
    model = ScriptedModel(
        turns=scripted_replies("parallel"), reply_delay_ms=reply_ms, block_delay_ms=block_ms
    )
    request = envelope("parallel")
    messages = list(request.messages)
    replies = 0
    started = time.monotonic()
    for _ in range(MAX_TURNS):
        response = None
        async for event in model.stream(replace(request, messages=tuple(messages))):
            if isinstance(event, TurnComplete):
                response = event.response
        replies += 1
        assert response is not None
        uses = response.tool_uses
        if not uses:
            break
        results = await asyncio.gather(*(tools[use.name](**dict(use.args)) for use in uses))
        messages.append(Message(role="assistant", content=response.content))
        messages.append(
            Message(
                role="user",
                content=tuple(
                    ToolResultBlock(tool_use_id=use.id, content=result)
                    for use, result in zip(uses, results, strict=True)
                ),
            )
        )
    wall_ms = (time.monotonic() - started) * 1000.0
    changes = {
        "restarted": sorted(m.row_id for m in world.mutations if m.tool == "restart_job"),
        "summaries": sum(1 for m in world.mutations if m.tool == "post_summary"),
    }
    return {
        "wall_ms": wall_ms,
        "replies": replies,
        "correct": changes == expected_world_changes(),
        "leaks": 0,
    }


async def handwritten_branches(tool_ms: int, reply_ms: float, block_ms: float) -> dict[str, Any]:
    """The same three checks and report, by hand: ``asyncio.gather`` over the checks."""
    from examples.fanout_agent.agent import MAX_TOKENS, SYSTEM, TOOL_DEFS
    from examples.fanout_agent.agent import build_tools as fanout_tools

    from specunode.core.model import Message, RequestEnvelope, TextBlock, TurnComplete

    world = _slowed(tool_ms)
    tools = {fn.__name__: fn for fn in fanout_tools(world)}  # type: ignore[attr-defined]
    model = KeyedScriptedModel(think_ms=reply_ms + 2 * block_ms)

    async def check(pipeline: str) -> Any:
        request = RequestEnvelope(
            model="scripted",
            system=(TextBlock(text=SYSTEM),),
            messages=(Message(role="user", content=(TextBlock(text=f"Check {pipeline}."),)),),
            tools=TOOL_DEFS,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        response = None
        async for event in model.stream(request):
            if isinstance(event, TurnComplete):
                response = event.response
        assert response is not None
        results = await asyncio.gather(
            *(tools[use.name](**dict(use.args)) for use in response.tool_uses)
        )
        return results[-1]

    started = time.monotonic()
    found = await asyncio.gather(*(check(pipeline) for pipeline in PIPELINES))
    lines = []
    for pipeline, status in zip(PIPELINES, found, strict=True):
        value = status.get("value") if isinstance(status, dict) else None
        lines.append(f"{pipeline}: {value.get('status') if isinstance(value, dict) else 'unknown'}")
    await tools["post_summary"](channel="#ops", text="; ".join(lines))
    wall_ms = (time.monotonic() - started) * 1000.0
    posts = [row["text"] for row in world.tables["messages"].values()]
    return {
        "wall_ms": wall_ms,
        "in_flight_max": model.high_water,
        "model_calls": len(model.calls),
        "correct": len(posts) == 1 and "unknown" not in str(posts[0]),
        "leaks": 0,
        "summary": posts[0] if posts else None,
    }


def _cost(runtime: Sequence[dict[str, Any]], by_hand: Sequence[dict[str, Any]]) -> float:
    """How much longer the runtime took than the loop by hand: mean over mean, minus one."""
    ours = statistics.fmean(row["wall_ms"] for row in runtime)
    theirs = statistics.fmean(row["wall_ms"] for row in by_hand)
    return round(ours / theirs - 1.0, 4)


def _cell(rows: Sequence[dict[str, Any]], counted: str) -> dict[str, Any]:
    walls = [row["wall_ms"] for row in rows]
    counts = sorted({row[counted] for row in rows})
    return {
        counted: counts[0] if len(counts) == 1 else counts,
        "wall_ms": [round(w, 1) for w in walls],
        "wall_ms_mean": round(statistics.fmean(walls), 1),
        "correct": sum(1 for row in rows if row["correct"]),
        "leaks": sum(row["leaks"] for row in rows),
        "n": len(rows),
    }


async def measure(
    latencies: Sequence[int], runs: int, reply_ms: float, block_ms: float
) -> dict[str, Any]:
    by_replies: dict[str, Any] = {}
    by_branches: dict[str, Any] = {}
    for tool_ms in latencies:
        # Interleaved, so drift in the machine's load lands on both arms alike.
        rows: dict[str, list[dict[str, Any]]] = {style: [] for style in STYLES}
        lanes: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
        by_hand: dict[str, list[dict[str, Any]]] = {"replies": [], "branches": []}
        for _ in range(runs):
            for style in STYLES:
                rows[style].append(await replies(style, tool_ms, reply_ms, block_ms))
            for parallel in (False, True):
                lanes[parallel].append(await branches(parallel, tool_ms, reply_ms, block_ms))
            by_hand["replies"].append(await handwritten_replies(tool_ms, reply_ms, block_ms))
            by_hand["branches"].append(await handwritten_branches(tool_ms, reply_ms, block_ms))
        by_replies[str(tool_ms)] = {
            **{style: _cell(rows[style], "replies") for style in STYLES},
            "saving_ci95": difference_ci(
                [r["wall_ms"] for r in rows["parallel"]], [r["wall_ms"] for r in rows["one_call"]]
            ),
            # What the safety costs: the same replies through the runtime, against by hand.
            "by_hand": _cell(by_hand["replies"], "replies"),
            "vs_by_hand_ci95": difference_ci(
                [r["wall_ms"] for r in rows["parallel"]],
                [r["wall_ms"] for r in by_hand["replies"]],
            ),
            # The same comparison as a price: how much longer the runtime took, as a fraction.
            "cost_vs_by_hand": _cost(rows["parallel"], by_hand["replies"]),
        }
        by_branches[str(tool_ms)] = {
            "one_at_a_time": _cell(lanes[False], "in_flight_max"),
            "side_by_side": _cell(lanes[True], "in_flight_max"),
            "saving_ci95": difference_ci(
                [r["wall_ms"] for r in lanes[True]], [r["wall_ms"] for r in lanes[False]]
            ),
            "summaries_identical": len({r["summary"] for r in lanes[False] + lanes[True]}) == 1,
            "by_hand": _cell(by_hand["branches"], "in_flight_max"),
            "vs_by_hand_ci95": difference_ci(
                [r["wall_ms"] for r in lanes[True]], [r["wall_ms"] for r in by_hand["branches"]]
            ),
            "cost_vs_by_hand": _cost(lanes[True], by_hand["branches"]),
        }
    # Runs through the runtime only: the loop by hand has no branches to leak from, and its
    # runs are the price list, not the product.
    cells = [
        cell
        for group in (*by_replies.values(), *by_branches.values())
        for name, cell in group.items()
        if name != "by_hand" and isinstance(cell, dict) and "n" in cell
    ]
    totals = {
        "runs": sum(cell["n"] for cell in cells),
        "correct": sum(cell["correct"] for cell in cells),
        "leaks": sum(cell["leaks"] for cell in cells),
    }
    return {"replies": by_replies, "branches": by_branches, "totals": totals}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("bench/results/model_bound.json"))
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--reply-ms", type=float, default=REPLY_MS)
    parser.add_argument("--block-ms", type=float, default=BLOCK_MS)
    parser.add_argument("--latencies", default=",".join(str(ms) for ms in TOOL_LATENCIES_MS))
    args = parser.parse_args(argv)
    latencies = [int(ms) for ms in str(args.latencies).split(",") if ms.strip()]

    started = time.monotonic()
    measured = asyncio.run(measure(latencies, args.runs, args.reply_ms, args.block_ms))
    report = {
        "bench": "model_bound",
        "is_real_model": False,
        "stand_in": {"reply_ms": args.reply_ms, "block_ms": args.block_ms},
        "policy": {"speculation": False, "early_issue": True},
        "runs_per_cell": args.runs,
        "tool_latencies_ms": latencies,
        **measured,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"\nMODEL-BOUND (stand-in: {args.reply_ms:.0f} ms per reply, {args.block_ms:.0f} ms/block)"
    )
    for ms in latencies:
        r = report["replies"][str(ms)]
        b = report["branches"][str(ms)]
        print(f"  tools {ms} ms")
        for style in STYLES:
            cell = r[style]
            print(
                f"    {style:9s} {cell['replies']} replies  {cell['wall_ms_mean']:8.0f} ms  "
                f"correct {cell['correct']}/{cell['n']}  leaks {cell['leaks']}"
            )
        print(f"    saving {r['saving_ci95']}")
        for arm in ("one_at_a_time", "side_by_side"):
            cell = b[arm]
            print(
                f"    {arm:13s} {cell['in_flight_max']} in flight  {cell['wall_ms_mean']:8.0f} ms"
                f"  correct {cell['correct']}/{cell['n']}  leaks {cell['leaks']}"
            )
        print(f"    saving {b['saving_ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
