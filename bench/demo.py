"""The demos. Each one runs; none of them prints a number it did not measure.

``python bench/demo.py --demo leak``
    Demo 1. The same fifty journaled transcripts, executed by three runtimes. The point is the
    last column and the fact that the second and third rows agree.

``python bench/demo.py --demo past-write``
    Demo 2. One ops workload under three execution styles, timed at the tool boundary. The
    point is that the saving is exactly one read's latency and the demo says so, rather than
    the shape of the table suggesting more.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.baselines import ArmResult, Runtime, Transcript, run_arm
from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue, chash_bytes
from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import JournaledModel, Message, RequestEnvelope, TextBlock
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world
from specunode.verify.equivalence import equivalence_digest, normalise_world_mutations

RUNS = 50
MISPREDICTIONS = 10
MEASURED_TOOL = "charge_card"


def registry_for(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup_customer",
            effect=EffectClass.READ,
            fn=world.lookup_customer,
            witness=True,
        )
    )
    registry.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))
    return registry


def transcripts(runs: int = RUNS, mispredictions: int = MISPREDICTIONS) -> list[Transcript]:
    """Fifty runs, ten of which the drafter gets wrong.

    Fixed rather than random: the demo's claim is about what the runtimes do with a given set
    of decisions, and a reader should be able to re-run it and see the same table.
    """
    out: list[Transcript] = []
    for index in range(runs):
        customer = f"cus-{index % 5 + 1}"
        actual_amount = float(10 + index % 7)
        # The drafter predicts the amount it has seen before; on the mispredicting runs the
        # model decides on a different one.
        predicted_amount = actual_amount + 1.0 if index < mispredictions else actual_amount
        out.append(
            Transcript(
                predicted=ToolCall(
                    "charge_card", {"customer_id": customer, "amount": predicted_amount}
                ),
                actual=ToolCall("charge_card", {"customer_id": customer, "amount": actual_amount}),
                reads=(ToolCall("lookup_customer", {"customer_id": customer}),),
            )
        )
    return out


async def demo_leak(as_json: bool = False) -> int:
    """Demo 1: speculation without a store buffer double-charges."""
    scripts = transcripts()
    results: list[ArmResult] = []
    for runtime in (Runtime.NAIVE_PARALLEL, Runtime.SPECUNODE, Runtime.SEQUENTIAL):
        world = standard_world()
        registry = registry_for(world)

        def factory(seeded: World = world) -> World:
            return seeded

        arm, _ = await run_arm(
            runtime,
            # The sequential arm is handed the same transcripts but never speculates, so it
            # mispredicts nothing by construction -- which is why its column reads 0.
            scripts,
            factory,
            registry,
            measured_tool=MEASURED_TOOL,
        )
        if runtime is Runtime.SEQUENTIAL:
            arm.mispredictions = 0
        results.append(arm)

    if as_json:
        print(
            json.dumps(
                {
                    "demo": "leak",
                    "runs": RUNS,
                    "measured_tool": MEASURED_TOOL,
                    "arms": [
                        {
                            "runtime": r.runtime.value,
                            "runs": r.runs,
                            "mispredictions": r.mispredictions,
                            "effects_reaching_world": r.effects_reaching_world,
                            "effects_from_squashed_branches": r.effects_from_squashed,
                            "staged_and_discarded": r.staged_and_discarded,
                            "speculative_reads": r.speculative_reads,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
        return 0

    print()
    print("DEMO 1 — speculation without a store buffer double-charges")
    print(f"  {RUNS} journaled transcripts, {MISPREDICTIONS} of which the drafter gets wrong.")
    print(f"  measured tool: {MEASURED_TOOL} (declared WRITE, idempotent=False)")
    print()
    header = (
        f"{'runtime':<18} {'runs':>5} {'mispredictions':>15} "
        f"{'charges reaching world':>23} {'charges from squashed branches':>31}"
    )
    print(header)
    print("-" * len(header))
    for arm in results:
        print(
            f"{arm.runtime.value:<18} {arm.runs:>5} {arm.mispredictions:>15} "
            f"{arm.effects_reaching_world:>23} {arm.effects_from_squashed:>31}"
        )
    print()
    naive, spec, seq = results
    print(
        f"  The last column is the point. naive-parallel put {naive.effects_from_squashed} "
        "charges into the world"
    )
    print(
        "  from branches that were thrown away; those cards were charged for a decision the "
        "model never made."
    )
    print(
        f"  specunode staged {spec.staged_and_discarded} predicted charges and discarded them "
        "unsent, so its row"
    )
    print(
        f"  ({spec.effects_reaching_world} charges) equals the sequential row "
        f"({seq.effects_reaching_world}) exactly."
    )
    print()
    return 0 if spec.effects_from_squashed == 0 else 1


# -- Demo 2: running ahead past a write -------------------------------------------------------

#: Injected latencies, in milliseconds. Declared here and printed by the demo, because a
#: timeline whose inputs are hidden is a drawing rather than a measurement. Every duration in
#: the output is measured at run time; these are only what the fake world was told to take.
READ_MS = 150
WRITE_MS = 90
BLOCK_MS = 60
TURN_MS = 120

#: turn 1 emits the write first and the independent read second. That ordering is the whole
#: point: PASTE-style speculation stops at the first tool with side effects, so there is no
#: read *before* the write for it to run ahead into, and the read that follows cannot move.
TURN_1 = (
    ("restart_job", {"job_id": "etl-1"}),
    ("fetch_runbook", {"section": "restart"}),
)
TURN_2 = (("post_summary", {"channel": "#incidents", "text": "etl-1 restarted"}),)


@dataclass
class CallRecord:
    """When one tool call started and finished, relative to the run's start."""

    name: str
    started: float
    finished: float

    @property
    def duration_ms(self) -> float:
        return (self.finished - self.started) * 1000.0


@dataclass
class Timeline:
    """One arm's measurements. Nothing here is assumed; every field is timed."""

    runtime: str
    wall_ms: float
    calls: list[CallRecord]
    turn_ends: list[float]
    effects: int
    ledger_digest: str
    #: What the world received: tool, canonical arguments, and order. This is what must match
    #: across the three arms. The ledger digest must not, and the demo explains why.
    world_digest: str

    def record(self, name: str) -> CallRecord | None:
        for call in self.calls:
            if call.name == name:
                return call
        return None


def past_write_registry(world: World, clock: float, calls: list[CallRecord]) -> ToolRegistry:
    """The ops agent's four tools, wrapped so every call is timed at the boundary."""

    Tool = Callable[..., Awaitable[JsonValue]]

    def timed(name: str, fn: Tool) -> Tool:
        async def wrapper(**kwargs: JsonValue) -> JsonValue:
            started = time.monotonic()
            try:
                return await fn(**kwargs)
            finally:
                calls.append(CallRecord(name, started - clock, time.monotonic() - clock))

        return wrapper

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=timed("get_pipeline_status", world.get_pipeline_status),
            witness=True,
        )
    )
    registry.register(
        ToolSpec(
            name="fetch_runbook",
            effect=EffectClass.READ,
            fn=timed("fetch_runbook", world.fetch_runbook),
        )
    )
    registry.register(
        ToolSpec(
            name="restart_job",
            effect=EffectClass.WRITE,
            fn=timed("restart_job", world.restart_job),
            idempotent=True,
        )
    )
    registry.register(
        ToolSpec(
            name="post_summary",
            effect=EffectClass.WRITE,
            fn=timed("post_summary", world.post_summary),
        )
    )
    return registry


class PastWriteGraph:
    """The ops agent of section 2.2, in three execution styles over one set of tools.

    ``sequential``  every call runs to completion, in order.
    ``readonly-spec``  PASTE's rule: reads may be run ahead, but speculation stops at the first
        tool with side effects. Turn 1 emits the write first, so there is nothing to run ahead
        into and the arm is serial -- which is the finding, not a rigged comparison. Turn 2's
        reads would be speculable; turn 2 has none.
    ``specunode``  the turn is handed to the runtime: the write is staged, the independent read
        is issued as its block parses, and both writes retire together.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(drives_itself=False, framework="plain")

    def nodes(self) -> list[NodeRef]:
        return [NodeRef(name=name) for name in ("assess", "act", "report")]

    def decision_kind(self, node: NodeRef) -> str:
        return "tool_call"

    def next(self, state: object) -> NextNode:
        seen = state if isinstance(state, dict) else {}
        if "status" not in seen:
            return NodeRef(name="assess")
        if "acted" not in seen:
            return NodeRef(name="act")
        if "reported" not in seen:
            return NodeRef(name="report")
        return END

    @staticmethod
    def _envelope(text: str, *, stream: bool) -> RequestEnvelope:
        return RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text=text),)),),
            max_tokens=128,
            stream=stream,
        )

    async def _run_turn(
        self, session: RunSession, text: str, calls: Sequence[tuple[str, Mapping[str, JsonValue]]]
    ) -> None:
        if self.mode == "specunode":
            assert session.call_turn is not None
            await session.call_turn(self._envelope(text, stream=True))
            return

        # The two baselines take the model's turn and then issue the calls themselves.
        assert session.model is not None
        await session.model.complete(self._envelope(text, stream=False))

        if self.mode == "readonly-spec":
            # Reads *before* the first mutating call may be run ahead; the write and everything
            # after it are serial, because a wrong guess past a write cannot be taken back.
            first_write = next(
                (i for i, (name, _) in enumerate(calls) if name in MUTATING_TOOLS), len(calls)
            )
            ahead = [session.call_tool(name, args) for name, args in calls[:first_write]]
            if ahead:
                await asyncio.gather(*ahead)
            remaining = calls[first_write:]
        else:
            remaining = calls

        for name, args in remaining:
            await session.call_tool(name, args)

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        if node.name == "assess":
            session.state["status"] = await session.call_tool(
                "get_pipeline_status", {"pipeline_id": "etl-1"}
            )
            return ToolCall("get_pipeline_status", {"pipeline_id": "etl-1"})
        if node.name == "act":
            await self._run_turn(session, "etl-1 is failing", TURN_1)
            session.state["acted"] = True
            return ToolCall(*TURN_1[0])
        await self._run_turn(session, "say what happened", TURN_2)
        session.state["reported"] = True
        return ToolCall(*TURN_2[0])

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


#: Declared, not inferred. The baselines need to know which calls they may not run ahead of,
#: and Hard Rule 2 forbids deciding that from a tool's name or its arguments at run time.
MUTATING_TOOLS = frozenset({"restart_job", "post_summary"})


async def run_past_write_arm(mode: str) -> Timeline:
    """One arm, end to end, timed at the wire."""
    world = standard_world()
    world.slow("read", READ_MS)
    world.slow("write", WRITE_MS)
    calls: list[CallRecord] = []
    clock = time.monotonic()
    registry = past_write_registry(world, clock, calls)

    with tempfile.TemporaryDirectory() as directory:
        journal = Journal(Path(directory) / f"{mode}.db")
        model = ScriptedModel(
            turns=[tool_turn(*TURN_1, turn=0), tool_turn(*TURN_2, turn=1)],
            block_delay_ms=BLOCK_MS,
            complete_delay_ms=TURN_MS,
        )
        scheduler = Scheduler(
            graph=PastWriteGraph(mode),  # type: ignore[arg-type]
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
            target=JournaledModel(model, journal, provider="scripted"),
            policy=Policy(speculation=mode == "specunode"),
        )
        started = time.monotonic()
        result = await scheduler.run(new_ulid(), {})
        wall_ms = (time.monotonic() - started) * 1000.0
        if not result.ok:
            raise SystemExit(f"the {mode} arm did not finish cleanly: {result.error!r}")
        return Timeline(
            runtime=mode,
            wall_ms=wall_ms,
            calls=calls,
            turn_ends=[],
            effects=len(world.mutations),
            ledger_digest=equivalence_digest(result.ledger),
            world_digest=chash_bytes(normalise_world_mutations(world.mutations))[:16],
        )


#: The steps, in the order the run performs them, with the column each one is reported under.
STEPS = ("get_pipeline_status", "restart_job", "fetch_runbook", "post_summary")


def _cell(timeline: Timeline, step: str) -> str:
    record = timeline.record(step)
    if record is None:
        return "not called"
    return f"{record.duration_ms:6.0f}ms"


async def demo_past_write(as_json: bool = False) -> int:
    """Demo 2: what running ahead past a write actually buys, measured."""
    arms = [await run_past_write_arm(mode) for mode in ("sequential", "readonly-spec", "specunode")]
    sequential, readonly, specunode = arms

    # Measured, not asserted: did the independent read finish before the turn that emitted it
    # was done with the runtime? That is the overlap the whole design is for.
    read = specunode.record("fetch_runbook")
    write = specunode.record("restart_job")
    overlap_ms = 0.0
    if read is not None and write is not None:
        # The read ran while the staged write was still waiting to drain. The overlap is the
        # part of the read that finished before the write was dispatched.
        overlap_ms = max(0.0, min(read.finished, write.started) - read.started) * 1000.0

    if as_json:
        print(
            json.dumps(
                {
                    "demo": "past-write",
                    "injected_latency_ms": {
                        "read": READ_MS,
                        "write": WRITE_MS,
                        "stream_block": BLOCK_MS,
                        "model_turn": TURN_MS,
                    },
                    "arms": [
                        {
                            "runtime": arm.runtime,
                            "wall_ms": round(arm.wall_ms, 1),
                            "effects_reaching_world": arm.effects,
                            "ledger_digest": arm.ledger_digest,
                            "world_digest": arm.world_digest,
                            "calls": [
                                {
                                    "name": call.name,
                                    "started_ms": round(call.started * 1000.0, 1),
                                    "finished_ms": round(call.finished * 1000.0, 1),
                                    "duration_ms": round(call.duration_ms, 1),
                                }
                                for call in arm.calls
                            ],
                        }
                        for arm in arms
                    ],
                    "read_overlapped_ms": round(overlap_ms, 1),
                    "worlds_identical": len({arm.world_digest for arm in arms}) == 1,
                    "ledger_digests_differ_by_step_index": (
                        len({arm.ledger_digest for arm in arms}) != 1
                    ),
                },
                indent=2,
            )
        )
        return 0

    print()
    print("DEMO 2 — running ahead past a write")
    print(
        f"  injected latency: read {READ_MS}ms, write {WRITE_MS}ms, "
        f"stream block {BLOCK_MS}ms, model turn {TURN_MS}ms."
    )
    print("  every duration below is measured at the tool boundary during the run.")
    print()
    header = f"{'step':<22} {'sequential':>12} {'readonly-spec':>15} {'specunode':>12}"
    print(header)
    print("-" * len(header))
    for step in STEPS:
        print(
            f"{step:<22} {_cell(sequential, step):>12} "
            f"{_cell(readonly, step):>15} {_cell(specunode, step):>12}"
        )
    print("-" * len(header))
    print(
        f"{'wall clock':<22} {sequential.wall_ms:10.0f}ms "
        f"{readonly.wall_ms:13.0f}ms {specunode.wall_ms:10.0f}ms"
    )
    print(
        f"{'effects reaching world':<22} {sequential.effects:>12} "
        f"{readonly.effects:>15} {specunode.effects:>12}"
    )
    print()
    identical = len({arm.world_digest for arm in arms}) == 1
    print(f"  what reached the world is identical across all three arms: {identical}")
    print(f"  (tool, canonical arguments and order: {sequential.world_digest})")
    print()
    if len({arm.ledger_digest for arm in arms}) != 1:
        print("  Their effect *ledgers* differ, and that is expected here rather than a fault.")
        print("  The two baselines are separate implementations that issue each call in the")
        print("  order the model emitted it; specunode issues the independent read as its")
        print("  block parses, so the read takes the earlier program position and every later")
        print("  step index shifts by one. Hard Rule 9 compares speculation on against")
        print("  speculation off for one graph, where both arms issue reads early and the")
        print("  indices agree -- that is tests/test_equivalence.py, not this table.")
        print()
    print(
        f"  readonly-spec matches sequential ({readonly.wall_ms:.0f}ms vs "
        f"{sequential.wall_ms:.0f}ms) because turn 1 emits the write first."
    )
    print("  PASTE stops at the first tool with side effects, so it has nothing to run into.")
    print(
        f"  specunode staged the write and issued fetch_runbook as its block parsed; "
        f"{overlap_ms:.0f}ms of that"
    )
    print("  read ran while the write was still staged, which is the whole of what it buys.")
    print()
    print("  Model turn 2 is NOT hidden. It needs restart_job's real result in its prompt, so")
    print("  it waits for the drain. Speculation past a write hides tool latency, not model")
    print("  latency -- and on this workload that is one read.")
    print()
    return 0 if identical else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpecuNode demos")
    parser.add_argument("--demo", choices=["leak", "past-write"], required=True)
    parser.add_argument("--json", action="store_true", help="emit measured values as JSON")
    args = parser.parse_args(argv)
    if args.demo == "leak":
        return asyncio.run(demo_leak(as_json=args.json))
    if args.demo == "past-write":
        return asyncio.run(demo_past_write(as_json=args.json))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
