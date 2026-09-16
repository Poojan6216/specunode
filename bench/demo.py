"""The demos. Each one runs; none of them prints a number it did not measure.

``python bench/demo.py --demo leak``
    Demo 1. The same fifty journaled transcripts, executed by three runtimes. The point is the
    last column and the fact that the second and third rows agree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.baselines import ArmResult, Runtime, Transcript, run_arm

from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.testing.world import World, standard_world

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpecuNode demos")
    parser.add_argument("--demo", choices=["leak"], required=True)
    parser.add_argument("--json", action="store_true", help="emit measured values as JSON")
    args = parser.parse_args(argv)
    if args.demo == "leak":
        return asyncio.run(demo_leak(as_json=args.json))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
