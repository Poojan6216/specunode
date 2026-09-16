"""The four runtimes the benchmark compares.

Each arm executes the *same* journaled model decisions, so any difference between them is a
difference in the runtime rather than in what the model said.

``B_seq``
    Sequential. No prediction, one call at a time. The reference.
``B_readonly_spec``
    PASTE's policy: speculate on reads, treat every non-read tool as a barrier. Safe, and it
    stops at the first write.
``B_naive_parallel``
    Speculate on everything and execute it for real, discarding the branch on a mismatch. This
    is the failure mode Demo 1 exists to show, and it is what langchain-nvidia's speculative
    mode does with both sides of a conditional edge.
``B_specunode``
    Speculate on everything, but hold writes in a store buffer until the model's real decision
    confirms the branch.

**The baselines cannot reach the store buffer.** ``B_naive_parallel`` in particular must be
able to leak, or Demo 1 measures nothing; a test asserts this module imports nothing from
``specunode.buffer`` so an arm cannot accidentally inherit the protection it is supposed to
lack.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from specunode.canonical import JsonValue
from specunode.core.decision import ToolCall, decisions_equal
from specunode.core.effects import EffectClass, ToolRegistry
from specunode.testing.world import World

__all__ = ["ArmResult", "Runtime", "run_arm"]


class Runtime(Enum):
    SEQUENTIAL = "sequential"
    READONLY_SPEC = "readonly-spec"
    NAIVE_PARALLEL = "naive-parallel"
    SPECUNODE = "specunode"


@dataclass
class ArmResult:
    """What one arm did, counted the same way for every arm."""

    runtime: Runtime
    runs: int = 0
    mispredictions: int = 0
    #: Calls of the measured tool that reached the world, from any branch.
    effects_reaching_world: int = 0
    #: Of those, the ones issued by a branch that never retired. The column that matters.
    effects_from_squashed: int = 0
    staged_and_discarded: int = 0
    speculative_reads: int = 0

    def row(self) -> tuple[str, int, int, int, int]:
        return (
            self.runtime.value,
            self.runs,
            self.mispredictions,
            self.effects_reaching_world,
            self.effects_from_squashed,
        )


@dataclass
class Transcript:
    """One journaled run: what the drafter predicted, and what the model actually decided."""

    predicted: ToolCall
    actual: ToolCall
    reads: Sequence[ToolCall] = field(default_factory=tuple)

    @property
    def mispredicted(self) -> bool:
        return not decisions_equal(self.predicted, self.actual)


async def run_arm(
    runtime: Runtime,
    transcripts: Sequence[Transcript],
    world_factory: Callable[[], World],
    registry: ToolRegistry,
    *,
    measured_tool: str,
    stage: Callable[[World, str, ToolCall, Mapping[str, JsonValue]], Awaitable[None]] | None = None,
) -> tuple[ArmResult, World]:
    """Execute every transcript under one runtime and count what reached the world."""
    world = world_factory()
    result = ArmResult(runtime=runtime, runs=len(transcripts))
    retired: set[str] = set()

    for index, transcript in enumerate(transcripts):
        branch = f"br-{index:04d}"
        if transcript.mispredicted:
            result.mispredictions += 1

        for read in transcript.reads:
            spec = registry.get(read.name)
            if spec.effect is not EffectClass.READ:
                continue
            with world.bind(branch_id=branch, effect_key=f"{branch}-read", speculative=True):
                await spec.fn(**dict(read.args))
            result.speculative_reads += 1

        if runtime is Runtime.SEQUENTIAL or runtime is Runtime.READONLY_SPEC:
            # No write is ever speculated: the real decision is executed, once.
            await _execute(world, branch, transcript.actual, registry)
            retired.add(branch)
            continue

        if runtime is Runtime.NAIVE_PARALLEL:
            # The predicted write runs for real, on a branch that may be about to be thrown
            # away. Nothing can take it back afterwards; that is the whole point.
            speculative_branch = f"{branch}-spec"
            await _execute(world, speculative_branch, transcript.predicted, registry)
            if transcript.mispredicted:
                # The branch is discarded -- and the effect is already in the world.
                await _execute(world, branch, transcript.actual, registry)
                retired.add(branch)
            else:
                retired.add(speculative_branch)
            continue

        # SpecuNode: the predicted write is staged, never executed. It reaches the world only
        # if the model's real decision confirms the branch it belongs to.
        if transcript.mispredicted:
            result.staged_and_discarded += 1
            await _execute(world, branch, transcript.actual, registry)
        else:
            await _execute(world, branch, transcript.predicted, registry)
        retired.add(branch)

    measured = world.mutations_by(measured_tool)
    result.effects_reaching_world = len(measured)
    result.effects_from_squashed = sum(1 for m in measured if m.branch_id not in retired)
    return result, world


async def _execute(world: World, branch: str, call: ToolCall, registry: ToolRegistry) -> None:
    spec = registry.get(call.name)
    with world.bind(branch_id=branch, effect_key=f"{branch}-{call.name}"):
        await spec.fn(**dict(call.args))
