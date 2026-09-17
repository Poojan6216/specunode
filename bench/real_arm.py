"""The SpecuNode row of Demo 1, executed by the actual runtime.

``bench/baselines.py`` deliberately cannot import ``specunode.buffer``: ``B_naive_parallel``
has to be *able* to leak, or Demo 1 measures nothing, and a test enforces that isolation. That
reasoning is right for the baselines and was quietly wrong for SpecuNode's own row, which was a
six-line ``if transcript.mispredicted:`` conditional describing what a store buffer would do.

So the table at the top of the README -- the project's front page, and the evidence a reader
weighs everything else against -- did not exercise the product. The numbers were not false; they
were simply not measurements of this software. This module runs the same transcripts through the
real ``Scheduler``, ``StoreBuffer`` and ``Dispatcher``, so that row now is.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from bench.baselines import ArmResult, Runtime, Transcript
from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import ToolRegistry
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import JournaledModel, Message, RequestEnvelope, TextBlock
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.drafters.base import DraftContext, Prediction
from specunode.ids import new_ulid
from specunode.journal.journal import Journal, close_all_writers
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World

#: Milliseconds between streamed blocks. Non-zero so the drafter's branch is genuinely in
#: flight when the model's real decision arrives, which is the situation being measured.
#:
#: Wide enough that the speculation reliably reaches ``stage()`` before the confirming block
#: does. Squashing a prediction that had not staged yet is perfectly correct and costs nothing,
#: but it makes the "staged and discarded" count depend on scheduling -- and Demo 1's committed
#: JSON is compared byte-for-byte against a fresh run, so a racy count is a flaky build. At
#: 2 ms roughly one transcript in fifty lost the race.
BLOCK_DELAY_MS = 25.0


class _OneShotDrafter:
    """Predicts one call, once. The transcript says what the drafter got wrong."""

    def __init__(self, decision: ToolCall) -> None:
        self._decision = decision
        self.tier = 1
        self._offered = False

    async def predict(self, ctx: DraftContext) -> list[Prediction]:
        if self._offered:
            return []
        self._offered = True
        return [Prediction(decision=self._decision, tier=1, score=0.9)]


class _OneTurnGraph:
    """A single node that hands one turn to the runtime and returns."""

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(drives_itself=False, framework="plain")

    def nodes(self) -> list[NodeRef]:
        return [NodeRef(name="agent")]

    def decision_kind(self, node: NodeRef) -> str:
        return "tool_call"

    def next(self, state: object) -> NextNode:
        return END if isinstance(state, dict) and state.get("done") else NodeRef(name="agent")

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        assert session.call_turn is not None
        await session.call_turn(
            RequestEnvelope(
                model="scripted",
                messages=(Message(role="user", content=(TextBlock(text="decide"),)),),
                max_tokens=128,
                stream=True,
            )
        )
        session.state["done"] = True
        return ToolCall("", {})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


async def run_specunode_arm(
    transcripts: Sequence[Transcript],
    world: World,
    registry: ToolRegistry,
    *,
    measured_tool: str,
) -> ArmResult:
    """Run every transcript through the real runtime and count what reached the world.

    One run per transcript, each with its own journal, because each transcript is a separate
    agent run. The world is shared so the counts are comparable with the other arms'.
    """
    result = ArmResult(runtime=Runtime.SPECUNODE, runs=len(transcripts))
    retired: set[str] = set()

    with tempfile.TemporaryDirectory() as root:
        for index, transcript in enumerate(transcripts):
            if transcript.mispredicted:
                result.mispredictions += 1

            journal = Journal(Path(root) / f"run-{index:04d}.db")
            # The turn the model really emits: the reads it asked for, then its real decision.
            blocks = [(read.name, dict(read.args)) for read in transcript.reads]
            blocks.append((transcript.actual.name, dict(transcript.actual.args)))
            scheduler = Scheduler(
                graph=_OneTurnGraph(),  # type: ignore[arg-type]
                registry=registry,
                journal=journal,
                buffer=StoreBuffer(journal=journal, run_id=""),
                dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
                target=JournaledModel(
                    ScriptedModel(
                        turns=[tool_turn(*blocks, turn=0)], block_delay_ms=BLOCK_DELAY_MS
                    ),
                    journal,
                    provider="scripted",
                ),
                policy=Policy(speculation=True),
                predictor=_OneShotDrafter(transcript.predicted),  # type: ignore[arg-type]
            )
            run_id = new_ulid()
            outcome = await scheduler.run(run_id, {})
            if not outcome.ok:
                raise SystemExit(f"the specunode arm failed on transcript {index}: {outcome.error}")

            result.speculative_reads += scheduler.counters.speculative_reads_upstream
            result.staged_and_discarded += scheduler.counters.effects_discarded
            for entry in journal.read(run_id, kinds=["branch_resolved"]):
                if entry.payload.get("status") == "retired":
                    retired.add(str(entry.payload["branch_id"]))

    close_all_writers()
    measured = world.mutations_by(measured_tool)
    result.effects_reaching_world = len(measured)
    result.effects_from_squashed = sum(1 for m in measured if m.branch_id not in retired)
    return result
