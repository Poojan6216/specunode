"""Tier-0 early issue (spec task 3.1).

Task 3.1's Verify, stated exactly: with a scripted model streaming three tool calls over
300 ms, the READ among them completes before the stream ends, asserted via timestamps; the
WRITEs stage and retire only after the stream's end-of-turn entry is journaled.

That pair is the whole claim of tier 0. The saving is real -- a read the model emitted first
runs while it is still writing the rest of its answer -- and it costs nothing in safety,
because the writes in the same turn are staged and cannot reach the world until the turn is on
disk and the branch has retired.
"""

from __future__ import annotations

from pathlib import Path

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    RequestEnvelope,
    TextBlock,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

#: A read first, then two writes -- the shape where early issue pays.
TURN = (
    ("get_pipeline_status", {"pipeline_id": "etl-1"}),
    ("restart_job", {"job_id": "etl-1"}),
    ("post_summary", {"channel": "#ops", "text": "restarted"}),
)
#: 100 ms per block, so a 3-block turn streams over ~300 ms as the spec describes.
BLOCK_DELAY_MS = 100.0


def registry_for(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=world.get_pipeline_status,
            witness=True,
        )
    )
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    registry.register(
        ToolSpec(name="post_summary", effect=EffectClass.WRITE, fn=world.post_summary)
    )
    return registry


class OneTurnGraph:
    """A single node that runs one model turn with early issue and stops."""

    def __init__(self) -> None:
        self.results: list[object] = []

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
        envelope = RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text="restart etl-1"),)),),
            max_tokens=256,
            stream=True,
        )
        self.results = list(await session.call_turn(envelope))
        session.state["done"] = True
        return ToolCall("post_summary", {"channel": "#ops", "text": "restarted"})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


def build(tmp_path: Path) -> tuple[Scheduler, World, Journal, str, OneTurnGraph]:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "journal.db")
    graph = OneTurnGraph()
    model = ScriptedModel(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=BLOCK_DELAY_MS)
    run_id = new_ulid()
    scheduler = Scheduler(
        graph=graph,  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=True),
    )
    return scheduler, world, journal, run_id, graph


async def test_the_read_completes_before_the_stream_ends(tmp_path: Path) -> None:
    """The whole point of tier 0, asserted on the clock rather than on the design."""
    scheduler, _world, _journal, run_id, _graph = build(tmp_path)
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    turn = scheduler._turns[0]
    assert turn.reads_issued_early == 1
    assert turn.stream_ended_at > 0.0
    read_finished = turn.completed_at[0]
    assert read_finished > 0.0, "the read never completed"
    assert read_finished < turn.stream_ended_at, (
        f"the read finished at {read_finished:.4f} but the stream only ended at "
        f"{turn.stream_ended_at:.4f}: it was not issued early"
    )


async def test_the_read_really_ran_and_returned_a_value(tmp_path: Path) -> None:
    """A read that was never issued would also trivially finish 'before' the stream ended."""
    scheduler, world, _journal, run_id, graph = build(tmp_path)
    await scheduler.run(run_id, {})
    assert [r.tool for r in world.reads] == ["get_pipeline_status"]
    assert isinstance(graph.results[0], dict) and "witness" in graph.results[0]


async def test_the_writes_reach_the_world_only_after_the_turn_is_durable(
    tmp_path: Path,
) -> None:
    """Hard Rule 3 through the early-issue path: streaming changes when, never whether."""
    scheduler, world, journal, run_id, _graph = build(tmp_path)
    await scheduler.run(run_id, {})

    kinds = [e.kind for e in journal.read(run_id)]
    response_at = kinds.index("model_response")
    for kind in ("effect_staged", "effect_dispatched"):
        assert kinds.index(kind) > response_at, (
            f"{kind} was journaled before the turn that authorised it was durable"
        )
    assert [m.tool for m in world.mutations] == ["restart_job", "post_summary"]


async def test_the_read_is_recorded_as_speculative_and_the_writes_are_not(
    tmp_path: Path,
) -> None:
    """The read went out before the turn was durable, so it is counted as a speculation."""
    scheduler, world, _journal, run_id, _graph = build(tmp_path)
    await scheduler.run(run_id, {})
    assert [r.speculative for r in world.reads] == [True]
    assert all(not m.speculative for m in world.mutations)


async def test_results_come_back_in_program_order(tmp_path: Path) -> None:
    """The model asked for these in this order and must be shown them in it (Hard Rule 13)."""
    scheduler, _world, _journal, run_id, graph = build(tmp_path)
    await scheduler.run(run_id, {})
    assert len(graph.results) == 3
    # The read's result first, then the two staged writes' acks, matching the emitted order.
    assert isinstance(graph.results[0], dict) and "witness" in graph.results[0]
    for ack in graph.results[1:]:
        assert isinstance(ack, dict) and ack.get("ok") is True


async def test_every_emitted_call_is_journaled_once(tmp_path: Path) -> None:
    scheduler, _world, journal, run_id, _graph = build(tmp_path)
    await scheduler.run(run_id, {})
    requests = [e.payload["tool"] for e in journal.read(run_id, kinds=["tool_request"])]
    assert requests == ["get_pipeline_status", "restart_job", "post_summary"]


async def test_a_read_still_in_flight_at_retirement_does_not_demote_the_branch(
    tmp_path: Path,
) -> None:
    """A regression test for a race that early issue reaches by design, not by accident.

    ``_timed_read`` used to mark its read speculative by saving the branch's ``status``,
    setting it to SPECULATIVE, and restoring the saved value in a ``finally``. A read still
    running when the scheduler confirmed the branch therefore put SPECULATIVE back afterwards,
    silently undoing the confirmation; the next drain refused, and the run died with a Hard
    Rule 3 message about a branch that had in fact been confirmed correctly.

    Overlapping a read with the drain is the whole point of early issue, so this was reachable
    on the ordinary path rather than under a fault. The same shape could also have promoted: a
    read that began on a CONFIRMED branch which was then squashed would have restored CONFIRMED
    over the squash, and a squashed branch that can drain is Rule 3 itself.

    **The read has to be the turn's last block.** A read emitted first finishes long before
    retirement even when it is slow, and the race never opens -- which is exactly why the first
    version of this test passed against the unfixed code and had to be rewritten.
    """
    world = standard_world()
    world.slow("read", 300)
    registry = registry_for(world)
    journal = Journal(tmp_path / "in-flight-read.db")
    model = ScriptedModel(
        # Write first, independent read second: the read is issued as the stream's last block
        # and is still running while the write is staged, confirmed and drained.
        turns=[
            tool_turn(
                ("restart_job", {"job_id": "etl-1"}),
                ("get_pipeline_status", {"pipeline_id": "etl-1"}),
                turn=0,
            )
        ],
        block_delay_ms=20.0,
    )
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=True),
    )
    run_id = new_ulid()

    result = await scheduler.run(run_id, {})

    assert result.ok, result.error
    assert len(result.ledger.rows) == 1, "the write must still have reached the world"
    # And the read is still reported as speculative: it did reach upstream before any durable
    # decision authorised it, which is the fact attack 7.2 counts.
    assert [hit for hit in world.reads if hit.speculative], (
        "the early-issued read stopped being reported as speculative"
    )
