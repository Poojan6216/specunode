"""The agent loop: fewer model turns, and nothing about correctness traded for them.

When the model is the slow part of a run, the lever infrastructure has is the number of model
turns, and the runtime is built for the reply that cuts it -- several independent calls at once:
reads issued as their blocks parse and run concurrently, writes held until the turn that asked
for them is durable. These tests hold the loop to three properties that decide whether a model
keeps asking for calls that way and whether the run is still the same run:

- every result of a turn goes back in **one** message, paired with the call that asked for it;
- the model's reply goes back **unchanged**, thinking blocks and signatures included;
- whatever the style, the world ends up changed exactly as a correct run changes it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from examples.incident_agent.agent import build, expected_world_changes, scripted_replies

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import (
    JournaledModel,
    ModelResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel
from specunode.testing.world import World, standard_world


async def run(
    tmp_path: Path, style: str, replies: list[ModelResponse] | None = None, *, db: str
) -> tuple[RunResult, World, Journal, str, Scheduler]:
    world = standard_world()
    adapter, registry = build(world, style)
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(turns=replies or scripted_replies(style)), journal, provider="scripted"
        ),
        policy=Policy(speculation=False),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    return result, world, journal, run_id, scheduler


def world_changes(world: World) -> dict[str, object]:
    return {
        "restarted": sorted(m.row_id for m in world.mutations if m.tool == "restart_job"),
        "summaries": sum(1 for m in world.mutations if m.tool == "post_summary"),
    }


def requests(journal: Journal, run_id: str) -> list[dict[str, object]]:
    return [
        dict(entry.payload["request"])  # type: ignore[arg-type]
        for entry in journal.read(run_id, kinds=["model_request"])
    ]


@pytest.mark.parametrize(("style", "turns"), [("one_call", 9), ("parallel", 4)])
async def test_the_same_work_in_fewer_turns_changes_the_world_identically(
    tmp_path: Path, style: str, turns: int
) -> None:
    """Fewer turns is only a gain if the run is still the same run."""
    result, world, journal, run_id, _ = await run(tmp_path, style, db=f"{style}.db")
    assert result.ok, result.error
    assert result.state["turns"] == turns
    assert result.state["calls"] == 8
    assert result.state["stopped"] == "end_turn"
    assert world_changes(world) == expected_world_changes()
    assert [e.kind for e in journal.read(run_id)][-1] == "run_finished"


async def test_a_turns_results_go_back_together_each_naming_its_call(tmp_path: Path) -> None:
    """Split across messages, results teach a model to stop asking for more than one at once."""
    result, _, journal, run_id, _ = await run(tmp_path, "parallel", db="together.db")
    assert result.ok, result.error
    second = requests(journal, run_id)[1]
    messages = second["messages"]
    assert isinstance(messages, list)
    last = messages[-1]
    assert last["role"] == "user"
    blocks = last["content"]
    assert [b["kind"] for b in blocks] == ["tool_result"] * 5, "one message, five results"
    previous = messages[-2]
    assert previous["role"] == "assistant"
    # The journal records a call's id as a positional ``ref`` -- the API's ids are random, and
    # a request hash that included them could never be reproduced -- so the pairing is checked
    # through the refs: each result answers the call in the same position, in order.
    calls = [b["ref"] for b in previous["content"] if b["kind"] == "tool_use"]
    assert len(calls) == 5
    assert [b["ref"] for b in blocks] == calls


async def test_the_models_reply_goes_back_unchanged_thinking_and_signature_included(
    tmp_path: Path,
) -> None:
    """A model that reasons across tool calls loses the thread if its thinking is dropped."""
    thinking = ThinkingBlock(text="", signature="sig-abc123")
    replies = scripted_replies("parallel")
    first = replies[0]
    replies[0] = ModelResponse(
        model=first.model,
        content=(thinking, *first.content),
        stop_reason=first.stop_reason,
        usage=first.usage,
    )
    result, _, journal, run_id, _ = await run(tmp_path, "parallel", replies, db="thinking.db")
    assert result.ok, result.error
    echoed = requests(journal, run_id)[1]["messages"][-2]  # type: ignore[index]
    assert echoed["content"][0] == {  # type: ignore[index]
        "kind": "thinking",
        "text": "",
        "signature": "sig-abc123",
    }


async def test_a_turns_reads_are_issued_as_they_parse_not_after_the_turn(
    tmp_path: Path,
) -> None:
    """The runtime's half of the bargain: a reply asking for five reads runs them together."""
    result, _, _, _, scheduler = await run(tmp_path, "parallel", db="early.db")
    assert result.ok, result.error
    first_turn = scheduler._turns[0]
    assert first_turn.reads_issued_early == 5, "the five reads were not issued as they parsed"


async def test_a_loop_stops_at_max_turns_and_says_so(tmp_path: Path) -> None:
    """A model that never stops asking must not run forever -- or bill forever."""
    from examples.incident_agent import agent as incident

    endless = [
        ModelResponse(
            model="scripted",
            content=(
                ToolUseBlock(id=f"toolu_{i}", name="fetch_runbook", args={"section": "restart"}),
            ),
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        for i in range(incident.MAX_TURNS + 3)
    ]
    result, world, _, _, _ = await run(tmp_path, "parallel", endless, db="endless.db")
    assert result.ok, result.error
    assert result.state["turns"] == incident.MAX_TURNS
    assert result.state["stopped"] == "max_turns"
    assert world.mutations == [], "a loop of reads changed the world"


async def test_a_loop_replays_against_its_own_journal() -> None:
    """Every request the loop builds must be reproducible, or replay refuses the run."""
    from specunode.journal.replay import ReplayModel, recover

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result, _, journal, run_id, _ = await run(root, "parallel", db="recorded.db")
        assert result.ok, result.error
        recovery = recover(journal, run_id)

        world = standard_world()
        adapter, registry = build(world, "parallel")
        into = Journal(root / "replayed.db")
        scheduler = Scheduler(
            graph=adapter,
            registry=registry,
            journal=into,
            buffer=StoreBuffer(journal=into, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
            target=ReplayModel(
                journal=journal, run_id=run_id, retired_branches=recovery.retired_branches
            ),
            policy=Policy(speculation=False),
        )
        replayed = await scheduler.run(new_ulid(), {})
        assert replayed.ok, replayed.error
        assert replayed.state["turns"] == 4
        assert world_changes(world) == expected_world_changes()


def test_the_final_reply_is_prose_and_ends_the_loop() -> None:
    last = scripted_replies("parallel")[-1]
    assert not last.tool_uses and isinstance(last.content[0], TextBlock)
