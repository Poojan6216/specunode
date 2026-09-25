"""A node with more than one model turn in flight, one of which fails.

The first fix for a failed ``call_turn`` restored the branch's open-turn count from a snapshot
and discarded every write that appeared on the branch during the turn. Both were wrong as soon
as a node did two things at once, which it may: a background ``complete()``, or a write sent
from another task. The count stuck at one, refusing every later write, or fell to minus one,
turning the guard off; and a write the node had decided elsewhere was thrown away, unsent.
Found by the ninth review; each test below is one of its scripts, made deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseComplete,
    TurnComplete,
    decisions_of,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.testing.models import free_text_turn, tool_turn


def ask(text: str) -> RequestEnvelope:
    return RequestEnvelope(
        model="scripted", messages=(Message(role="user", content=(TextBlock(text=text),)),)
    )


class Model:
    """``complete("draft a note")`` waits to be released; streams asked to be flaky fail."""

    def __init__(self) -> None:
        self.note_open = asyncio.Event()
        self.release_note = asyncio.Event()
        self.release_failure = asyncio.Event()

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        block = envelope.messages[0].content[0]
        assert isinstance(block, TextBlock)
        if "note" in block.text:
            self.note_open.set()
            await self.release_note.wait()
            return free_text_turn("note")
        return tool_turn(("charge_card", {"amount": 25.0}))

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        block = envelope.messages[0].content[0]
        assert isinstance(block, TextBlock)
        if "flaky" in block.text:
            yield TextDelta(index=0, text="Let me ")
            await self.release_failure.wait()
            raise ModelError("overloaded_error (mid-stream)")
        response = tool_turn(("charge_card", {"amount": 25.0}))
        yield ToolUseComplete(index=0, block=response.content[0])  # type: ignore[arg-type]
        await asyncio.sleep(0.01)
        yield TurnComplete(response=response)


async def run(tmp_path: Path, body: object, model: Model) -> tuple[RunResult, list[str], Journal]:
    world: list[str] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(amount: float) -> JsonValue:
        world.append(f"charge {amount}")
        return {"charge_id": "ch_1"}

    @tool(effect="write", idempotent=False)
    async def send_note(text: str) -> JsonValue:
        world.append(f"note {text}")
        return {"sent": True}

    def route(state: Mapping[str, JsonValue]) -> str | None:
        return None if state.get("done") else "act"

    registry = registry_of([charge_card, send_note])
    journal = Journal(tmp_path / "journal.db")
    scheduler = Scheduler(
        graph=PlainAdapter.of([node(name="act")(body)], route),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(model, journal),  # type: ignore[arg-type]
        policy=Policy(speculation=False),
    )
    return await scheduler.run(new_ulid(), {}), world, journal


async def test_a_turn_that_closed_while_another_failed_does_not_block_the_node(
    tmp_path: Path,
) -> None:
    """A background turn is open when a call_turn begins and closes before it fails. Restoring
    the snapshot left one turn counted with none open, and every later write was refused."""
    model = Model()

    async def body(session: RunSession) -> Decision:
        note = asyncio.create_task(session.model.complete(ask("draft a note")))  # type: ignore[union-attr]
        await model.note_open.wait()
        turn = asyncio.create_task(session.call_turn(ask("flaky: bill cus-1")))  # type: ignore[misc]
        await asyncio.sleep(0.01)
        model.release_note.set()
        await note
        model.release_failure.set()
        with contextlib.suppress(ModelError):
            await turn
        decision = decisions_of(await session.model.complete(ask("bill cus-1")))[0]  # type: ignore[union-attr]
        assert isinstance(decision, ToolCall)
        await session.call_tool(decision.name, dict(decision.args))
        session.state["done"] = True
        return decision

    result, world, _journal = await run(tmp_path, body, model)
    assert result.ok, result.error
    assert world == ["charge 25.0"]


async def test_a_turn_that_opened_while_another_failed_keeps_the_guard_on(
    tmp_path: Path,
) -> None:
    """A background turn opens after a call_turn begins and closes after it fails. Restoring
    the snapshot took the count to minus one, and a write made while a later streamed turn was
    still open went out before that turn was journaled."""
    model = Model()

    async def body(session: RunSession) -> Decision:
        turn = asyncio.create_task(session.call_turn(ask("flaky: bill cus-1")))  # type: ignore[misc]
        await asyncio.sleep(0.01)
        note = asyncio.create_task(session.model.complete(ask("draft a note")))  # type: ignore[union-attr]
        await model.note_open.wait()
        model.release_failure.set()
        with contextlib.suppress(ModelError):
            await turn
        model.release_note.set()
        await note
        async for event in session.model.stream(ask("bill cus-1")):  # type: ignore[union-attr]
            if isinstance(event, ToolUseComplete):
                await session.call_tool(event.block.name, dict(event.block.args))
        session.state["done"] = True
        return ToolCall("charge_card", {"amount": 25.0})

    result, world, journal = await run(tmp_path, body, model)
    assert not result.ok and "was not journaled" in (result.error or ""), result.error
    assert world == []
    assert not list(journal.read(result.run_id, kinds=["effect_dispatched"]))


async def test_a_failed_turn_leaves_another_tasks_write_alone(tmp_path: Path) -> None:
    """A write the node decided and sent from another task is not the failed turn's to throw
    away. It was discarded unsent, its task cancelled, and the run ended without a record."""
    model = Model()
    model.release_failure.set()

    async def body(session: RunSession) -> Decision:
        note = asyncio.create_task(session.call_tool("send_note", {"text": "on it"}))
        await asyncio.sleep(0)
        with contextlib.suppress(ModelError):
            await session.call_turn(ask("flaky: bill cus-1"))  # type: ignore[misc]
        await note
        session.state["done"] = True
        return ToolCall("send_note", {"text": "on it"})

    result, world, journal = await run(tmp_path, body, model)
    assert result.ok, result.error
    assert world == ["note on it"]
    assert not list(journal.read(result.run_id, kinds=["effect_discarded"]))
    assert list(journal.read(result.run_id, kinds=["run_finished"]))
