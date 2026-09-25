"""A write waits for the model turn that decided it to be journaled (Hard Rule 5).

``call_turn`` stages a turn's writes only after its stream ends, so its writes always wait. A
node that reads ``session.model.stream()`` itself did not: a write it made as a block parsed was
staged, the node parked on it, and the scheduler drained it -- before the turn's response was on
disk. A crash in between left a charge in the world whose decision a resume could not find, and
a model asked again could make a second, different charge. Found by the seventh review.
"""

from __future__ import annotations

from collections.abc import Mapping

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    RequestEnvelope,
    TextBlock,
    ToolUseComplete,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn

ASK = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="bill cus-1"),)),),
    max_tokens=64,
)
TURN = tool_turn(
    ("get_customer", {"customer_id": "cus-1"}),
    ("charge_card", {"customer_id": "cus-1", "amount": 25.0}),
    turn=0,
)


async def run_node(tmp_path: object, body: object) -> tuple[RunResult, list[str], Journal]:
    """Run one node against a streamed turn: a read, then a charge."""
    world: list[str] = []

    @tool(effect="read")
    async def get_customer(customer_id: str) -> JsonValue:
        world.append(f"read {customer_id}")
        return {"id": customer_id}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        world.append(f"charge {customer_id} {amount}")
        return {"charge_id": "ch_1"}

    def route(state: Mapping[str, JsonValue]) -> str | None:
        return None if state.get("done") else "act"

    registry = registry_of([get_customer, charge_card])
    journal = Journal(tmp_path / "journal.db")  # type: ignore[operator]
    scheduler = Scheduler(
        graph=PlainAdapter.of([node(name="act")(body)], route),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(ScriptedModel(turns=[TURN], block_delay_ms=2.0), journal),
        policy=Policy(speculation=False),
    )
    result = await scheduler.run(new_ulid(), {})
    return result, world, journal


async def test_a_write_made_as_its_block_parses_is_refused(tmp_path: object) -> None:
    async def act_on_each_block(session: RunSession) -> Decision:
        async for event in session.model.stream(ASK):
            if isinstance(event, ToolUseComplete):
                await session.call_tool(event.block.name, dict(event.block.args))
        session.state["done"] = True
        return ToolCall("charge_card", {})

    result, world, journal = await run_node(tmp_path, act_on_each_block)
    assert not result.ok
    assert "not yet journaled" in (result.error or ""), result.error
    # The read as its block parsed is fine -- a read is not an effect. The charge is not.
    assert world == ["read cus-1"]
    assert not list(journal.read(result.run_id, kinds=["effect_dispatched"]))


async def test_a_write_after_the_node_stopped_reading_the_turn_is_refused(tmp_path: object) -> None:
    """The rest of the turn is never journaled, so the decision never will be."""

    async def take_the_first_write(session: RunSession) -> Decision:
        call = None
        async for event in session.model.stream(ASK):
            if isinstance(event, ToolUseComplete) and event.block.name == "charge_card":
                call = event.block
                break
        assert call is not None
        await session.call_tool(call.name, dict(call.args))
        session.state["done"] = True
        return ToolCall(call.name, call.args)

    result, world, _journal = await run_node(tmp_path, take_the_first_write)
    assert not result.ok
    assert "not yet journaled" in (result.error or ""), result.error
    assert world == []


async def test_a_write_after_the_turn_is_read_to_its_end_goes_out_after_it_is_journaled(
    tmp_path: object,
) -> None:
    async def read_then_act(session: RunSession) -> Decision:
        blocks = [
            event.block
            async for event in session.model.stream(ASK)
            if isinstance(event, ToolUseComplete)
        ]
        for block in blocks:
            await session.call_tool(block.name, dict(block.args))
        session.state["done"] = True
        return ToolCall(blocks[-1].name, blocks[-1].args)

    result, world, journal = await run_node(tmp_path, read_then_act)
    assert result.ok, result.error
    assert world == ["read cus-1", "charge cus-1 25.0"]
    kinds = [
        entry.kind
        for entry in journal.read(result.run_id, kinds=["model_response", "effect_dispatched"])
    ]
    assert kinds == ["model_response", "effect_dispatched"], kinds
