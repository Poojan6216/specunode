"""A model turn that fails part-way gives back what it staged, and does not block the node.

``call_turn`` hands a node nothing of a turn that fails: it raises. But a guess the model
confirmed before the failure had already been adopted, and its write moved onto the node's
branch -- so a node that caught the failure and finished drained it, and sent an effect whose
deciding turn was never journaled. And the failed turn stayed open on the branch, so a node that
asked again with ``complete()`` and wrote on that answer was refused. Found by the eighth review.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable

from tests.integration.test_speculation import FixedDrafter, registry_for

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
    ToolUseComplete,
    decisions_of,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import tool_turn
from specunode.testing.world import World, standard_world

ASK = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="etl-1 is failing"),)),),
    max_tokens=128,
    stream=True,
)
RESTART = ToolCall("restart_job", {"job_id": "etl-1"})


class FailsPartWay:
    """Streams a read and then the restart, and fails before the turn ends.

    The restart is held back until ``ready()`` -- until the guess has staged its write -- so
    what is being tested is the failure, not a race with the guess.
    """

    def __init__(self, ready: Callable[[], bool]) -> None:
        self.ready = ready

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return tool_turn(("restart_job", {"job_id": "etl-1"}))

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        yield ToolUseComplete(
            index=0,
            block=ToolUseBlock(id="t0", name="fetch_runbook", args={"section": "restart"}),
        )
        deadline = time.monotonic() + 10.0
        while not self.ready() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.001)
        yield ToolUseComplete(
            index=1, block=ToolUseBlock(id="t1", name=RESTART.name, args=dict(RESTART.args))
        )
        raise ModelError("overloaded_error (mid-stream)")


class OneNode:
    def __init__(self, body: Callable[[RunSession], object]) -> None:
        self.body = body

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(drives_itself=False, framework="plain")

    def nodes(self) -> list[NodeRef]:
        return [NodeRef(name="agent")]

    def decision_kind(self, node: NodeRef) -> str:
        return "tool_call"

    def next(self, state: object) -> NextNode:
        return END if isinstance(state, dict) and state.get("done") else NodeRef(name="agent")

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        await self.body(session)  # type: ignore[misc]
        session.state["done"] = True
        return ToolCall("noop", {})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


async def run(
    tmp_path: object, body: Callable[[RunSession], object], *, guess: bool
) -> tuple[RunResult, World, Journal]:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "journal.db")  # type: ignore[operator]
    scheduler: Scheduler | None = None

    def guess_staged() -> bool:
        return not guess or (scheduler is not None and scheduler.counters.effects_staged > 0)

    scheduler = Scheduler(
        graph=OneNode(body),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(FailsPartWay(guess_staged), journal),
        policy=Policy(speculation=guess),
        predictor=FixedDrafter(RESTART) if guess else None,  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    return result, world, journal


async def test_a_guess_confirmed_before_the_turn_failed_is_not_sent(tmp_path: object) -> None:
    async def tolerant(session: RunSession) -> None:
        with contextlib.suppress(ModelError):  # a failed turn: nothing to do this time
            await session.call_turn(ASK)  # type: ignore[misc]

    result, world, journal = await run(tmp_path, tolerant, guess=True)
    assert result.ok, result.error
    assert [m.tool for m in world.mutations] == [], "a restart went out on a turn never journaled"
    assert not list(journal.read(result.run_id, kinds=["model_response"]))
    reasons = [e.payload["reason"] for e in journal.read(result.run_id, kinds=["effect_discarded"])]
    assert "turn_failed" in reasons


async def test_a_node_that_asks_again_after_a_failed_turn_can_write(tmp_path: object) -> None:
    async def ask_again(session: RunSession) -> None:
        try:
            await session.call_turn(ASK)  # type: ignore[misc]
        except ModelError:
            answer = await session.model.complete(ASK)  # type: ignore[union-attr]
            decision = decisions_of(answer)[0]
            assert isinstance(decision, ToolCall)
            await session.call_tool(decision.name, dict(decision.args))

    result, world, _journal = await run(tmp_path, ask_again, guess=False)
    assert result.ok, result.error
    assert [m.tool for m in world.mutations] == ["restart_job"]
