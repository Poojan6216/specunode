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
from specunode.core.effects import EffectClass, ToolSpec
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


def assert_failed_turn(journal: Journal, run_id: str) -> None:
    """The turn is on disk as the failure it was, with nothing in it recorded as an answer."""
    outcomes = [e.payload for e in journal.read(run_id, kinds=["model_response"])]
    assert outcomes and all(o.get("failed") and o["decision"] is None for o in outcomes), outcomes


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
    assert_failed_turn(journal, result.run_id)
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


async def test_another_tasks_drain_does_not_send_a_guess_whose_turn_is_streaming(
    tmp_path: object,
) -> None:
    """The node sends a note from another task while a turn streams. The model confirms a guess,
    and the note's drain -- which reads the branch's buffer live -- sent the guessed restart
    too, before the turn was journaled; then the turn failed, and it was out. A confirmed
    guess's write joins the branch only once the turn is journaled."""
    world = standard_world()
    registry = registry_for(world)
    registry.register(ToolSpec(name="send_note", effect=EffectClass.WRITE, fn=world.post_summary))
    journal = Journal(tmp_path / "journal.db")  # type: ignore[operator]
    scheduler: Scheduler | None = None

    class ConfirmsThenFails:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            raise NotImplementedError

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            yield ToolUseComplete(
                index=0,
                block=ToolUseBlock(id="t0", name="fetch_runbook", args={"section": "restart"}),
            )
            deadline = time.monotonic() + 10.0
            # The note and the guess are both staged before the model confirms the guess.
            while (  # noqa: ASYNC110
                scheduler is None or scheduler.counters.effects_staged < 2
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            yield ToolUseComplete(
                index=1, block=ToolUseBlock(id="t1", name=RESTART.name, args=dict(RESTART.args))
            )
            # Long enough for the note's drain to run, and send the guess if it can.
            deadline = time.monotonic() + 0.3
            while not world.mutations_by("post_summary") and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.05)
            raise ModelError("overloaded_error (mid-stream)")

    async def note_while_streaming(session: RunSession) -> None:
        note = asyncio.create_task(
            session.call_tool("send_note", {"channel": "ops", "text": "on it"})
        )
        await asyncio.sleep(0)
        with contextlib.suppress(ModelError):
            await session.call_turn(ASK)  # type: ignore[misc]
        await note

    scheduler = Scheduler(
        graph=OneNode(note_while_streaming),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(ConfirmsThenFails(), journal),
        policy=Policy(speculation=True),
        predictor=FixedDrafter(RESTART),  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    assert result.ok, result.error
    assert [m.tool for m in world.mutations] == ["post_summary"], "the guessed restart went out"
    reasons = [e.payload["reason"] for e in journal.read(result.run_id, kinds=["effect_discarded"])]
    assert "turn_failed" in reasons


async def test_a_stream_that_just_stops_is_a_failed_turn(tmp_path: object) -> None:
    """No TurnComplete -- a dropped connection the client did not report. The turn was read as
    complete, and a guess it had confirmed went out with no decision on disk."""

    class JustStops:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            raise NotImplementedError

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            yield ToolUseComplete(
                index=0,
                block=ToolUseBlock(id="t0", name="fetch_runbook", args={"section": "restart"}),
            )
            deadline = time.monotonic() + 10.0
            while (  # noqa: ASYNC110
                scheduler is None or scheduler.counters.effects_staged < 1
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            yield ToolUseComplete(
                index=1, block=ToolUseBlock(id="t1", name=RESTART.name, args=dict(RESTART.args))
            )

    async def tolerant(session: RunSession) -> None:
        with contextlib.suppress(ModelError):
            await session.call_turn(ASK)  # type: ignore[misc]

    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "journal.db")  # type: ignore[operator]
    scheduler: Scheduler | None = None
    scheduler = Scheduler(
        graph=OneNode(tolerant),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(JustStops(), journal),
        policy=Policy(speculation=True),
        predictor=FixedDrafter(RESTART),  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    assert result.ok, result.error
    assert world.mutations_by("restart_job") == [], "a guess went out on a turn never completed"
    assert_failed_turn(journal, result.run_id)


async def test_a_turn_that_fails_on_the_runtimes_side_is_closed_at_once(tmp_path: object) -> None:
    """The drafter raised mid-stream, not the model. The stream was left for the garbage
    collector to close, and until it did, the node's next write was refused."""

    class Broken:
        async def predict(self, context: object) -> list[object]:
            raise KeyError("pattern index corrupt")

    async def write_after_failure(session: RunSession) -> None:
        try:
            await session.call_turn(ASK)  # type: ignore[misc]
        except KeyError:
            await session.call_tool("restart_job", {"job_id": "etl-1"})

    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "journal.db")  # type: ignore[operator]
    scheduler = Scheduler(
        graph=OneNode(write_after_failure),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(FailsPartWay(lambda: True), journal),
        policy=Policy(speculation=True),
        predictor=Broken(),  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    assert result.ok, result.error
    assert [m.tool for m in world.mutations] == ["restart_job"]


async def test_a_guess_whose_adoption_was_not_recorded_is_not_sent(tmp_path: object) -> None:
    """The turn is journaled and confirms a guessed charge, but the record of moving the charge
    onto the node's branch fails to write -- a transient ``JournalBusy``. The move came first,
    so the charge stayed on the branch while the turn failed; the cleanup discarded the guess,
    whose list was already empty, and the node, catching the failure and sending an apology,
    sent the charge with it. The move now waits for its record. Found by the eleventh review."""
    from pathlib import Path

    from examples.support_agent.agent import build_tools
    from tests.chaos._kill_agent import seeded_world

    from specunode.canonical import JsonValue
    from specunode.drafters.base import Prediction
    from specunode.integrations.plain import PlainAdapter, node, registry_of
    from specunode.journal.journal import JournalBusy

    charge = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})
    scheduler: Scheduler | None = None
    caught: list[str] = []

    class GuessesTheCharge:
        async def predict(self, context: object) -> list[Prediction]:
            history = getattr(context, "history", ())
            if history and history[-1].name == "lookup_customer":
                return [Prediction(decision=ToolCall(*charge), tier=1, score=0.9)]
            return []

    class Model:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            raise NotImplementedError

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            from specunode.core.model import TurnComplete

            reply = tool_turn(("lookup_customer", {"customer_id": "cus-1"}), charge)
            yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[arg-type]
            deadline = time.monotonic() + 10.0
            while (  # noqa: ASYNC110
                scheduler is None or scheduler.counters.effects_staged < 1
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            yield ToolUseComplete(index=1, block=reply.content[1])  # type: ignore[arg-type]
            yield TurnComplete(response=reply)

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await session.call_turn(ASK)  # type: ignore[misc]
        except JournalBusy as exc:
            caught.append(str(exc))
            await session.call_tool("send_receipt", {"customer_id": "cus-1", "charge_id": "none"})
        session.state["billed"] = True
        return ToolCall("send_receipt", {})

    world = seeded_world(Path(str(tmp_path)))
    registry = registry_of(build_tools(world))
    journal = Journal(Path(str(tmp_path)) / "journal.db")
    append = journal.append_async
    failed: list[str] = []

    async def busy_once(run_id: str, kind: str, payload: dict[str, JsonValue]) -> int:
        if kind == "effect_adopted" and not failed:
            failed.append(kind)
            raise JournalBusy("database is locked (transient)")
        return await append(run_id, kind, payload)

    journal.append_async = busy_once  # type: ignore[method-assign,assignment]
    scheduler = Scheduler(
        graph=PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill"),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(Model(), journal),  # type: ignore[arg-type]
        policy=Policy(speculation=True),
        predictor=GuessesTheCharge(),  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    world.close()
    assert result.ok, result.error
    assert failed and caught, "the adoption record never failed, so nothing was tested"
    assert [m.tool for m in world.mutations] == ["send_receipt"], "the guessed charge was sent"
    discarded = [e.payload for e in journal.read(result.run_id, kinds=["effect_discarded"])]
    assert [d["reason"] for d in discarded] == ["turn_failed"]


async def test_a_run_whose_confirmed_guess_was_discarded_is_not_reported_in_flight(
    tmp_path: object,
) -> None:
    """The turn fails after confirming a guess, and the node lets the failure end the run. The
    guess's resolution stayed "confirmed", which recovery reads as a drain the process died in:
    a run that sent nothing was reported resumable, with a branch that might have sent
    something. Found by the twelfth review."""
    from specunode.journal.replay import recover

    async def gives_up(session: RunSession) -> None:
        await session.call_turn(ASK)  # type: ignore[misc]

    result, world, journal = await run(tmp_path, gives_up, guess=True)
    assert not result.ok
    assert world.mutations == []
    recovery = recover(journal, result.run_id)
    assert recovery.confirmed_not_retired == () and not recovery.resumable
    resolved = journal.read(result.run_id, kinds=["branch_resolved"])
    resolutions = [(e.payload["status"], e.payload.get("reason")) for e in resolved]
    assert ("squashed", "turn_failed") in resolutions, resolutions


async def test_a_run_cancelled_from_outside_writes_nothing_after_it_returns(
    tmp_path: object,
) -> None:
    """A timeout or a shutdown cancels a run with a guess open. The node bodies were cancelled
    and not waited for, so their own cleanup -- the guess squashed, its write discarded -- was
    journaled after the run returned and let go of the run, where a resume may already be
    writing. Found by the twelfth review."""

    class Hangs:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            raise NotImplementedError

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            yield ToolUseComplete(
                index=0,
                block=ToolUseBlock(id="t0", name="fetch_runbook", args={"section": "restart"}),
            )
            await asyncio.Event().wait()  # the rest of the reply never comes

    async def body(session: RunSession) -> None:
        await session.call_turn(ASK)  # type: ignore[misc]

    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "journal.db")  # type: ignore[operator]
    scheduler = Scheduler(
        graph=OneNode(body),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(Hangs(), journal),
        policy=Policy(speculation=True),
        predictor=FixedDrafter(RESTART),  # type: ignore[arg-type]
    )
    run_id = new_ulid()
    running = asyncio.create_task(scheduler.run(run_id, {}))
    deadline = time.monotonic() + 10.0
    while scheduler.counters.effects_staged < 1 and time.monotonic() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.001)
    running.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await running
    written = journal.last_offset(run_id)
    for _ in range(20):
        await asyncio.sleep(0.005)
    assert journal.last_offset(run_id) == written, "the run wrote after it had returned"
    discarded = list(journal.read(run_id, kinds=["effect_discarded"]))
    assert discarded, "the guess's staged write was never recorded as discarded"
