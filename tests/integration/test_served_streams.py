"""What a resumed or replayed stream hands a node: the pieces it had, where, and when they came.

Found by the eighteenth review. A turn the node gave up on, or that failed, was recorded with
its blocks packed together while its pieces kept the stream's own positions, so after a block
that streams nothing -- thinking -- every piece was served as the wrong block or dropped. Each
piece was timed by when the node took it rather than when the model sent it, so a node busy with
a slow lookup made the model look slow on a resume whose lookup was quick. And ``ask_abandoned``
lost a race to a later answer's own wait, then asked the model on a node already stopped -- after
its run was over, even.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from tests.integration.test_crash_then_resume import (
    ROOMY,
    SMALL,
    Crash,
    CrashingJournal,
    before_commit_of,
    bury_the_dead_process,
    charge,
    replay_of,
    scheduler,
)

from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import RunSession
from specunode.core.model import (
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolUseBlock,
    ToolUseComplete,
    TurnAbandoned,
    TurnComplete,
)
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, free_text_turn

LOOKUP = ToolUseBlock(id="tu_1", name="lookup_plan", args={"customer_id": "cus-1"})


def charging(amount: float) -> ToolUseBlock:
    return ToolUseBlock(
        id="tu_2", name="charge_card", args={"customer_id": "cus-1", "amount": amount}
    )


def plan_tools(sent: list[str]) -> tuple[object, object]:
    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(f"charge {amount}")
        return {"charge_id": f"ch_{len(sent)}"}

    return lookup_plan, charge_card


# -- a block that streams nothing keeps its place ------------------------------------------------


async def test_a_turn_given_up_on_after_a_thinking_block_is_served_where_it_was(
    tmp_path: Path,
) -> None:
    """The model thinks, then asks to look up the plan; the node's deadline fires before the turn
    ends, and it charges a standard 10. Recorded with its blocks packed, the lookup was served on
    resume as a block that did not exist, dropped -- and the fallback moved to a new position,
    under a new key, and went out twice."""
    sent: list[str] = []
    lookup_plan, charge_card = plan_tools(sent)

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            async with asyncio.timeout(0.3):
                await session.call_turn(SMALL)  # type: ignore[misc]
        except TimeoutError:
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    def thinks(amount: float) -> ModelResponse:
        return ModelResponse(
            model="scripted",
            content=(
                ThinkingBlock(text="the plan first", signature="sig"),
                LOOKUP,
                charging(amount),
            ),
            stop_reason="tool_use",
        )

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([lookup_plan, charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            ScriptedModel(turns=[thinks(25.0)], block_delay_ms=120),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 10.0"], "the setup: the deadline fired mid-turn"
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, ScriptedModel(turns=[thinks(99.0)])).resume(
            run_id
        ),
        timeout=30,
    )
    assert resumed.ok, resumed.error
    assert sent == ["charge 10.0"], f"the fallback went out twice: {sent}"


class DropsAfterThinking:
    """Thinks, asks to look up the plan -- and the connection drops at once. Asked again with
    more room, it thinks and charges ``amount``."""

    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(envelope.max_tokens)
        await asyncio.sleep(0.02)
        if envelope.max_tokens == SMALL.max_tokens:
            yield ToolUseComplete(index=1, block=LOOKUP)
            raise ModelError("the stream ended before the model finished its reply")
        use = charging(self.amount)
        yield ToolUseComplete(index=1, block=use)
        yield TurnComplete(
            response=ModelResponse(
                model="scripted",
                content=(ThinkingBlock(text="again", signature="sig"), use),
                stop_reason="tool_use",
            )
        )


def retrying(sent: list[str]) -> tuple[PlainAdapter, object]:
    lookup_plan, charge_card = plan_tools(sent)

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await session.call_turn(SMALL)  # type: ignore[misc]
        except ModelError:  # as documented: give the model room, and ask again
            await session.call_turn(ROOMY)  # type: ignore[misc]
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([lookup_plan, charge_card])


async def test_a_failed_turn_after_a_thinking_block_is_retried_where_it_was(
    tmp_path: Path,
) -> None:
    """The documented pattern -- catch ModelError, ask again with more room -- after a dropped
    connection. The failed turn's lookup was served as nothing, the retry's charge took the
    lookup's position, and a resume that changed nothing charged again."""
    sent: list[str] = []
    adapter, registry = retrying(sent)
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            DropsAfterThinking(25.0),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 25.0"]
    asked_again = DropsAfterThinking(25.0)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_again).resume(run_id), timeout=30
    )
    assert resumed.ok, resumed.error
    assert asked_again.asked == []
    assert sent == ["charge 25.0"], f"charged twice on a resume that changed nothing: {sent}"


async def test_a_replay_of_a_failed_turn_charges_where_the_run_did(tmp_path: Path) -> None:
    """A replay writes nothing before it raises the recorded failure, so the lookup it had just
    handed over was cancelled before its task first ran -- and a position taken only when the
    task ran was never taken: the retry's charge sat one place earlier than the run's. A read
    issued early takes its position as its block arrives."""
    sent: list[str] = []
    adapter, registry = retrying(sent)
    journal = Journal(tmp_path / "source.db")
    run_id = new_ulid()
    result = await scheduler(journal, adapter, registry, DropsAfterThinking(25.0)).run(run_id, {})
    assert result.ok, result.error
    again: list[str] = []
    adapter, registry = retrying(again)
    replayed = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, run_id)
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert replayed.ok, replayed.error
    assert [(r.call.name, r.step_index) for r in replayed.ledger.rows] == [
        (r.call.name, r.step_index) for r in result.ledger.rows
    ]


# -- a piece is timed by when the model sent it ---------------------------------------------------


class Quick:
    """Answers within 60 ms: a word, then a charge of ``amount``."""

    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.asked = 0

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked += 1
        use = charging(self.amount)
        await asyncio.sleep(0.01)
        yield TextDelta(index=0, text="Pricing")
        await asyncio.sleep(0.04)
        yield ToolUseComplete(index=1, block=use)
        yield TurnComplete(
            response=ModelResponse(
                model="scripted", content=(TextBlock(text="Pricing"), use), stop_reason="tool_use"
            )
        )


def reads_between_pieces(slow: dict[str, float], sent: list[str]) -> tuple[PlainAdapter, object]:
    """The node takes the first piece, looks the plan up, then reads the rest -- and a model
    still not done 0.3 s after the lookup is too slow: it charges a standard 10."""
    _quick_lookup, charge_card = plan_tools(sent)

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        await asyncio.sleep(slow["s"])
        return {"plan": "basic"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        loop = asyncio.get_running_loop()
        events = model.stream(SMALL).__aiter__()
        await events.__anext__()
        await session.call_tool("lookup_plan", {"customer_id": "cus-1"})
        began = loop.time()
        reply = None
        async for event in events:
            if isinstance(event, TurnComplete):
                reply = event.response
        assert reply is not None
        in_time = loop.time() - began < 0.3
        amount = float(reply.tool_uses[0].args["amount"]) if in_time else 10.0  # type: ignore[arg-type]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": amount})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([lookup_plan, charge_card])


async def test_a_slow_reader_does_not_make_the_model_look_slow_on_resume(tmp_path: Path) -> None:
    """The model was done in 50 ms; the node took its second piece after a one-second lookup.
    Timed by the node's reads, the piece was recorded as arriving then, and a resume whose
    lookup was quick held it back a second: the model looked slow, and the node charged its
    fallback as well."""
    slow = {"s": 1.0}
    sent: list[str] = []
    adapter, registry = reads_between_pieces(slow, sent)
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, Quick(25.0)
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 25.0"]
    slow["s"] = 0.01
    asked_again = Quick(99.0)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_again).resume(run_id), timeout=30
    )
    assert resumed.ok, resumed.error
    assert asked_again.asked == 0
    assert sent == ["charge 25.0"], f"a second, different charge: {sent}"


async def test_a_slow_reader_does_not_make_the_model_look_slow_in_replay(tmp_path: Path) -> None:
    slow = {"s": 1.0}
    sent: list[str] = []
    adapter, registry = reads_between_pieces(slow, sent)
    journal = Journal(tmp_path / "source.db")
    result = await scheduler(journal, adapter, registry, Quick(25.0)).run(new_ulid(), {})
    assert result.ok and sent == ["charge 25.0"], (result.error, sent)
    slow["s"] = 0.01
    again: list[str] = []
    adapter, registry = reads_between_pieces(slow, again)
    replayed = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert replayed.ok, replayed.error
    assert again == sent, f"the replay charged {again}; the run charged {sent}"


# -- ask_abandoned: not beaten by a later answer, and never on a stopped node --------------------

CACHED = replace(SMALL, max_tokens=32, stream=False)
QUOTE = replace(SMALL, max_tokens=64, stream=False)


class CachedOrQuoted:
    """The recorded run's model never answers the cache check and quotes 25 after 0.3 s; asked
    live, it answers the cache check with 40 at once."""

    def __init__(self, *, recorded: bool) -> None:
        self.recorded = recorded
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.asked.append(envelope.max_tokens)
        if envelope.max_tokens == CACHED.max_tokens:
            if self.recorded:
                await asyncio.Event().wait()
            return charge(40.0)  # type: ignore[return-value]
        await asyncio.sleep(0.3 if self.recorded else 0.0)
        return charge(25.0)  # type: ignore[return-value]

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


async def test_ask_abandoned_is_not_beaten_by_a_later_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later answer's wait for the abandoned turn was taken before that turn's question was
    written, and its own wait after: the later one gave up first, and stopped the node where
    ``ask_abandoned`` would have asked again -- every time. And the turn asked again live says
    so: its question reads ``recorded_from``, and a model call was made all the same."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.5)
    cache = {"hit": True}
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        cached = asyncio.create_task(model.complete(CACHED))
        quoted = asyncio.create_task(model.complete(QUOTE))
        await asyncio.sleep(0.05)
        if cache["hit"]:
            cached.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cached
        else:
            await cached  # the cached price is gone: wait for the question
        answer = await quoted
        amount = float(answer.tool_uses[0].args["amount"])  # type: ignore[arg-type]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": amount})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            CachedOrQuoted(recorded=True),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]
    cache["hit"] = False
    live = CachedOrQuoted(recorded=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, live).resume(run_id, ask_abandoned=True),
        timeout=30,
    )
    assert TurnAbandoned.__name__ not in (resumed.error or ""), resumed.error
    assert live.asked == [CACHED.max_tokens]
    asked_live = [
        e.payload
        for e in Journal(db).read(run_id, kinds=["model_response"])
        if e.payload.get("asked_again")
    ]
    assert len(asked_live) == 1 and "recorded_from" not in asked_live[0], asked_live


class NeverAnswersWhenRecorded:
    """The recorded run's model starts a stream and never finishes it, and never answers a
    question asked whole; asked live, it answers both. ``first`` is set once the stream's first
    piece is out, so the recorded node gives up only after it had that piece -- however fast the
    journal writes."""

    def __init__(self, *, recorded: bool, first: asyncio.Event | None = None) -> None:
        self.recorded = recorded
        self.first = first
        self.asked: list[str] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.asked.append(f"complete {envelope.max_tokens}")
        if self.recorded:
            await asyncio.Event().wait()
        return charge(20.0)  # type: ignore[return-value]

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(f"stream {envelope.max_tokens}")
        await asyncio.sleep(0.01)
        yield TextDelta(index=0, text="Looking")
        if self.first is not None:
            self.first.set()
        if self.recorded:
            await asyncio.Event().wait()
        yield TurnComplete(response=free_text_turn("Looking"))


@pytest.mark.parametrize("ask_abandoned", [True, False])
@pytest.mark.parametrize("gathered", [False, True])
async def test_a_stopped_node_asks_nothing_and_writes_nothing_after_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gathered: bool, ask_abandoned: bool
) -> None:
    """Two questions the recorded node gave up on; on resume it waits for both. The streamed
    one had reached it part-way, so it cannot be asked again and the node is stopped -- and
    the other, still running, was asked live all the same, on the stopped node; gathered, after
    the resume had returned, writing into the journal after ``run_finished``. Found by the
    eighteenth review."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    first = asyncio.Event()
    cache = {"hit": True}
    mode = {"gathered": False}
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        streamed = asyncio.create_task(session.call_turn(SMALL))  # type: ignore[misc]
        cached = asyncio.create_task(model.complete(CACHED))
        await asyncio.sleep(0.05)
        if cache["hit"]:
            await first.wait()  # the stream's first piece had reached the node
            await asyncio.sleep(0.02)
            streamed.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await streamed
            await asyncio.sleep(0.25)
            cached.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cached
            amount = 10.0
        elif mode["gathered"]:
            _results, answer = await asyncio.gather(streamed, cached)
            amount = float(answer.tool_uses[0].args["amount"])  # type: ignore[arg-type]
        else:
            answer = await cached
            await streamed
            amount = float(answer.tool_uses[0].args["amount"])  # type: ignore[arg-type]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": amount})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            NeverAnswersWhenRecorded(recorded=True, first=first),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [10.0]
    cache["hit"] = False
    mode["gathered"] = gathered
    live = NeverAnswersWhenRecorded(recorded=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, live).resume(run_id, ask_abandoned=ask_abandoned),
        timeout=30,
    )
    await asyncio.sleep(1.0)  # anything the node left running
    assert not resumed.ok and TurnAbandoned.__name__ in (resumed.error or ""), resumed.error
    assert live.asked == [], f"the stopped node asked the model live: {live.asked}"
    kinds = [e.kind for e in Journal(db).read(run_id)]
    last = max(i for i, kind in enumerate(kinds) if kind == "run_finished")
    assert kinds[last + 1 :] == [], f"written after the run finished: {kinds[last + 1 :]}"
    assert charged == [10.0]


async def test_a_turn_that_cannot_be_asked_again_does_not_say_to_ask_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part of a streamed turn had reached the node, and the node was told to resume with
    ``--ask-abandoned`` -- which then waited as long again and refused."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    first = asyncio.Event()
    cache = {"hit": True}
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        streamed = asyncio.create_task(session.call_turn(SMALL))  # type: ignore[misc]
        await asyncio.sleep(0.05)
        if cache["hit"]:
            await first.wait()  # the stream's first piece had reached the node
            await asyncio.sleep(0.02)
            streamed.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await streamed
        else:
            await streamed
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            NeverAnswersWhenRecorded(recorded=True, first=first),
        ).run(run_id, {})
    await bury_the_dead_process()
    cache["hit"] = False
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, NeverAnswersWhenRecorded(recorded=False)).resume(
            run_id
        ),
        timeout=30,
    )
    assert not resumed.ok and TurnAbandoned.__name__ in (resumed.error or ""), resumed.error
    assert "--ask-abandoned" not in (resumed.error or ""), resumed.error
    assert "part-way" in (resumed.error or ""), resumed.error
