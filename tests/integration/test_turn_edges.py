"""A turn at its edges: refused at once, cut off, stopped mid-write, outliving its run.

Found by the nineteenth review. A client whose ``stream()`` refused before returning anything
left its question with no outcome, and a resume asked the model again. A reply refused as cut
off was timed by when the node got to it, not when it came. A question written while its node
was being stopped went to the live model. A call left running wrote into the journal after
``run_finished`` -- while it was being written, or into a resume of the same run in the same
process, or as the first question of its drive. And a turn that failed after a guessed block
left the next call at another position with speculation on than off.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
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
from tests.integration.test_speculation import FixedDrafter, registry_for

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
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
    TextDelta,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.testing.world import standard_world


def charging(amount: float) -> ToolUseBlock:
    return ToolUseBlock(
        id="tu_1", name="charge_card", args={"customer_id": "cus-1", "amount": amount}
    )


# -- a stream refused at once ---------------------------------------------------------------------


class RefusesAtOnceWhenBusy:
    """``stream`` is a plain method, as the protocol declares it: while the client's own rate
    limiter is closed it raises at once, before any iterator exists."""

    def __init__(self, amount: float, *, busy: bool, error: type[Exception] = ModelError) -> None:
        self.amount = amount
        self.busy = busy
        self.error = error
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(envelope.max_tokens)
        if self.busy and envelope.max_tokens == SMALL.max_tokens:
            raise self.error("rate limited: try again with a larger budget")
        return self._stream()

    async def _stream(self) -> AsyncIterator[StreamEvent]:
        use = charging(self.amount)
        await asyncio.sleep(0.01)
        yield ToolUseComplete(index=0, block=use)
        yield TurnComplete(
            response=ModelResponse(model="scripted", content=(use,), stop_reason="tool_use")
        )


def asks_again() -> tuple[PlainAdapter, object, list[float]]:
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await session.call_turn(SMALL)  # type: ignore[misc]
        except ModelError:  # as documented: give the model room, and ask again
            await session.call_turn(ROOMY)  # type: ignore[misc]
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card]), charged


async def test_a_stream_refused_at_once_is_recorded_and_served_again(tmp_path: Path) -> None:
    db = tmp_path / "source.db"
    adapter, registry, charged = asks_again()
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            RefusesAtOnceWhenBusy(25.0, busy=True),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]
    journal = Journal(db)
    answered = {e.payload["request_id"] for e in journal.read(run_id, kinds=["model_response"])}
    asked = [e.payload["request_id"] for e in journal.read(run_id, kinds=["model_request"])]
    assert all(request in answered for request in asked), "the refused question has no outcome"

    not_busy = RefusesAtOnceWhenBusy(30.0, busy=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, not_busy).resume(run_id), timeout=30
    )
    assert resumed.ok, resumed.error
    assert not_busy.asked == [], "the resume asked the model what the journal answers"
    assert charged == [25.0], f"a second, different charge went out: {charged}"


async def test_a_stream_refused_at_once_reaches_the_node_as_a_model_error(tmp_path: Path) -> None:
    adapter, registry, charged = asks_again()
    result = await scheduler(
        Journal(tmp_path / "j.db"),
        adapter,
        registry,
        RefusesAtOnceWhenBusy(25.0, busy=True, error=ConnectionError),
    ).run(new_ulid(), {})
    assert result.ok, result.error
    assert charged == [25.0]


async def test_a_replay_of_a_stream_refused_at_once(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "source.db")
    adapter, registry, _charged = asks_again()
    result = await scheduler(
        journal, adapter, registry, RefusesAtOnceWhenBusy(25.0, busy=True)
    ).run(new_ulid(), {})
    assert result.ok, result.error
    adapter, registry, again = asks_again()
    replayed = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert replayed.ok, replayed.error
    assert again == [25.0]


# -- a reply refused as cut off is timed by when it came ------------------------------------------


class QuickButCutOff:
    """Streams a word at 10 ms and, at 50 ms, a reply cut off mid-call. Asked again with room
    (``complete``), it quotes 25."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return ModelResponse(model="scripted", content=(charging(25.0),), stop_reason="tool_use")

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(0.01)
        yield TextDelta(index=0, text="Pricing")
        await asyncio.sleep(0.04)
        yield TurnComplete(
            response=ModelResponse(
                model="scripted",
                content=(TextBlock(text="Pricing"), charging(2.0)),
                stop_reason="max_tokens",
            )
        )


def quote_then_charge(slow: dict[str, float], sent: list[str]) -> tuple[PlainAdapter, object]:
    """The quote takes the first piece, looks the plan up, then reads the rest: a model that
    answers -- even cut off -- within 0.3 s of the lookup is asked again with room, and a slower
    one gets the standard 10."""

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        await asyncio.sleep(slow["s"])
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(f"charge {amount}")
        return {"charge_id": f"ch_{len(sent)}"}

    @node(name="quote")
    async def quote(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        loop = asyncio.get_running_loop()
        events = model.stream(SMALL).__aiter__()
        await events.__anext__()
        await session.call_tool("lookup_plan", {"customer_id": "cus-1"})
        began = loop.time()
        amount = 99.0
        try:
            async for _event in events:
                pass
        except ModelError:
            if loop.time() - began < 0.3:
                answer = await model.complete(replace(ROOMY, stream=False))
                amount = float(answer.tool_uses[0].args["amount"])  # type: ignore[arg-type]
            else:
                amount = 10.0
        session.state["amount"] = amount
        return ToolCall("charge_card", {})

    @node(name="charge")
    async def pay(session: RunSession) -> Decision:
        amount = float(session.state["amount"])  # type: ignore[arg-type]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": amount})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    def route(state: Mapping[str, JsonValue]) -> str | None:
        if state.get("billed"):
            return None
        return "charge" if "amount" in state else "quote"

    adapter = PlainAdapter.of([quote, pay], route)
    return adapter, registry_of([lookup_plan, charge_card])


async def test_a_cut_off_reply_is_replayed_when_it_arrived(tmp_path: Path) -> None:
    """The reply was refused as cut off at 50 ms, and the node, busy with a one-second lookup,
    got to the refusal after it. Timed by the node, a replay whose lookup was quick waited that
    whole second for the refusal, found the model too slow, and charged its fallback."""
    slow = {"s": 1.0}
    sent: list[str] = []
    adapter, registry = quote_then_charge(slow, sent)
    journal = Journal(tmp_path / "source.db")
    result = await scheduler(journal, adapter, registry, QuickButCutOff()).run(new_ulid(), {})
    assert result.ok and sent == ["charge 25.0"], (result.error, sent)
    slow["s"] = 0.01
    again: list[str] = []
    adapter, registry = quote_then_charge(slow, again)
    replayed = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert replayed.ok, replayed.error
    assert again == sent, f"the replay charged {again}; the run charged {sent}"


# -- a question written while its node is stopped --------------------------------------------------

CACHED = replace(SMALL, max_tokens=32, stream=False)
QUOTE = replace(SMALL, max_tokens=64, stream=False)
SECOND_OPINION = replace(SMALL, max_tokens=128, stream=False)


class NotesWhenAsked:
    def __init__(self, *, recorded: bool) -> None:
        self.recorded = recorded
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.asked.append(envelope.max_tokens)
        if envelope.max_tokens == CACHED.max_tokens and self.recorded:
            await asyncio.Event().wait()
        return charge(25.0)  # type: ignore[return-value]

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


class SlowSecondOpinion(Journal):
    """The second-opinion question takes 0.3 s to reach the disk."""

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        request = payload.get("request")
        if kind == "model_request" and isinstance(request, Mapping):
            params = request.get("params")
            if isinstance(params, Mapping) and params.get("max_tokens") == 128:
                await asyncio.sleep(0.3)
        return await super().append_async(run_id, kind, payload)


async def test_a_question_written_while_its_node_is_stopped_is_not_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.3)
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
        await asyncio.sleep(0.05)
        if cache["hit"]:
            cached.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cached
            answer = await model.complete(QUOTE)
        else:
            # The cache is gone: while waiting, ask for a second opinion too.
            await asyncio.sleep(0.3)
            opinion = asyncio.create_task(model.complete(SECOND_OPINION))
            answer = await cached
            await opinion
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
            NotesWhenAsked(recorded=True),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]
    cache["hit"] = False
    live = NotesWhenAsked(recorded=False)
    resumed = await asyncio.wait_for(
        scheduler(SlowSecondOpinion(db), adapter, registry, live).resume(run_id), timeout=30
    )
    await asyncio.sleep(0.5)
    assert not resumed.ok and "TurnAbandoned" in (resumed.error or ""), resumed.error
    assert live.asked == [], f"the stopped node asked the model live: {live.asked}"


# -- a call that outlives its drive writes nothing after it --------------------------------------


class SlowToFinish(Journal):
    """``run_finished`` is handed to the writer, and the disk is slow to take it."""

    def __init__(self, path: Path, finishing: asyncio.Event) -> None:
        super().__init__(path)
        self.finishing = finishing

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind != "run_finished":
            return await super().append_async(run_id, kind, payload)
        writing = asyncio.ensure_future(super().append_async(run_id, kind, payload))
        await asyncio.sleep(0)
        self.finishing.set()
        await asyncio.sleep(0.05)
        return await writing


class AnswersWhen:
    def __init__(self, go: asyncio.Event) -> None:
        self.go = go
        self.asked = 0

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.asked += 1
        await self.go.wait()
        return charge(99.0)  # type: ignore[return-value]

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


def after_the_first_end(db: Path, run_id: str) -> list[str]:
    kinds = [e.kind for e in Journal(db).read(run_id)]
    return kinds[kinds.index("run_finished") + 1 :]


async def test_an_answer_landing_while_run_finished_is_written_is_not_written(
    tmp_path: Path,
) -> None:
    finishing = asyncio.Event()

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        asyncio.get_running_loop().create_task(model.complete(SMALL))  # never awaited
        await asyncio.sleep(0.05)
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    run_id = new_ulid()
    result = await scheduler(
        SlowToFinish(tmp_path / "j.db", finishing), adapter, registry_of([]), AnswersWhen(finishing)
    ).run(run_id, {})
    await asyncio.sleep(0.5)
    assert result.ok, result.error
    assert after_the_first_end(tmp_path / "j.db", run_id) == []


async def test_a_call_left_over_from_one_drive_writes_nothing_into_the_next(
    tmp_path: Path,
) -> None:
    go = asyncio.Event()
    mode = {"first": True}

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        if mode["first"]:
            asyncio.get_running_loop().create_task(model.complete(SMALL))  # never awaited
            await asyncio.sleep(0.05)
            raise RuntimeError("the node failed with a question still out")
        go.set()  # the resumed node runs while the old question's answer comes back
        await asyncio.sleep(0.2)
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    db = tmp_path / "j.db"
    run_id = new_ulid()
    first = await scheduler(Journal(db), adapter, registry_of([]), AnswersWhen(go)).run(run_id, {})
    assert not first.ok
    mode["first"] = False
    await scheduler(Journal(db), adapter, registry_of([]), AnswersWhen(go)).resume(run_id)
    await asyncio.sleep(0.3)
    later = [
        kind
        for kind in after_the_first_end(db, run_id)
        if kind in ("model_request", "model_response")
    ]
    assert later == [], f"the first drive's call wrote into the resume: {later}"


async def test_a_first_question_asked_after_the_run_is_over_is_refused(tmp_path: Path) -> None:
    model = AnswersWhen(asyncio.Event())
    model.go.set()
    over = asyncio.Event()

    async def later(asking: object) -> None:
        await over.wait()  # asked once the run has returned, however slow its disk
        with contextlib.suppress(BaseException):
            await asking.complete(SMALL)  # type: ignore[attr-defined]

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        asyncio.get_running_loop().create_task(later(session.model))  # never awaited
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    db = tmp_path / "j.db"
    run_id = new_ulid()
    result = await scheduler(Journal(db), adapter, registry_of([]), model).run(run_id, {})
    over.set()
    await asyncio.sleep(0.3)
    assert result.ok, result.error
    assert model.asked == 0, "the model was asked live after the run was over"
    assert after_the_first_end(db, run_id) == []


# -- a failed turn leaves the next call where it would be, speculation on or off -----------------

TURN = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="go"),)),),
    max_tokens=128,
    stream=True,
)


class ReadsThenDrops:
    """Asks for the pipeline's status, then -- late enough for a guess at it to be under way --
    for the runbook, and the connection drops."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(0.01)
        yield ToolUseComplete(
            index=0,
            block=ToolUseBlock(id="t0", name="get_pipeline_status", args={"pipeline_id": "etl"}),
        )
        await asyncio.sleep(0.3)
        yield ToolUseComplete(
            index=1,
            block=ToolUseBlock(id="t1", name="fetch_runbook", args={"section": "restart"}),
        )
        raise ModelError("the stream ended before the model finished its reply")


class RestartsAfterAFailedTurn:
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
        with contextlib.suppress(ModelError):
            await session.call_turn(TURN)
        await session.call_tool("restart_job", {"job_id": "etl-1"})
        session.state["done"] = True
        return ToolCall("restart_job", {"job_id": "etl-1"})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


async def restart_step(tmp_path: Path, *, speculation: bool) -> int:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / f"spec-{speculation}.db")
    drafter = FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}))
    scheduler = Scheduler(
        graph=RestartsAfterAFailedTurn(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(ReadsThenDrops(), journal, provider="scripted"),
        policy=Policy(speculation=speculation, max_speculation_depth=3),
        predictor=drafter if speculation else None,  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    assert result.ok, result.error
    if speculation:
        assert sum(turn.adopted for turn in scheduler._turns) == 1, (
            "the setup: the guessed read was adopted"
        )
    (row,) = result.ledger.rows
    return row.step_index


async def test_a_failed_turn_leaves_the_next_call_where_it_is_without_speculation(
    tmp_path: Path,
) -> None:
    """The guessed read was adopted and never took its position on the node's branch, while the
    same read issued early without speculation did: after the turn failed, the node's restart
    sat one position earlier with speculation on, under another key. Suspected by the
    nineteenth review."""
    assert await restart_step(tmp_path, speculation=True) == await restart_step(
        tmp_path, speculation=False
    )


# -- a drive that could not hold its run is not over ----------------------------------------------


async def test_a_resume_retried_after_run_busy_can_ask_the_model(tmp_path: Path) -> None:
    """A Scheduler that could not hold the run "drove nothing, and may try again" -- but it had
    ended its drive, and on the retry every question it asked was refused as after the run.
    Found by the twentieth review."""
    from tests.integration.test_crash_then_resume import build_billing

    from specunode.journal.journal import RunBusy
    from specunode.testing.models import ScriptedModel
    from specunode.testing.world import standard_world

    db = tmp_path / "source.db"
    world = standard_world()
    adapter, registry = build_billing(world)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            ScriptedModel(turns=[charge(25.0)]),  # type: ignore[list-item]
        ).run(run_id, {})
    await bury_the_dead_process()
    adapter, registry = build_billing(world)
    resumer = scheduler(Journal(db), adapter, registry, ScriptedModel(turns=[charge(25.0)]))  # type: ignore[list-item]
    with Journal(db).hold_run(run_id), pytest.raises(RunBusy):
        await resumer.resume(run_id)
    resumed = await asyncio.wait_for(resumer.resume(run_id), timeout=30)
    assert resumed.ok, resumed.error
    assert [row["amount"] for row in world.tables["charges"].values()] == [25.0]


# -- nothing is asked once a drive is over or a node stopped, however the loop orders it ---------

LEFTOVER = replace(SMALL, max_tokens=77, stream=True)


class Notes:
    """Records, when it is asked, whether its caller's drive was over or its node stopped."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        from specunode.core.model import current_scope, drive_over

        scope = current_scope()
        stopped = scope.halted is not None and scope.halted()
        self.asked.append(f"drive over={drive_over(scope.drive)}, node stopped={stopped}")
        return self._answer()

    async def _answer(self) -> AsyncIterator[StreamEvent]:
        yield TurnComplete(response=charge(1.0))  # type: ignore[arg-type]


class FinishesAsTheQuestionLands(Journal):
    """The leftover question lands on disk just as the Scheduler goes to write run_finished."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.in_check = asyncio.Event()
        self.finishing = asyncio.Event()

    async def check_run_lock(self, run_id: str) -> None:
        await super().check_run_lock(run_id)
        self.in_check.set()
        await self.finishing.wait()

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        request = payload.get("request")
        params = request.get("params") if isinstance(request, Mapping) else None
        leftover = (
            kind == "model_request"
            and isinstance(params, Mapping)
            and params.get("max_tokens") == LEFTOVER.max_tokens
        )
        if leftover:
            await self.in_check.wait()
        offset = await super().append_async(run_id, kind, payload)
        if leftover:
            loop = asyncio.get_running_loop()
            loop.call_soon(lambda: loop.call_soon(self.finishing.set))
        return offset


async def test_the_model_is_not_asked_once_the_drive_is_over(tmp_path: Path) -> None:
    """Looked at before the stream was opened, and opened a step of the event loop later: a
    drive that ended in that step had its question asked all the same. Found by the twentieth
    review."""
    model = Notes()

    async def read(stream: AsyncIterator[StreamEvent]) -> None:
        with contextlib.suppress(BaseException):
            async for _event in stream:
                pass

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        assert session.model is not None
        asyncio.get_running_loop().create_task(
            read(session.model.stream(LEFTOVER))
        )  # never awaited
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    db = tmp_path / "j.db"
    run_id = new_ulid()
    result = await asyncio.wait_for(
        scheduler(FinishesAsTheQuestionLands(db), adapter, registry_of([]), model).run(run_id, {}),
        timeout=30,
    )
    await asyncio.sleep(0.3)
    assert result.ok, result.error
    assert model.asked == [], model.asked


class HaltsAsTheQuestionLands(Journal):
    """The node's question lands on disk, and another task of the node stops it one step of the
    event loop later -- as a served turn given up on does."""

    def __init__(self, path: Path, halt_now: asyncio.Event) -> None:
        super().__init__(path)
        self.halt_now = halt_now

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        request = payload.get("request")
        params = request.get("params") if isinstance(request, Mapping) else None
        second = (
            kind == "model_request"
            and isinstance(params, Mapping)
            and params.get("max_tokens") == LEFTOVER.max_tokens
        )
        offset = await super().append_async(run_id, kind, payload)
        if second:
            loop = asyncio.get_running_loop()
            loop.call_soon(lambda: loop.call_soon(self.halt_now.set))
        return offset


async def test_the_model_is_not_asked_for_a_node_stopped_as_its_stream_opens(
    tmp_path: Path,
) -> None:
    from specunode.core.model import current_scope

    model = Notes()
    halt_now = asyncio.Event()

    async def stopper() -> None:
        scope = current_scope()
        await halt_now.wait()
        assert scope.halt is not None
        scope.halt("a served turn of this node was abandoned")

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        assert session.model is not None
        asyncio.get_running_loop().create_task(stopper())
        with contextlib.suppress(BaseException):
            async for _event in session.model.stream(LEFTOVER):
                pass
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    db = tmp_path / "j.db"
    run_id = new_ulid()
    await asyncio.wait_for(
        scheduler(HaltsAsTheQuestionLands(db, halt_now), adapter, registry_of([]), model).run(
            run_id, {}
        ),
        timeout=30,
    )
    assert model.asked == [], model.asked
    outcomes = [e.payload for e in Journal(db).read(run_id, kinds=["model_response"])]
    assert [o.get("cancelled") for o in outcomes] == [True], "the question has no outcome"


# -- nor by a drafter's model, nor through a tool ------------------------------------------------

NOW = replace(SMALL, max_tokens=100, stream=True)
LATER = replace(SMALL, max_tokens=200, stream=True)


class PlansAtOnce:
    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        if envelope.max_tokens == LATER.max_tokens:
            await asyncio.sleep(0.3)  # long after the run is over
        use = ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"})
        yield ToolUseComplete(index=0, block=use)
        yield TurnComplete(
            response=ModelResponse(model="scripted", content=(use,), stop_reason="tool_use")
        )


class Draft:
    def __init__(self) -> None:
        self.asked = 0

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.asked += 1
        use = ToolUseBlock(id="d0", name="lookup_plan", args={"customer_id": "cus-2"})
        return ModelResponse(model="draft", content=(use,), stop_reason="tool_use")

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


async def test_a_drafter_is_not_asked_after_the_run(tmp_path: Path) -> None:
    """Only the target model was told the drive was over: a drafter's model was asked after
    the run, and wrote into the journal after ``run_finished``. Found by the twentieth review."""
    from specunode.drafters.t2_model import ModelDrafter

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @node(name="plan")
    async def plan(session: RunSession) -> Decision:
        assert session.call_turn is not None
        await session.call_turn(NOW)  # type: ignore[misc]
        asyncio.get_running_loop().create_task(session.call_turn(LATER))  # never awaited
        session.state["done"] = True
        return ToolCall("lookup_plan", {})

    adapter = PlainAdapter.of([plan], lambda s: None if s.get("done") else "plan")
    registry = registry_of([lookup_plan])
    db = tmp_path / "j.db"
    journal = Journal(db)
    draft = Draft()
    driver = Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
        target=JournaledModel(PlansAtOnce(), journal, provider="scripted"),
        policy=Policy(speculation=True, max_speculation_depth=3),
        predictor=ModelDrafter(  # type: ignore[arg-type]
            client=JournaledModel(draft, journal, role="draft", provider="scripted"),
            tools=(),
        ),
    )
    run_id = new_ulid()
    result = await asyncio.wait_for(driver.run(run_id, {}), timeout=30)
    asked_in_run = draft.asked
    await asyncio.sleep(1.0)
    assert result.ok, result.error
    assert draft.asked == asked_in_run, "the draft model was asked after the run was over"
    assert after_the_first_end(db, run_id) == []


async def test_a_tool_is_not_called_after_the_run(tmp_path: Path) -> None:
    """A task the node never awaited reached the upstream after the run, and wrote its call and
    result into the journal after ``run_finished``. Found by the twentieth review."""
    reached: list[str] = []
    over = asyncio.Event()

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        reached.append(customer_id)
        return {"plan": "basic"}

    async def later(session: RunSession) -> None:
        await over.wait()
        with contextlib.suppress(BaseException):
            await session.call_tool("lookup_plan", {"customer_id": "cus-1"})

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        asyncio.get_running_loop().create_task(later(session))  # never awaited
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    db = tmp_path / "j.db"
    run_id = new_ulid()
    result = await scheduler(
        Journal(db), adapter, registry_of([lookup_plan]), AnswersWhen(asyncio.Event())
    ).run(run_id, {})
    over.set()
    await asyncio.sleep(0.3)
    assert result.ok, result.error
    assert reached == [], "the upstream was reached after the run"
    assert after_the_first_end(db, run_id) == []
