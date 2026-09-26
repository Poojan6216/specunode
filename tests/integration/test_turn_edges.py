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
from collections.abc import AsyncIterator, Callable, Mapping
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
    TurnAbandoned,
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
    """The leftover question lands on disk just as the Scheduler goes to write run_finished --
    if it is written at all: a question from a node that has returned is refused before it is
    (the twenty-sixth review), and the run then finishes after a short wait."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.in_check = asyncio.Event()
        self.finishing = asyncio.Event()

    async def check_run_lock(self, run_id: str) -> None:
        await super().check_run_lock(run_id)
        self.in_check.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.finishing.wait(), timeout=0.5)

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


# -- the twenty-first review ---------------------------------------------------------------------


class NotesTheOrder:
    """Asks to note the order, a second after the question."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        use = ToolUseBlock(id="t0", name="note_order", args={"customer_id": "cus-1"})
        await asyncio.sleep(1.0)
        yield ToolUseComplete(index=0, block=use)
        yield TurnComplete(ModelResponse(model="scripted", content=(use,), stop_reason="tool_use"))


class SlowerToAsk(Journal):
    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "model_request":
            await asyncio.sleep(0.9)
        return await super().append_async(run_id, kind, payload)


async def test_a_resume_slower_to_write_its_question_decides_as_the_run_did(
    tmp_path: Path,
) -> None:
    """The node gives the model 1.6 s, its question's write included; the model answered in a
    second. On a resume whose disk took 0.9 s to write the question, the served answer's clock
    started after that write, the deadline fired, and the node charged a fallback the run never
    had. Found by the twenty-first review. The deadline leaves room for every other write in the
    turn to be slow, as a loaded CI disk is."""
    world: list[str] = []

    @tool(effect="write", idempotent=False)
    async def note_order(customer_id: str) -> JsonValue:
        world.append("note_order")
        return {"noted": True}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        world.append("charge_card")
        return {"charge_id": "ch"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await asyncio.wait_for(session.call_turn(TURN_FOR_ORDERS), timeout=1.6)  # type: ignore[misc]
        except (TimeoutError, ModelError):
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("note_order", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([note_order, charge_card])

    def driver(journal: Journal) -> Scheduler:
        return Scheduler(
            graph=adapter,  # type: ignore[arg-type]
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(NotesTheOrder(), journal, provider="scripted"),
            policy=Policy(speculation=False),
        )

    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            driver(CrashingJournal(db, before_commit_of("bill#0"))).run(run_id, {}), timeout=30
        )
    await bury_the_dead_process()
    assert world == ["note_order"]
    resumed = await asyncio.wait_for(driver(SlowerToAsk(db)).resume(run_id), timeout=60)
    assert resumed.ok, resumed.error
    assert world == ["note_order"], f"the resume decided otherwise: {world}"


TURN_FOR_ORDERS = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="order for cus-1"),)),),
    max_tokens=256,
    stream=True,
)


def reads_left_running(fail_second: bool) -> tuple[PlainAdapter, object]:
    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        await asyncio.sleep(0.5)
        return {"plan": "basic"}

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        # A lookup started and never awaited: still at the upstream when the run ends.
        asyncio.get_running_loop().create_task(
            session.call_tool("lookup_plan", {"customer_id": "cus-1"})
        )
        await asyncio.sleep(0.05)
        session.state["asked"] = True
        return ToolCall("noop", {})

    @node(name="fail")
    async def fail(session: RunSession) -> Decision:
        if fail_second:
            raise RuntimeError("the second node fails this time")
        session.state["failed_once"] = True
        return ToolCall("noop", {})

    def route(state: Mapping[str, JsonValue]) -> str | None:
        if not state.get("asked"):
            return "ask"
        return None if state.get("failed_once") else "fail"

    return PlainAdapter.of([ask, fail], route), registry_of([lookup_plan])


@pytest.mark.parametrize("resumed_at_once", [False, True])
async def test_a_read_left_running_writes_nothing_after_its_run(
    tmp_path: Path, resumed_at_once: bool
) -> None:
    """Checked only when it began, a read still at the upstream when its run ended wrote its
    result after ``run_finished`` -- or into a resume of the same run the process had begun
    meanwhile. Found by the twenty-first review."""
    db = tmp_path / "j.db"
    run_id = new_ulid()
    adapter, registry = reads_left_running(fail_second=resumed_at_once)
    await scheduler(Journal(db), adapter, registry, AnswersWhen(asyncio.Event())).run(run_id, {})
    first_end = max(e.offset for e in Journal(db).read(run_id, kinds=["run_finished"]))
    if resumed_at_once:
        adapter, registry = reads_left_running(fail_second=False)
        await scheduler(Journal(db), adapter, registry, AnswersWhen(asyncio.Event())).resume(run_id)
    await asyncio.sleep(0.8)
    late = [
        (e.offset, e.kind)
        for e in Journal(db).read(run_id, kinds=["tool_result"])
        if e.offset > first_end
    ]
    assert late == [], late


class TwoReads:
    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        a = ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"})
        b = ToolUseBlock(id="t1", name="lookup_plan", args={"customer_id": "cus-2"})
        await asyncio.sleep(0.01)
        yield ToolUseComplete(index=0, block=a)
        await asyncio.sleep(0.09)
        yield ToolUseComplete(index=1, block=b)
        await asyncio.sleep(0.5)
        yield TurnComplete(ModelResponse(model="scripted", content=(a, b), stop_reason="tool_use"))


class SlowOn(Journal):
    """A disk slow (400 ms) to take one kind of entry."""

    def __init__(self, path: Path, slow: Callable[[str, Mapping[str, JsonValue]], bool]) -> None:
        super().__init__(path)
        self.slow = slow

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if self.slow(kind, payload):
            await asyncio.sleep(0.4)
        return await super().append_async(run_id, kind, payload)


@pytest.mark.parametrize("slow_at", ["the guess is confirmed", "the guess is forked"])
async def test_a_guess_a_deadline_lands_on_is_resolved_once(tmp_path: Path, slow_at: str) -> None:
    """The node's deadline fired while a guess was being confirmed -- and it was squashed again
    from the failure path, counted twice -- or while its fork was being written, and nothing
    ever resolved it. Found by the twenty-first review."""
    from tests.integration.test_speculation import FixedDrafter

    def slow(kind: str, payload: Mapping[str, JsonValue]) -> bool:
        if slow_at == "the guess is confirmed":
            return kind == "branch_resolved" and bool(payload.get("adopted_by"))
        return kind == "branch_forked" and bool(payload.get("predicted_hash"))

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        return {"charge_id": "ch_1"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        with contextlib.suppress(TimeoutError, ModelError):
            await asyncio.wait_for(session.call_turn(TURN_FOR_ORDERS), timeout=0.3)  # type: ignore[misc]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([lookup_plan, charge_card])
    journal = SlowOn(tmp_path / "j.db", slow)
    driver = Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
        target=JournaledModel(TwoReads(), journal, provider="scripted"),
        policy=Policy(speculation=True, max_speculation_depth=3),
        predictor=FixedDrafter(ToolCall("lookup_plan", {"customer_id": "cus-2"})),  # type: ignore[arg-type]
    )
    run_id = new_ulid()
    result = await asyncio.wait_for(driver.run(run_id, {}), timeout=30)
    await asyncio.sleep(0.5)
    assert result.ok, result.error
    forked = {
        e.payload.get("branch_id")
        for e in Journal(tmp_path / "j.db").read(run_id, kinds=["branch_forked"])
        if e.payload.get("predicted_hash")
    }
    resolved = [
        e.payload.get("branch_id")
        for e in Journal(tmp_path / "j.db").read(run_id, kinds=["branch_resolved"])
        if e.payload.get("branch_id") in forked
        and e.payload.get("status") in ("squashed", "retired", "stalled")
    ]
    assert driver.budget.inflight_branches == 0
    assert driver.budget.window.samples == len(forked)
    assert sorted(resolved) == sorted(forked), (resolved, forked)


def test_status_says_what_resume_would(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``status`` called resumable a run ``resume`` refused. Found by the twenty-first review."""
    from tests.integration.test_positions import billing, driving
    from typer.testing import CliRunner

    from specunode.cli import app
    from specunode.core import scheduler as scheduler_module

    async def crashed() -> str:
        monkeypatch.setattr(scheduler_module, "POSITION_RULE", 1)
        charged: list[float] = []
        adapter, registry = billing(charged)
        run_id = new_ulid()
        with pytest.raises(Crash):
            await driving(
                CrashingJournal(tmp_path / "j.db", before_commit_of("bill#0")),
                adapter,
                registry,
                speculation=False,
            ).run(run_id, {})
        await bury_the_dead_process()
        monkeypatch.undo()
        return run_id

    run_id = asyncio.run(crashed())
    shown = CliRunner().invoke(app, ["status", run_id, "--journal", str(tmp_path / "j.db")])
    assert "resumable: False" in shown.output, shown.output


# -- the twenty-second review --------------------------------------------------------------------


class Connection:
    """An HTTP stream's connection: closed in its ``async with`` exit, which takes ``closing``
    seconds -- and cannot be cut short, as a connection pool's release often cannot -- and, with
    ``fails``, fails."""

    def __init__(self, closing: float, *, fails: bool = False) -> None:
        self.closing = closing
        self.fails = fails

    async def __aenter__(self) -> Connection:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.sleep(self.closing)
        if self.fails:
            raise ConnectionResetError("connection reset while closing the stream")


class AnswersThenClosesSlowly:
    """Asks to note the order at 100 ms, inside its own ``async with`` -- as a client built on an
    HTTP stream context does -- whose exit takes ``closing`` seconds to close the connection,
    and with ``fails`` fails doing it. The server keeps its end open for ``lingers`` seconds
    after the answer."""

    def __init__(self, *, closing: float = 1.5, fails: bool = False, lingers: float = 30) -> None:
        self.closing = closing
        self.fails = fails
        self.lingers = lingers

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        use = ToolUseBlock(id="t0", name="note_order", args={"customer_id": "cus-1"})
        async with Connection(self.closing, fails=self.fails):
            await asyncio.sleep(0.1)
            yield ToolUseComplete(index=0, block=use)
            yield TurnComplete(
                ModelResponse(model="scripted", content=(use,), stop_reason="tool_use")
            )
            await asyncio.sleep(self.lingers)


def notes_orders(world: list[str], *, reads_it_itself: bool) -> tuple[PlainAdapter, object]:
    """Note the order if the model answers within a second; charge a fallback if it does not."""

    @tool(effect="write", idempotent=False)
    async def note_order(customer_id: str) -> JsonValue:
        world.append("note_order")
        return {"noted": True}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        world.append("charge_card")
        return {"charge_id": "ch"}

    async def ask(session: RunSession) -> None:
        if not reads_it_itself:
            await session.call_turn(TURN_FOR_ORDERS)  # type: ignore[misc]
            return
        assert session.model is not None
        answered = False
        async for event in session.model.stream(TURN_FOR_ORDERS):
            answered = answered or isinstance(event, TurnComplete)
        assert answered
        await session.call_tool("note_order", {"customer_id": "cus-1"})

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await asyncio.wait_for(ask(session), timeout=1.0)
        except (TimeoutError, ModelError):
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("note_order", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([note_order, charge_card])


@pytest.mark.parametrize("reads_it_itself", [False, True])
async def test_an_answer_on_disk_is_not_given_up_on_while_the_client_closes(
    tmp_path: Path, reads_it_itself: bool
) -> None:
    """The model answered at 100 ms and its client took 1.5 s more to close its connection. The
    turn was journaled as answered, but its stream was read on to its end: the node's one-second
    deadline fired, and it charged a fallback. A replay and a resume, served a stream that ends
    at its answer, took the model's path instead -- a write the run never sent. Found by the
    twenty-second review. The deadline leaves room for every write in the turn to be slow."""
    world: list[str] = []
    adapter, registry = notes_orders(world, reads_it_itself=reads_it_itself)
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            scheduler(
                CrashingJournal(db, before_commit_of("bill#0")),
                adapter,
                registry,
                AnswersThenClosesSlowly(),
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    assert world == ["note_order"], f"the node gave up on an answer on disk: {world}"

    replayed: list[str] = []
    adapter, registry = notes_orders(replayed, reads_it_itself=reads_it_itself)
    result = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(Journal(db), run_id)
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert result.ok, result.error
    assert replayed == ["note_order"]

    adapter, registry = notes_orders(world, reads_it_itself=reads_it_itself)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, AnswersThenClosesSlowly()).resume(run_id),
        timeout=60,
    )
    assert resumed.ok, resumed.error
    assert world == ["note_order"], f"the resume decided otherwise: {world}"


@pytest.mark.parametrize("closing", [0.0, 0.05])
async def test_a_client_failing_as_it_closes_does_not_undo_an_answer_on_disk(
    tmp_path: Path, closing: float
) -> None:
    """A client that failed as it closed its connection, after its answer: the answer was on
    disk, and the node was handed the client's own error in its place -- not a ``ModelError``,
    so a node catching that failed. Found by the twenty-second review."""
    seen: list[str] = []

    @tool(effect="write", idempotent=False)
    async def note_order(customer_id: str) -> JsonValue:
        return {"noted": True}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await session.call_turn(TURN_FOR_ORDERS)  # type: ignore[misc]
            seen.append("answered")
        except ModelError as exc:
            seen.append(f"ModelError: {exc}")
        session.state["billed"] = True
        return ToolCall("note_order", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    result = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "j.db"),
            adapter,
            registry_of([note_order]),
            AnswersThenClosesSlowly(closing=closing, fails=True, lingers=0),
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert result.ok, result.error
    assert seen == ["answered"]


class LooksUpForAWhile:
    """A lookup at ``block_at``, and the turn's end at ``end_at`` -- or, with ``fails``, a
    failure there. ``complete`` prices the order."""

    def __init__(self, block_at: float, end_at: float, *, fails: bool = False) -> None:
        self.block_at = block_at
        self.end_at = end_at
        self.fails = fails

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return ModelResponse(model="scripted", content=(TextBlock(text="25"),))

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        use = ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"})
        await asyncio.sleep(self.block_at)
        yield ToolUseComplete(index=0, block=use)
        await asyncio.sleep(self.end_at - self.block_at)
        if self.fails:
            raise ModelError("overloaded")
        yield TurnComplete(ModelResponse(model="scripted", content=(use,), stop_reason="tool_use"))


class SlowToConfirm(Journal):
    """A disk slow (300 ms) to take that ``plan#0`` is confirmed -- its retirement's start."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.owners: dict[str, str] = {}

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "branch_forked":
            self.owners[str(payload.get("branch_id"))] = str(payload.get("node_id"))
        if (
            kind == "branch_resolved"
            and payload.get("status") == "confirmed"
            and self.owners.get(str(payload.get("branch_id"))) == "plan#0"
        ):
            await asyncio.sleep(0.3)
        return await super().append_async(run_id, kind, payload)


SUMMARY_TURN = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="summarise cus-1"),)),),
    max_tokens=256,
    stream=True,
)
PRICE_TURN = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="price for cus-1"),)),),
    max_tokens=256,
)


def cursors_after(journal: Journal, run_id: str) -> list[tuple[str, object]]:
    """Each retired node, and the position its retirement journaled."""
    nodes = {
        e.payload.get("branch_id"): e.payload.get("node_id")
        for e in journal.read(run_id, kinds=["branch_forked"])
    }
    retired: list[tuple[str, object]] = []
    for entry in journal.read(run_id, kinds=["branch_resolved"]):
        if entry.payload.get("status") == "retired":
            cursor = entry.payload.get("cursor_after")
            step = cursor.get("step_index") if isinstance(cursor, Mapping) else None
            retired.append((str(nodes.get(entry.payload.get("branch_id"))), step))
    return retired


@pytest.mark.parametrize("left_running", ["a turn", "a turn that fails", "a lookup"])
async def test_what_a_node_left_running_takes_no_position_while_it_retires(
    tmp_path: Path, left_running: str
) -> None:
    """A node left something running and returned; it arrived -- a turn's block, the failure of
    a turn whose block had come while the node ran, a lookup -- while the node's retirement was
    being written, and moved the position that retirement journaled: a block took one, a
    failure gave one back. The replay, on a quick disk, retired the node first: the next node
    asked at a position the journal had no turn for, and a finished run did not replay. Found
    by the twenty-second review."""
    charged: list[str] = []

    def build() -> tuple[PlainAdapter, object]:
        @tool(effect="read")
        async def lookup_plan(customer_id: str) -> JsonValue:
            return {"plan": "basic"}

        @tool(effect="write", idempotent=False)
        async def charge_card(customer_id: str, amount: float) -> JsonValue:
            charged.append(f"charge {amount}")
            return {"charge_id": "ch"}

        async def left(session: RunSession) -> None:
            with contextlib.suppress(BaseException):
                if left_running == "a lookup":
                    await asyncio.sleep(0.25)
                    await session.call_tool("lookup_plan", {"customer_id": "cus-1"})
                else:
                    await session.call_turn(SUMMARY_TURN)  # type: ignore[misc]

        @node(name="plan")
        async def plan(session: RunSession) -> Decision:
            asyncio.get_running_loop().create_task(left(session))  # never awaited
            await asyncio.sleep(0.15)
            session.state["planned"] = True
            return ToolCall("lookup_plan", {})

        @node(name="bill")
        async def bill(session: RunSession) -> Decision:
            assert session.model is not None
            price = await session.model.complete(PRICE_TURN)
            await session.call_tool(
                "charge_card", {"customer_id": "cus-1", "amount": float(price.text)}
            )
            session.state["billed"] = True
            return ToolCall("charge_card", {})

        def route(state: Mapping[str, JsonValue]) -> str | None:
            if not state.get("planned"):
                return "plan"
            return None if state.get("billed") else "bill"

        return PlainAdapter.of([plan, bill], route), registry_of([lookup_plan, charge_card])

    # The node returns at 150 ms; its retirement is written from then until about 450 ms. A block
    # at 20 ms comes while it runs, one at 250 ms while it retires.
    model = (
        LooksUpForAWhile(0.02, 0.2, fails=True)
        if left_running == "a turn that fails"
        else LooksUpForAWhile(0.25, 0.35)
    )
    db = tmp_path / "run.db"
    run_id = new_ulid()
    adapter, registry = build()
    result = await asyncio.wait_for(
        scheduler(SlowToConfirm(db), adapter, registry, model).run(run_id, {}), timeout=30
    )
    await asyncio.sleep(0.5)  # what was left running ends
    assert result.ok, result.error

    adapter, registry = build()
    replay_db = tmp_path / "replay.db"
    replay_run = new_ulid()
    replayed = await asyncio.wait_for(
        scheduler(Journal(replay_db), adapter, registry, replay_of(Journal(db), run_id)).run(
            replay_run, {}
        ),
        timeout=30,
    )
    await asyncio.sleep(0.5)
    assert replayed.ok, replayed.error
    assert cursors_after(Journal(db), run_id) == cursors_after(Journal(replay_db), replay_run)


class LooksUpTwiceSlowly:
    """A lookup at 10 ms; a second, different one at ``second_at``; the turn's end at 400 ms --
    after the run is over."""

    def __init__(self, second_at: float) -> None:
        self.second_at = second_at

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        first = ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"})
        second = ToolUseBlock(id="t1", name="lookup_notes", args={"customer_id": "cus-1"})
        await asyncio.sleep(0.01)
        yield ToolUseComplete(index=0, block=first)
        await asyncio.sleep(self.second_at - 0.01)
        yield ToolUseComplete(index=1, block=second)
        await asyncio.sleep(0.4 - self.second_at)
        yield TurnComplete(
            ModelResponse(model="scripted", content=(first, second), stop_reason="tool_use")
        )


@pytest.mark.parametrize(
    ("guess", "second_at"),
    [("wrong", 0.39), ("right", 0.39), ("right", 0.02)],
    ids=["wrong", "right", "confirmed before its node returned"],
)
async def test_a_guess_of_a_turn_left_running_is_settled_before_the_run_ends(
    tmp_path: Path, guess: str, second_at: float
) -> None:
    """A turn its node left running had a guess open; the model's next block settled it after
    the run was over -- squashed or confirmed, written after ``run_finished`` -- and a confirmed
    one made the finished run read as resumable. One confirmed before its node returned was
    discarded, unadopted, when the turn ended after the run -- and that was written after
    ``run_finished`` too. Found by the twenty-second review."""
    from typer.testing import CliRunner

    from specunode.cli import app
    from specunode.journal.replay import recover

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="read")
    async def lookup_notes(customer_id: str) -> JsonValue:
        return {"notes": []}

    async def summarise(session: RunSession) -> None:
        with contextlib.suppress(BaseException):
            await session.call_turn(SUMMARY_TURN)  # type: ignore[misc]

    @node(name="plan")
    async def plan(session: RunSession) -> Decision:
        asyncio.get_running_loop().create_task(summarise(session))  # never awaited
        await asyncio.sleep(0.15)
        session.state["planned"] = True
        return ToolCall("lookup_plan", {})

    predicted = (
        ToolCall("lookup_notes", {"customer_id": "cus-1"})
        if guess == "right"
        else ToolCall("lookup_plan", {"customer_id": "cus-2"})
    )
    registry = registry_of([lookup_plan, lookup_notes])
    db = tmp_path / "j.db"
    journal = Journal(db)
    run_id = new_ulid()
    result = await asyncio.wait_for(
        Scheduler(
            graph=PlainAdapter.of([plan], lambda s: None if s.get("planned") else "plan"),  # type: ignore[arg-type]
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(LooksUpTwiceSlowly(second_at), journal, provider="scripted"),
            policy=Policy(speculation=True, max_speculation_depth=3),
            predictor=FixedDrafter(predicted),  # type: ignore[arg-type]
        ).run(run_id, {}),
        timeout=30,
    )
    assert result.ok, result.error
    await asyncio.sleep(0.8)  # the rest of the reply arrives
    assert after_the_first_end(db, run_id) == []
    entries = list(Journal(db).read(run_id))
    forked = {e.payload.get("branch_id") for e in entries if e.kind == "branch_forked"}
    resolved = {e.payload.get("branch_id") for e in entries if e.kind == "branch_resolved"}
    assert forked <= resolved, f"{forked - resolved} forked and never resolved"
    assert not recover(Journal(db), run_id).resumable
    shown = await asyncio.to_thread(
        CliRunner().invoke, app, ["status", run_id, "--journal", str(db)]
    )
    assert "resumable: False" in shown.output, shown.output


# -- the twenty-third review ---------------------------------------------------------------------


class SummarisesThenClosesSlowly:
    """A summary -- text, no call in it -- at 100 ms, inside an ``async with`` whose exit takes
    1.5 s to close the connection; the server keeps its end open after the answer. ``complete``
    prices the order, in a second."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        await asyncio.sleep(1.0)
        return ModelResponse(model="scripted", content=(TextBlock(text="25"),))

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        text = "cus-1 ordered one widget"
        async with Connection(1.5):
            await asyncio.sleep(0.1)
            yield TextDelta(index=0, text=text)
            yield TurnComplete(
                ModelResponse(
                    model="scripted", content=(TextBlock(text=text),), stop_reason="end_turn"
                )
            )
            await asyncio.sleep(30)


def prices_orders(
    world: list[str], seen: list[str], *, reads_it_itself: bool = False, race: bool = False
) -> tuple[PlainAdapter, object]:
    """Summarise, then price, within a second -- or charge the standard 10 if that is not done
    by then. With ``race``: no deadline, but whichever of the summary and the price is back
    first decides -- 30 for the summary, the model's price for the price."""

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        world.append(f"charge_card {amount}")
        return {"charge_id": f"ch_{amount}"}

    async def summarise(session: RunSession) -> None:
        if reads_it_itself:
            assert session.model is not None
            async for _event in session.model.stream(SUMMARY_TURN):
                pass
        else:
            await session.call_turn(SUMMARY_TURN)  # type: ignore[misc]

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        assert session.model is not None
        loop = asyncio.get_running_loop()
        began = loop.time()
        if race:
            summary = asyncio.ensure_future(summarise(session))
            price = asyncio.ensure_future(session.model.complete(PRICE_TURN))
            done, _ = await asyncio.wait({summary, price}, return_when=asyncio.FIRST_COMPLETED)
            amount = 30.0 if summary in done else float(price.result().text)
            seen.append("the summary came first" if summary in done else "the price came first")
            await asyncio.wait({summary, price})
        else:
            try:
                async with asyncio.timeout(1.0):
                    await summarise(session)
                    priced = await session.model.complete(PRICE_TURN)
                amount = float(priced.text)
                seen.append(f"priced at {loop.time() - began:.1f} s")
            except TimeoutError:
                amount = 10.0
                seen.append(f"fell back at {loop.time() - began:.1f} s")
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": amount})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card])


@pytest.mark.parametrize(
    ("reads_it_itself", "race"),
    [(False, False), (True, False), (False, True)],
    ids=["call_turn under a deadline", "stream read by the node under a deadline", "a race"],
)
async def test_a_client_closing_its_connection_does_not_decide_for_the_node(
    tmp_path: Path, reads_it_itself: bool, race: bool
) -> None:
    """The answer was on disk at 100 ms, and the stream then waited -- uncancellably -- for the
    client to close its connection in its ``async with`` exit. A deadline that fired meanwhile
    was swallowed, and the node priced the order at 2.6 s where a resume, with no connection to
    close, fell back at 1 s; a node racing the summary against the price had the price win, and
    the resume the summary. Either way the resume charged a second, different amount under a
    second key, and said ok. Found by the twenty-third review."""
    world: list[str] = []
    seen: list[str] = []
    adapter, registry = prices_orders(world, seen, reads_it_itself=reads_it_itself, race=race)
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            scheduler(
                CrashingJournal(db, before_commit_of("bill#0")),
                adapter,
                registry,
                SummarisesThenClosesSlowly(),
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    decided = "the summary came first" if race else "fell back"
    assert len(seen) == 1 and seen[0].startswith(decided), f"the run decided otherwise: {seen}"
    if not race:
        # Not held up by the 1.5 s close: at the deadline, and a write or two after it.
        assert float(seen[0].split()[-2]) < 1.5, f"the deadline was held up: {seen}"

    resumed_seen: list[str] = []
    adapter, registry = prices_orders(
        world, resumed_seen, reads_it_itself=reads_it_itself, race=race
    )
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, SummarisesThenClosesSlowly()).resume(run_id),
        timeout=60,
    )
    assert resumed.ok, resumed.error
    assert len(resumed_seen) == 1 and resumed_seen[0].startswith(decided), resumed_seen
    assert len(world) == 1, f"charged {world} for one order"


class GuessesSlowly:
    """A drafter that takes ``delay`` seconds over its first guess -- a tier-2 drafter asking
    its own model -- and guesses nothing after it."""

    def __init__(self, decision: ToolCall, delay: float) -> None:
        self.decision = decision
        self.delay = delay
        self.asked = 0

    async def predict(self, context: object) -> list[object]:
        from specunode.drafters.base import Prediction

        self.asked += 1
        if self.asked > 1:
            return []
        await asyncio.sleep(self.delay)
        return [Prediction(decision=self.decision, tier=2, score=0.5)]


class LooksUpOnceThenEnds:
    """A lookup at ``block_at``; the turn's end at 800 ms."""

    def __init__(self, block_at: float) -> None:
        self.block_at = block_at

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        first = ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"})
        await asyncio.sleep(self.block_at)
        yield ToolUseComplete(index=0, block=first)
        await asyncio.sleep(0.8 - self.block_at)
        yield TurnComplete(
            ModelResponse(model="scripted", content=(first,), stop_reason="tool_use")
        )


class SlowToFork(Journal):
    """A disk slow (200 ms) to take a guess's fork."""

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "branch_forked" and payload.get("parent_id") is not None:
            await asyncio.sleep(0.2)
        return await super().append_async(run_id, kind, payload)


@pytest.mark.parametrize(
    "while_it",
    ["guessed", "wrote its fork"],
    ids=["the drafter was guessing", "the guess's fork was being written"],
)
async def test_a_guess_is_not_opened_or_run_once_its_node_returned(
    tmp_path: Path, while_it: str
) -> None:
    """A turn its node left running was guessing -- a drafter asking its own model -- or writing
    its guess's fork when the node returned. The guess was opened all the same: its call reached
    the upstream after the node had returned, and it was never resolved -- or it was resolved
    before its fork was on disk. Found by the twenty-third review."""
    upstream: list[str] = []

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        upstream.append("lookup_plan")
        return {"plan": "basic"}

    @tool(effect="read")
    async def lookup_notes(customer_id: str) -> JsonValue:
        upstream.append("lookup_notes")
        return {"notes": []}

    async def summarise(session: RunSession) -> None:
        with contextlib.suppress(BaseException):
            await session.call_turn(SUMMARY_TURN)  # type: ignore[misc]

    @node(name="plan")
    async def plan(session: RunSession) -> Decision:
        asyncio.get_running_loop().create_task(summarise(session))  # never awaited
        await asyncio.sleep(0.15)
        session.state["planned"] = True
        return ToolCall("lookup_plan", {})

    guess = ToolCall("lookup_notes", {"customer_id": "cus-1"})
    registry = registry_of([lookup_plan, lookup_notes])
    db = tmp_path / "j.db"
    # The node returns at 150 ms: while a drafter slow to guess is at it (from 10 ms to 210 ms)
    # -- its retirement still being written then -- or while a guess's fork is written (from 50
    # ms to 250 ms).
    journal = SlowToFork(db) if while_it == "wrote its fork" else SlowToConfirm(db)
    drafter = FixedDrafter(guess) if while_it == "wrote its fork" else GuessesSlowly(guess, 0.2)
    run_id = new_ulid()
    result = await asyncio.wait_for(
        Scheduler(
            graph=PlainAdapter.of([plan], lambda s: None if s.get("planned") else "plan"),  # type: ignore[arg-type]
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(
                LooksUpOnceThenEnds(0.05 if while_it == "wrote its fork" else 0.01),
                journal,
                provider="scripted",
            ),
            policy=Policy(speculation=True, max_speculation_depth=3),
            predictor=drafter,  # type: ignore[arg-type]
        ).run(run_id, {}),
        timeout=30,
    )
    assert result.ok, result.error
    await asyncio.sleep(1.0)  # the rest of the reply arrives, and the turn ends
    assert "lookup_notes" not in upstream, "a guess ran after its node returned"
    assert after_the_first_end(db, run_id) == []
    entries = list(Journal(db).read(run_id))
    forked = {e.payload.get("branch_id"): e.offset for e in entries if e.kind == "branch_forked"}
    resolved = {
        e.payload.get("branch_id"): e.offset for e in entries if e.kind == "branch_resolved"
    }
    assert set(forked) <= set(resolved), f"{set(forked) - set(resolved)} never resolved"
    assert all(forked[b] < resolved[b] for b in forked), "resolved before its fork was on disk"
    guesses = [e for e in entries if e.kind == "branch_forked" and e.payload.get("parent_id")]
    if while_it == "guessed":
        assert guesses == [], "a guess was forked after its node returned"


@pytest.mark.parametrize("fails_at", [0.13, 0.17], ids=["before", "after"])
async def test_a_turn_left_running_takes_no_positions_whenever_it_fails(
    tmp_path: Path, fails_at: float
) -> None:
    """A turn a node left running took a position with a block that came while the node ran.
    Failing before the node returned, it gave the position back; failing after, it kept it --
    and where the node's retirement said the run carries on from depended on which. The node
    returns at 150 ms. Found by the twenty-third review."""

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    async def summarise(session: RunSession) -> None:
        with contextlib.suppress(BaseException):
            await session.call_turn(SUMMARY_TURN)  # type: ignore[misc]

    @node(name="plan")
    async def plan(session: RunSession) -> Decision:
        asyncio.get_running_loop().create_task(summarise(session))  # never awaited
        await asyncio.sleep(0.15)
        session.state["planned"] = True
        return ToolCall("lookup_plan", {})

    db = tmp_path / "j.db"
    run_id = new_ulid()
    result = await asyncio.wait_for(
        scheduler(
            Journal(db),
            PlainAdapter.of([plan], lambda s: None if s.get("planned") else "plan"),
            registry_of([lookup_plan]),
            LooksUpForAWhile(0.02, fails_at, fails=True),
        ).run(run_id, {}),
        timeout=30,
    )
    await asyncio.sleep(0.3)
    assert result.ok, result.error
    assert cursors_after(Journal(db), run_id) == [("plan#0", 0)]


class SlowTo(Journal):
    """A disk slow (100 ms) to take entries of ``kinds``."""

    def __init__(self, path: Path, kinds: set[str]) -> None:
        super().__init__(path)
        self.kinds = kinds

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind in self.kinds:
            await asyncio.sleep(0.1)
        return await super().append_async(run_id, kind, payload)


@pytest.mark.parametrize(
    "slow",
    [set(), {"tool_request"}, {"branch_resolved"}],
    ids=["a quick disk", "slow to take the call", "slow to take the retirement"],
)
async def test_a_write_left_running_is_not_staged_once_its_node_returned(
    tmp_path: Path, slow: set[str]
) -> None:
    """A write the node started and never awaited, begun before the node returned, reached the
    store buffer after it: staged during the retirement and sent -- or, a little later, left
    waiting on a drain that had already been made. Found by the twenty-third review."""
    sent: list[str] = []
    left: list[asyncio.Task[JsonValue]] = []

    @tool(effect="write", idempotent=False)
    async def notify(customer_id: str) -> JsonValue:
        sent.append("notify")
        return {"ok": True}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        call = session.call_tool("notify", {"customer_id": "cus-1"})
        left.append(asyncio.get_running_loop().create_task(call))  # type: ignore[arg-type]
        await asyncio.sleep(0)  # the call begins -- takes its position -- before the return
        session.state["billed"] = True
        return ToolCall("notify", {})

    db = tmp_path / "j.db"
    run_id = new_ulid()
    result = await asyncio.wait_for(
        scheduler(
            SlowTo(db, slow) if slow else Journal(db),
            PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill"),
            registry_of([notify]),
            AnswersThenClosesSlowly(),
        ).run(run_id, {}),
        timeout=30,
    )
    await asyncio.wait(left, timeout=2)
    assert result.ok, result.error
    assert left[0].done(), "the call left running is still waiting"
    assert isinstance(left[0].exception(), TurnAbandoned)
    assert sent == []


# -- the twenty-fourth review --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ends_at", "fails"),
    [(0.06, False), (0.1, False), (0.13, False), (0.1, True)],
    ids=["answered at 60 ms", "answered at 100 ms", "answered at 130 ms", "failed at 100 ms"],
)
async def test_a_turn_left_running_is_judged_by_when_its_answer_arrived(
    tmp_path: Path, ends_at: float, fails: bool
) -> None:
    """The node returns at 150 ms, and the run's disk takes 100 ms to write an answer. A turn it
    left running, answered just before the return, was still under way live -- its answer being
    written -- and done in the replay, which writes nothing: live it gave its positions back,
    replayed it kept them, and the next node asked at a position the journal had no turn for.
    A finished run did not replay. Found by the twenty-fourth review."""
    charged: list[str] = []

    def build() -> tuple[PlainAdapter, object]:
        @tool(effect="read")
        async def lookup_plan(customer_id: str) -> JsonValue:
            return {"plan": "basic"}

        @tool(effect="write", idempotent=False)
        async def charge_card(customer_id: str, amount: float) -> JsonValue:
            charged.append(f"charge {amount}")
            return {"charge_id": "ch"}

        async def summarise(session: RunSession) -> None:
            with contextlib.suppress(BaseException):
                await session.call_turn(SUMMARY_TURN)  # type: ignore[misc]

        @node(name="plan")
        async def plan(session: RunSession) -> Decision:
            asyncio.get_running_loop().create_task(summarise(session))  # never awaited
            await asyncio.sleep(0.15)
            session.state["planned"] = True
            return ToolCall("lookup_plan", {})

        @node(name="bill")
        async def bill(session: RunSession) -> Decision:
            assert session.model is not None
            price = await session.model.complete(PRICE_TURN)
            await session.call_tool(
                "charge_card", {"customer_id": "cus-1", "amount": float(price.text)}
            )
            session.state["billed"] = True
            return ToolCall("charge_card", {})

        def route(state: Mapping[str, JsonValue]) -> str | None:
            if not state.get("planned"):
                return "plan"
            return None if state.get("billed") else "bill"

        return PlainAdapter.of([plan, bill], route), registry_of([lookup_plan, charge_card])

    db = tmp_path / "run.db"
    run_id = new_ulid()
    adapter, registry = build()
    result = await asyncio.wait_for(
        scheduler(
            SlowTo(db, {"model_response"}),
            adapter,
            registry,
            LooksUpForAWhile(0.01, ends_at, fails=fails),
        ).run(run_id, {}),
        timeout=30,
    )
    await asyncio.sleep(0.5)  # what was left running ends
    assert result.ok, result.error

    adapter, registry = build()
    replay_db = tmp_path / "replay.db"
    replay_run = new_ulid()
    replayed = await asyncio.wait_for(
        scheduler(Journal(replay_db), adapter, registry, replay_of(Journal(db), run_id)).run(
            replay_run, {}
        ),
        timeout=30,
    )
    await asyncio.sleep(0.5)
    assert replayed.ok, replayed.error
    assert cursors_after(Journal(db), run_id) == cursors_after(Journal(replay_db), replay_run)


# -- the twenty-sixth review ---------------------------------------------------------------------


class LooksUpThenCharges:
    """A lookup at 10 ms, a charge at 30 ms, and the turn's end at 40 ms."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return ModelResponse(model="scripted", content=(TextBlock(text="ok"),))

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        look = ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"})
        charge = ToolUseBlock(
            id="t1", name="charge_card", args={"customer_id": "cus-1", "amount": 25.0}
        )
        await asyncio.sleep(0.01)
        yield ToolUseComplete(index=0, block=look)
        await asyncio.sleep(0.02)
        yield ToolUseComplete(index=1, block=charge)
        await asyncio.sleep(0.01)
        yield TurnComplete(
            ModelResponse(model="scripted", content=(look, charge), stop_reason="tool_use")
        )


@pytest.mark.parametrize(
    ("lookup_takes", "returns_at"),
    [(0.01, 1.0), (0.4, 0.15)],
    ids=["settled before its node returns", "a slow lookup outlasts its node"],
)
async def test_a_turn_settling_as_its_node_returns_ends_and_adopts_nothing_after(
    tmp_path: Path, lookup_takes: float, returns_at: float
) -> None:
    """A turn a node left running was answered before the node returned, and still settling --
    waiting on a slow lookup -- when it did. It then adopted a confirmed guess's charge into the
    node, which had retired by then: the charge was never sent, never discarded, and the turn
    waited on its ack for ever. Found by the twenty-sixth review."""
    charged: list[float] = []
    left: list[asyncio.Task[object]] = []
    collected: list[str] = []

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        await asyncio.sleep(lookup_takes)
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": "ch"}

    @node(name="plan")
    async def plan(session: RunSession) -> Decision:
        turn = session.call_turn(TURN_FOR_ORDERS)  # type: ignore[misc]
        left.append(asyncio.get_running_loop().create_task(turn))  # never awaited here
        await asyncio.sleep(returns_at)
        session.state["planned"] = True
        return ToolCall("lookup_plan", {})

    @node(name="finish")
    async def finish(session: RunSession) -> Decision:
        try:
            await asyncio.wait_for(asyncio.shield(left[0]), timeout=3.0)
            collected.append("its results")
        except TimeoutError:
            collected.append("still waiting after 3 s")
        except BaseException as exc:
            collected.append(type(exc).__name__)
        session.state["finished"] = True
        return ToolCall("lookup_plan", {})

    def route(state: Mapping[str, JsonValue]) -> str | None:
        if not state.get("planned"):
            return "plan"
        return None if state.get("finished") else "finish"

    registry = registry_of([lookup_plan, charge_card])
    db = tmp_path / "run.db"
    journal = Journal(db)
    run_id = new_ulid()
    result = await asyncio.wait_for(
        Scheduler(
            graph=PlainAdapter.of([plan, finish], route),  # type: ignore[arg-type]
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(LooksUpThenCharges(), journal, provider="scripted"),
            policy=Policy(speculation=True, speculate_writes=True),
            predictor=FixedDrafter(  # type: ignore[arg-type]
                ToolCall("charge_card", {"customer_id": "cus-1", "amount": 25.0})
            ),
        ).run(run_id, {}),
        timeout=30,
    )
    assert result.ok, result.error
    kinds = [e.kind for e in Journal(db).read(run_id)]
    if returns_at > lookup_takes:
        # Settled while its node ran: the charge went out with the node.
        assert collected == ["its results"] and charged == [25.0], (collected, charged)
    else:
        assert collected == ["TurnAbandoned"], collected
        assert charged == [] and "effect_adopted" not in kinds, kinds
        # Staged, if it got that far, and then discarded -- never left waiting on a drain.
        entries = list(Journal(db).read(run_id))
        staged = {e.payload.get("effect_id") for e in entries if e.kind == "effect_staged"}
        discarded = {
            effect
            for e in entries
            if e.kind == "effect_discarded"
            for effect in e.payload.get("effect_ids") or []  # type: ignore[union-attr]
        }
        assert staged <= discarded, f"staged and left: {staged - discarded}"


async def test_a_question_left_running_is_not_put_once_its_node_returned(tmp_path: Path) -> None:
    """A task a node left running put a question to the model after the node had returned: it
    was asked live, and journaled after the node retired. Found by the twenty-sixth review."""
    asked: list[str] = []
    refused: list[str] = []

    class Counts:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            asked.append("complete")
            return ModelResponse(model="scripted", content=(TextBlock(text="ok"),))

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            raise NotImplementedError
            yield  # pragma: no cover

    async def later(session: RunSession) -> None:
        await asyncio.sleep(0.1)
        assert session.model is not None
        try:
            await session.model.complete(PRICE_TURN)
        except TurnAbandoned as exc:
            refused.append(str(exc))

    @node(name="plan")
    async def plan(session: RunSession) -> Decision:
        asyncio.get_running_loop().create_task(later(session))  # never awaited
        session.state["planned"] = True
        return ToolCall("noop", {})

    @node(name="wait")
    async def wait(session: RunSession) -> Decision:
        await asyncio.sleep(0.3)  # the run is still driven when the question is put
        session.state["waited"] = True
        return ToolCall("noop", {})

    def route(state: Mapping[str, JsonValue]) -> str | None:
        if not state.get("planned"):
            return "plan"
        return None if state.get("waited") else "wait"

    result = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "j.db"),
            PlainAdapter.of([plan, wait], route),
            registry_of([]),
            Counts(),
        ).run(new_ulid(), {}),
        timeout=30,
    )
    assert result.ok, result.error
    assert asked == [], "a question was put after its node returned"
    assert refused and "has returned" in refused[0], refused
