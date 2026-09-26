"""A resumed node is handed its recorded answers as it first had them -- and stopped cleanly.

Served at once, in the order the questions were asked, a node that acts on whichever of two
answers comes first, or falls back when one is not back in time without cancelling it, decided
otherwise on resume and in replay, and a different charge went out under a new key. Found by
the sixteenth review, with the rest of this file: a node stopped by ``TurnAbandoned`` whose
``finally`` still wrote; a resume that finished while an earlier attempt's charge might be out,
and said nothing; and a node whose deadline covers earlier work, abandoned on a resume doing
exactly what the run did.

The pacing itself then hung resumes, found by the slow-disk run: an answer waited for every
earlier one of its node to be handed over, and one its node stopped waiting for never was.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.integration.test_crash_then_resume import (
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
    Message,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolUseComplete,
    TurnAbandoned,
    TurnComplete,
)
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger, render_ledger

#: The backup, asked beside the primary: a question of its own.
BACKUP = RequestEnvelope(model="scripted", max_tokens=128, stream=True, messages=SMALL.messages)


class Paced:
    """Answers each question with a charge, after a delay: ``{max_tokens: (seconds, amount)}``.

    ``paced`` is the recorded run's model; a resumed or replayed one answers at once, with
    other amounts -- a model that would decide differently if it were asked.
    """

    def __init__(self, answers: dict[int | None, tuple[float, float]]) -> None:
        self.answers = answers
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.asked.append(envelope.max_tokens)
        delay, amount = self.answers[envelope.max_tokens]
        await asyncio.sleep(delay)
        return charge(amount)  # type: ignore[return-value]

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


def amount_of(response: ModelResponse) -> float:
    return float(response.tool_uses[0].args["amount"])  # type: ignore[arg-type]


def billing(shape: str) -> tuple[PlainAdapter, object, list[float]]:
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}", "amount": amount}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        if shape == "hedge":
            # Ask a primary and a backup at once; charge what the first to answer says, and
            # finish the other before writing -- a turn left open refuses the write.
            primary = asyncio.create_task(model.complete(SMALL))
            backup = asyncio.create_task(model.complete(BACKUP))
            done, _pending = await asyncio.wait(
                {primary, backup}, return_when=asyncio.FIRST_COMPLETED
            )
            first = done.pop()
            await asyncio.gather(primary, backup)
            amount = amount_of(first.result())
        else:
            # Fall back when the answer is not back in time -- without cancelling it.
            asking = asyncio.create_task(model.complete(SMALL))
            done, _pending = await asyncio.wait({asking}, timeout=0.2)
            amount = amount_of(asking.result()) if done else 10.0
            await asking
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": amount})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card]), charged


RECORDED = {
    # The backup answers first, so the node charges the backup's 30.
    "hedge": {64: (0.4, 25.0), 128: (0.05, 30.0)},
    # The answer is not back within 0.2 s, so the node charges its fallback, 10.
    "timeout": {64: (0.5, 25.0)},
}
ASKED_AGAIN = {64: (0.0, 45.0), 128: (0.0, 55.0)}


@pytest.mark.parametrize("shape", ["hedge", "timeout"])
async def test_a_resumed_node_decides_as_it_did_when_its_answers_came_back(
    tmp_path: Path, shape: str
) -> None:
    db = tmp_path / "source.db"
    adapter, registry, charged = billing(shape)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            Paced(RECORDED[shape]),
        ).run(run_id, {})
    await bury_the_dead_process()
    first = list(charged)

    asked_again = Paced(ASKED_AGAIN)
    resumed = await scheduler(Journal(db), adapter, registry, asked_again).resume(run_id)
    assert resumed.ok, resumed.error
    assert asked_again.asked == [], "the resume asked the model what the journal answers"
    assert charged == first, f"a second, different charge: {charged}"


@pytest.mark.parametrize("shape", ["hedge", "timeout"])
async def test_a_replay_decides_as_the_run_did(tmp_path: Path, shape: str) -> None:
    journal = Journal(tmp_path / "source.db")
    adapter, registry, charged = billing(shape)
    result = await scheduler(journal, adapter, registry, Paced(RECORDED[shape])).run(new_ulid(), {})
    assert result.ok, result.error
    adapter, registry, again = billing(shape)
    replayed = await scheduler(
        Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
    ).run(new_ulid(), {})
    assert replayed.ok, replayed.error
    assert again == charged, f"the replay charged {again}; the run charged {charged}"


# -- a node stopped by TurnAbandoned, and the ways out ------------------------------------------


class SlowFirst:
    """Never answers the first question in the recorded run; answers any other at once."""

    def __init__(self, *, hang: bool) -> None:
        self.hang = hang
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(envelope.max_tokens)
        if self.hang and envelope.max_tokens == 64:
            await asyncio.Event().wait()
        reply = charge(20.0)
        yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[attr-defined]
        yield TurnComplete(response=reply)  # type: ignore[arg-type]


def cached_billing() -> tuple[PlainAdapter, object, list[str], dict[str, bool]]:
    """The node asks, and on a cache hit cancels the call and charges; on a miss it waits for
    the call -- and whatever happens, its ``finally`` tells the customer."""
    cache = {"hit": True}
    sent: list[str] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(f"charge {amount}")
        return {"charge_id": "ch_1"}

    @tool(effect="write", idempotent=False)
    async def notify(customer_id: str, text: str) -> JsonValue:
        sent.append(f"notify {text}")
        return {"sent": True}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        asking = asyncio.create_task(session.call_turn(SMALL))  # type: ignore[misc]
        await asyncio.sleep(0.05)
        try:
            if cache["hit"]:
                asking.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asking
                await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
            else:
                await asking
        finally:
            if not cache["hit"]:
                await session.call_tool("notify", {"customer_id": "cus-1", "text": "not charged"})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card, notify]), sent, cache


async def crashed_after_the_cache_hit(
    tmp_path: Path,
) -> tuple[Path, str, object, object, list[str], dict[str, bool]]:
    db = tmp_path / "source.db"
    adapter, registry, sent, cache = cached_billing()
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, SlowFirst(hang=True)
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 10.0"]
    cache["hit"] = False  # expired across the crash: the node waits for the call this time
    return db, run_id, adapter, registry, sent, cache


async def test_an_abandoned_node_neither_writes_nor_asks_on_its_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    db, run_id, adapter, registry, sent, _cache = await crashed_after_the_cache_hit(tmp_path)
    live = SlowFirst(hang=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, live).resume(run_id), timeout=10
    )
    assert not resumed.ok and TurnAbandoned.__name__ in (resumed.error or ""), resumed.error
    assert sent == ["charge 10.0"], "the stopped node's finally reached a customer"
    assert live.asked == []


async def test_an_operator_can_have_an_abandoned_turn_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every resume of that run ends the same way until someone decides; ``ask_abandoned``
    asks the turn again, live, instead."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    db, run_id, adapter, registry, sent, _cache = await crashed_after_the_cache_hit(tmp_path)
    live = SlowFirst(hang=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, live).resume(run_id, ask_abandoned=True),
        timeout=10,
    )
    assert live.asked == [64], "the abandoned turn was not asked again"
    assert resumed.ok, resumed.error
    # What the operator accepted, and the reason it is not the default: the recorded run's
    # charge stands, and the live answer charged again.
    assert sent == ["charge 10.0", "charge 20.0", "notify not charged"]


async def test_a_resume_that_leaves_an_earlier_charge_unsettled_does_not_report_success(
    tmp_path: Path,
) -> None:
    """The process died while the charge was out -- claimed, sent, no reply -- and the resumed
    node charged something else: its code had changed. Nothing ever settled the first claim.
    The resume reported plain success, and the ledger showed only the second charge, while the
    world may hold both."""
    version = {"amount": 25.0}
    sent: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(amount)
        if len(sent) == 1:
            raise Crash("the process died with the charge out and its reply lost")
        return {"charge_id": f"ch_{len(sent)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        await session.call_tool(
            "charge_card", {"customer_id": "cus-1", "amount": version["amount"]}
        )
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).run(run_id, {})
    await bury_the_dead_process()
    version["amount"] = 30.0  # the code changed across the crash
    resumed = await scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id)
    assert sent == [25.0, 30.0]
    assert not resumed.ok and "never settled" in (resumed.error or ""), resumed.error
    assert "MAY HAVE BEEN SENT" in render_ledger(build_ledger(Journal(db), run_id))


async def test_a_deadline_that_covers_earlier_work_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node's deadline covers a read and the model call. The read was slow in the recorded
    run, so the call was cancelled soon after it began; on resume the read is quick, and the
    node's deadline falls well into the call. Measured on the call alone, the wait gave up
    first, and every resume after it did too."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    slow_read = {"seconds": 0.5}
    sent: list[str] = []

    @tool(effect="read")
    async def lookup(customer_id: str) -> JsonValue:
        await asyncio.sleep(slow_read["seconds"])
        return {"customer": customer_id}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(f"charge {amount}")
        return {"charge_id": "ch_1"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            async with asyncio.timeout(0.7):
                await session.call_tool("lookup", {"customer_id": "cus-1"})
                await session.call_turn(SMALL)  # type: ignore[misc]
        except TimeoutError:
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([lookup, charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, SlowFirst(hang=True)
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 10.0"]
    slow_read["seconds"] = 0.01
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id),
        timeout=10,
    )
    assert TurnAbandoned.__name__ not in (resumed.error or ""), resumed.error
    assert sent == ["charge 10.0"]


# -- the pace never hangs a resume ---------------------------------------------------------------


def priced_by_plan(order: dict[str, str]) -> tuple[PlainAdapter, object, list[float]]:
    """The node starts a stream, prices the plan it looks up, then reads the stream to its end
    and charges the price. ``order["reads"]`` says when it reads the rest of the stream: after
    the price comes back (``"price first"``), or before it asks (``"stream first"``)."""
    sent: list[float] = []

    @tool(effect="read")
    async def lookup(customer_id: str) -> JsonValue:
        return {"plan": order["plan"]}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(amount)
        return {"charge_id": f"ch_{len(sent)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        events = model.stream(SMALL).__aiter__()
        await events.__anext__()
        found = await session.call_tool("lookup", {"customer_id": "cus-1"})
        question = RequestEnvelope(
            model="scripted",
            max_tokens=128,
            messages=(Message(role="user", content=(TextBlock(text=str(found["plan"])),)),),  # type: ignore[call-overload,index]
        )
        if order["reads"] == "stream first":
            async for _event in events:
                pass
            priced = await model.complete(question)
        else:
            priced = await model.complete(question)
            async for _event in events:
                pass
        await session.call_tool(
            "charge_card", {"customer_id": "cus-1", "amount": amount_of(priced)}
        )
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([lookup, charge_card]), sent


class Prices:
    """Streams a reply to the first question; prices the plan named in any other."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        plan = envelope.messages[-1].content[0].text  # type: ignore[union-attr]
        self.asked.append(plan)
        return charge({"basic": 25.0, "pro": 30.0}[plan])  # type: ignore[return-value]

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append("stream")
        reply = charge(1.0)
        yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[attr-defined]
        yield TurnComplete(response=reply)  # type: ignore[arg-type]


async def test_a_resume_of_a_resume_serves_its_answers_in_the_order_they_came(
    tmp_path: Path,
) -> None:
    """The first resume was served the stream, and asked for the price live -- the plan had
    changed across the crash. The price came back while the node still held the stream, which
    it read to its end after. Both went into the next resume ordered by where each was first
    recorded: the stream, served and so first recorded by the run before, ahead of the price.
    That resume held the price back until the stream was read, and the node would read the
    stream only once it had the price."""
    db = tmp_path / "source.db"
    order = {"plan": "basic", "reads": "price first"}
    adapter, registry, sent = priced_by_plan(order)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, Prices()
        ).run(run_id, {})
    await bury_the_dead_process()
    order["plan"] = "pro"
    first_resume = Prices()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, first_resume
        ).resume(run_id)
    await bury_the_dead_process()
    assert first_resume.asked == ["pro"], "the setup: the stream served, the price asked live"
    assert sent == [25.0, 30.0]

    asked_again = Prices()
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_again).resume(run_id), timeout=10
    )
    assert resumed.ok, resumed.error
    assert asked_again.asked == []
    assert sent == [25.0, 30.0]


async def test_a_node_that_holds_an_earlier_answer_unread_is_stopped_not_left_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded node read the stream to its end, then asked for the price; changed across
    the crash, it asks for the price first and reads the stream after. The price came back
    after the stream, so it waits for the stream -- which the node reads only once it has the
    price. It waits no longer than the stream's own bound, and the node is stopped."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    db = tmp_path / "source.db"
    order = {"plan": "basic", "reads": "stream first"}
    adapter, registry, sent = priced_by_plan(order)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, Prices()
        ).run(run_id, {})
    await bury_the_dead_process()
    order["reads"] = "price first"
    asked_again = Prices()
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_again).resume(run_id), timeout=10
    )
    assert not resumed.ok and TurnAbandoned.__name__ in (resumed.error or ""), resumed.error
    assert asked_again.asked == []
    assert sent == [25.0], "the stopped node charged again"


async def test_a_node_that_catches_turn_abandoned_is_not_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Told not to, a node catches ``TurnAbandoned`` and returns. Its branch was stopped, so it
    could write nothing -- but what it returned was committed, and the run went on from a node
    that had not done what the recorded one did, and reported success."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    patient = {"now": False}
    sent: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(amount)
        return {"charge_id": "ch_1"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        if patient["now"]:
            try:
                await session.call_turn(SMALL)  # type: ignore[misc]
            except BaseException:  # the mistake under test
                session.state["gave_up"] = True
                return ToolCall("charge_card", {})
        else:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(0.05):
                    await session.call_turn(SMALL)  # type: ignore[misc]
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of(
        [bill], lambda s: None if s.get("billed") or s.get("gave_up") else "bill"
    )
    registry = registry_of([charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, SlowFirst(hang=True)
        ).run(run_id, {})
    await bury_the_dead_process()
    patient["now"] = True
    for _attempt in range(2):
        # Nor the resume after it, which found the node retired and finished the run.
        resumed = await asyncio.wait_for(
            scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id),
            timeout=10,
        )
        assert not resumed.ok and TurnAbandoned.__name__ in (resumed.error or ""), resumed.error
    assert sent == [10.0]
