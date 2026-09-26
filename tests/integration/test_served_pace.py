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
from dataclasses import replace
from pathlib import Path
from typing import Any

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

from specunode.buffer.dispatcher import ToolDispatchError
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import RunSession
from specunode.core.model import (
    Message,
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
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger, render_ledger
from specunode.testing.models import free_text_turn, tool_turn

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
    first, and every resume after it did too.

    The deadline is set from the node's start, and in the recorded run it is placed just after
    the question reached the model -- so where it falls does not depend on how fast the
    journal writes. Fixed at 0.7 s, a slower disk moved it before the question was asked."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    slow_read = {"seconds": 0.5}
    clock: dict[str, Any] = {}
    sent: list[str] = []

    class ThenTheDeadline(SlowFirst):
        """The recorded run's model: never answers, and the node's deadline comes 0.2 s after
        the question reached it."""

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            arm = clock.pop("arm", None)
            if arm is not None:
                arm()
            async for event in super().stream(envelope):
                yield event

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
        loop = asyncio.get_running_loop()
        began = loop.time()
        try:
            async with asyncio.timeout(None) as deadline:
                if "after" in clock:
                    deadline.reschedule(began + clock["after"])
                else:

                    def arm() -> None:
                        when = loop.time() + 0.2
                        clock["after"] = when - began
                        deadline.reschedule(when)

                    clock["arm"] = arm
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
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            ThenTheDeadline(hang=True),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 10.0"]
    assert "after" in clock, "the setup: the recorded question reached the model"
    slow_read["seconds"] = 0.01
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id),
        timeout=30,
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


# -- what a served stream hands over: the pieces, when they came (review 17) ---------------------


class Talks:
    """Starts talking at once, and talks slowly: its text in four pieces, the first at 0.05 s
    and the last at 0.65 s, then a charge of ``amount``."""

    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.asked = 0

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked += 1
        use = charge(self.amount).content[0]  # type: ignore[attr-defined]
        pieces = ["Let me ", "look at ", "your plan ", "and price it. "]
        for n, piece in enumerate(pieces):
            await asyncio.sleep(0.05 if n == 0 else 0.2)
            yield TextDelta(index=0, text=piece)
        yield ToolUseComplete(index=1, block=use)
        text = TextBlock(text="".join(pieces))
        yield TurnComplete(
            response=ModelResponse(model="scripted", content=(text, use), stop_reason="tool_use")
        )


def slow_to_start_billing() -> tuple[PlainAdapter, object, list[float]]:
    """A model slow to say its first word is not waited for: the node charges a standard 10."""
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        model = session.model
        assert model is not None
        events = model.stream(SMALL).__aiter__()
        try:
            await asyncio.wait_for(events.__anext__(), timeout=0.4)
        except TimeoutError:
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        else:
            reply = None
            async for event in events:
                if isinstance(event, TurnComplete):
                    reply = event.response
            assert reply is not None
            use = next(b for b in reply.content if isinstance(b, ToolUseBlock))
            await session.call_tool("charge_card", dict(use.args))
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card]), charged


async def test_a_served_stream_starts_talking_when_the_model_did(tmp_path: Path) -> None:
    """A text block was timed by its last piece and handed over whole then: the first word of a
    served turn came as late as the recorded turn's last, the node gave up on a model that had
    started at once, and charged its fallback -- a second charge, on a resume that changed
    nothing, and a different one in replay. Found by the seventeenth review."""
    db = tmp_path / "source.db"
    adapter, registry, charged = slow_to_start_billing()
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, Talks(25.0)
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0], "the setup: the model started in time"
    asked_again = Talks(99.0)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_again).resume(run_id), timeout=20
    )
    assert resumed.ok, resumed.error
    assert asked_again.asked == 0
    assert charged == [25.0], f"a second, different charge: {charged}"


async def test_a_replayed_stream_starts_talking_when_the_model_did(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "source.db")
    adapter, registry, charged = slow_to_start_billing()
    result = await scheduler(journal, adapter, registry, Talks(25.0)).run(new_ulid(), {})
    assert result.ok and charged == [25.0], result.error
    adapter, registry, again = slow_to_start_billing()
    replayed = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
        ).run(new_ulid(), {}),
        timeout=20,
    )
    assert replayed.ok, replayed.error
    assert again == charged, f"the replay charged {again}; the run charged {charged}"


class ReadsThenCharges:
    """Asks to look up the plan at 1.0 s, then to charge ``amount``."""

    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.asked = 0

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked += 1
        reply = tool_turn(
            ("lookup_plan", {"customer_id": "cus-1"}),
            ("charge_card", {"customer_id": "cus-1", "amount": self.amount}),
        )
        await asyncio.sleep(1.0)
        yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[arg-type]
        await asyncio.sleep(0.1)
        yield ToolUseComplete(index=1, block=reply.content[1])  # type: ignore[arg-type]
        yield TurnComplete(response=reply)


async def test_a_served_stream_given_up_on_is_recorded_as_the_node_saw_it(tmp_path: Path) -> None:
    """The first resume's deadline was shorter, and fired before the served turn's first block:
    the node charged its fallback. The turn was recorded as cancelled with every block of it,
    untimed, and the next resume -- nothing changed -- handed the node both blocks at once. Its
    read of the first was issued early and took the fallback's position, so the fallback went
    out again under a new key. Found by the seventeenth review."""
    limit: dict[str, float | None] = {"s": None}
    sent: list[str] = []

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(f"charge {amount}")
        return {"charge_id": f"ch_{len(sent)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            async with asyncio.timeout(limit["s"]):
                await session.call_turn(SMALL)  # type: ignore[misc]
        except TimeoutError:
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([lookup_plan, charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, ReadsThenCharges(25)
        ).run(run_id, {})
    await bury_the_dead_process()
    assert sent == ["charge 25"]
    limit["s"] = 0.3
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")), adapter, registry, ReadsThenCharges(99)
        ).resume(run_id)
    await bury_the_dead_process()
    assert sent == ["charge 25", "charge 10.0"], "the setup: the first resume fell back"

    asked_again = ReadsThenCharges(99)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_again).resume(run_id), timeout=30
    )
    assert asked_again.asked == 0
    assert sent == ["charge 25", "charge 10.0"], f"the fallback went out twice: {sent}"
    assert "may have been sent" not in (resumed.error or ""), resumed.error


# -- ask_abandoned asks again only what was abandoned (review 17) --------------------------------

CACHED = replace(SMALL, max_tokens=32)


class Quotes:
    """The recorded run's model never answers the cache check or the first quote, and quotes 25
    with more room. Asked live on a resume: no cached price, 150 at once, and 30."""

    def __init__(self, *, recorded: bool) -> None:
        self.recorded = recorded
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(envelope.max_tokens)
        if self.recorded and envelope.max_tokens in (32, 64):
            await asyncio.Event().wait()
        if envelope.max_tokens == 32:
            yield TextDelta(index=0, text="no cached price")
            yield TurnComplete(response=free_text_turn("no cached price"))
            return
        reply = charge({64: 150.0}.get(envelope.max_tokens or 0, 25.0 if self.recorded else 30.0))
        yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[attr-defined]
        yield TurnComplete(response=reply)  # type: ignore[arg-type]


async def test_ask_abandoned_asks_again_only_the_turn_that_was_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node checks a cache, cancelling the question on a hit; then asks for a quote with a
    deadline, and again with more room when it is slow. Across the crash the cache expired, so
    the resume waits on the cache check and is abandoned. ``--ask-abandoned`` asked every turn
    the recorded node had given up on again, at once -- the quote too, which answered in time
    now, and the node charged a price no run had decided. Found by the seventeenth review."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    cache = {"hit": True}
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        asking = asyncio.create_task(session.call_turn(CACHED))  # type: ignore[misc]
        await asyncio.sleep(0.05)
        if cache["hit"]:
            asking.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asking
        else:
            await asking
        try:
            await asyncio.wait_for(session.call_turn(SMALL), timeout=0.2)  # type: ignore[misc]
        except TimeoutError:
            await session.call_turn(ROOMY)  # type: ignore[misc]
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
            Quotes(recorded=True),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]
    cache["hit"] = False

    live = Quotes(recorded=False)
    plain = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, live).resume(run_id), timeout=20
    )
    assert not plain.ok and TurnAbandoned.__name__ in (plain.error or ""), plain.error
    assert "ask-abandoned" in (plain.error or ""), plain.error
    live = Quotes(recorded=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, live).resume(run_id, ask_abandoned=True),
        timeout=20,
    )
    assert live.asked == [32], f"asked live beyond the abandoned turn: {live.asked}"
    assert resumed.ok, resumed.error
    assert charged == [25.0], f"a charge the recorded run never made: {charged}"


# -- what the operator is told (review 17) --------------------------------------------------------


async def test_a_node_stopped_on_its_way_out_is_reported_by_why_it_was_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node's ``finally`` tried to write after it was stopped; the refusal replaced the
    reason, and the run said only that the write "is not made" -- not which turn was abandoned,
    nor the way past it. Found by the seventeenth review."""
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    db, run_id, adapter, registry, _sent, _cache = await crashed_after_the_cache_hit(tmp_path)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id), timeout=10
    )
    error = resumed.error or ""
    assert "stopped waiting for this turn" in error and "ask-abandoned" in error, error


async def test_a_replay_is_not_told_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from specunode.core import model as model_module

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    journal = Journal(tmp_path / "source.db")
    adapter, registry, _sent, cache = cached_billing()
    result = await scheduler(journal, adapter, registry, SlowFirst(hang=True)).run(new_ulid(), {})
    assert result.ok, result.error
    cache["hit"] = False
    replayed = await asyncio.wait_for(
        scheduler(
            Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
        ).run(new_ulid(), {}),
        timeout=20,
    )
    assert not replayed.ok and TurnAbandoned.__name__ in (replayed.error or "")
    assert "specunode resume" not in (replayed.error or ""), replayed.error


# -- what a run reports while something may be out (review 17) -----------------------------------


async def test_a_dead_letter_that_may_have_been_sent_keeps_the_run_from_success(
    tmp_path: Path,
) -> None:
    """The charge's reply was lost and it was dead-lettered, maybe sent; the code changed, and
    the resume charged something else and reported success over it. Found by the seventeenth
    review."""
    version = {"amount": 25.0}
    sent: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        sent.append(amount)
        if len(sent) == 1:
            raise ConnectionResetError("the charge went out and its reply was lost")
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
    first = await scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).run(run_id, {})
    assert not first.ok
    version["amount"] = 30.0
    resumed = await scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id)
    assert sent == [25.0, 30.0]
    assert not resumed.ok and "never settled" in (resumed.error or ""), resumed.error
    assert "MAY HAVE BEEN SENT" in render_ledger(build_ledger(Journal(db), run_id))


class DiesBeforeTheDeadLetter(Journal):
    async def settle_dispatch(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if kwargs.get("status") == "dead_letter":
            raise Crash("died after marking the claim never sent, before the dead letter")
        return await super().settle_dispatch(*args, **kwargs)


async def test_a_charge_that_never_left_is_not_reported_as_maybe_sent(tmp_path: Path) -> None:
    """The dead process had marked the claim as never sent -- the upstream refused the
    connection -- and the resume failed the run over it, "may have been sent". Found by the
    seventeenth review."""
    version = {"amount": 25.0}
    sent: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        if version["amount"] == 25.0:
            raise ToolDispatchError("connection refused: nothing left", sent="no", retriable=False)
        sent.append(amount)
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
        await scheduler(DiesBeforeTheDeadLetter(db), adapter, registry, SlowFirst(hang=False)).run(
            run_id, {}
        )
    claims = Journal(db).unresolved_dispatches(run_id)
    assert [(c["status"], c["last_outcome"]) for c in claims] == [("in_flight", "not_sent")]
    version["amount"] = 30.0
    resumed = await scheduler(Journal(db), adapter, registry, SlowFirst(hang=False)).resume(run_id)
    assert resumed.ok, resumed.error
    assert "MAY HAVE BEEN SENT" not in render_ledger(build_ledger(Journal(db), run_id))
