"""Where a node's calls sit after a model turn that did not complete -- and so their keys.

Found by the twentieth review. The positions a turn's blocks took as they arrived were kept
when the turn failed, or its node gave up on it: how many had arrived by then is a matter of
timing -- a slow disk, a guess being settled -- that a resume reproduces only roughly, so a
fallback charge moved to another position, under a new key, and went out twice; a finished run
failed to replay. A turn that does not complete takes no positions now (``POSITION_RULE`` 2).
And a journal written under the earlier rule, with such a turn in it, is refused, not resumed
into different keys.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from tests.integration.test_crash_then_resume import (
    Crash,
    CrashingJournal,
    before_commit_of,
    bury_the_dead_process,
    replay_of,
)
from tests.integration.test_speculation import FixedDrafter

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
    ToolUseBlock,
    ToolUseComplete,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler, SchedulerError
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.journal.replay import PositionRuleMismatch

TURN = RequestEnvelope(
    model="scripted",
    messages=(Message(role="user", content=(TextBlock(text="order for cus-1"),)),),
    max_tokens=256,
    stream=True,
)


class LooksUpNotesThenStalls:
    """Block 0 (a read) at 10 ms, block 1 (a write) at 100 ms, then nothing more."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return ModelResponse(
            model="scripted",
            content=(ToolUseBlock(id="c0", name="lookup_plan", args={"customer_id": "cus-1"}),),
            stop_reason="tool_use",
        )

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(0.01)
        yield ToolUseComplete(
            index=0,
            block=ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"}),
        )
        await asyncio.sleep(0.09)
        yield ToolUseComplete(
            index=1,
            block=ToolUseBlock(id="t1", name="note_order", args={"customer_id": "cus-1"}),
        )
        await asyncio.Event().wait()  # the rest of the reply never comes


class SlowToSettleAGuess(CrashingJournal):
    """A disk slow to take a guess's resolution (400 ms)."""

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "policy_event" and payload.get("event") == "alpha_observed":
            await asyncio.sleep(0.4)
        return await super().append_async(run_id, kind, payload)


class SlowToAsk(Journal):
    """A disk slower to take a question (250 ms) than the crashed run's was."""

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "model_request":
            await asyncio.sleep(0.25)
        return await super().append_async(run_id, kind, payload)


def billing(charged: list[float]) -> tuple[PlainAdapter, object]:
    """The node asks the model, and past 0.3 s charges the standard price instead."""

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def note_order(customer_id: str) -> JsonValue:
        return {"noted": True}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        # The model is too slow today past 0.3 s: charge the standard price.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(session.call_turn(TURN), timeout=0.3)  # type: ignore[misc]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([lookup_plan, note_order, charge_card])


def driving(journal: Journal, adapter: object, registry: object, *, speculation: bool) -> Scheduler:
    return Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),  # type: ignore[arg-type]
        target=JournaledModel(LooksUpNotesThenStalls(), journal, provider="scripted"),
        policy=Policy(speculation=speculation, max_speculation_depth=3),
        # A guess at the call after the read: a second read, which the write then contradicts.
        predictor=(
            FixedDrafter(ToolCall("lookup_plan", {"customer_id": "cus-2"}))  # type: ignore[arg-type]
            if speculation
            else None
        ),
    )


@pytest.mark.parametrize("resume_speculates", [True, False])
async def test_a_deadline_inside_a_guess_does_not_move_the_fallback(
    tmp_path: Path, resume_speculates: bool
) -> None:
    """The node's deadline fired while a guess was being settled on a slow disk, before the
    turn's write block took its position; on resume the block was handed over in time, took
    it, and the fallback charge moved one along -- a new key, a second charge."""
    db = tmp_path / "source.db"
    charged: list[float] = []
    adapter, registry = billing(charged)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            driving(
                SlowToSettleAGuess(db, before_commit_of("bill#0")),
                adapter,
                registry,
                speculation=True,
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    assert charged == [10.0]
    adapter, registry = billing(charged)
    resumed = await asyncio.wait_for(
        driving(Journal(db), adapter, registry, speculation=resume_speculates).resume(run_id),
        timeout=60,
    )
    assert resumed.ok, resumed.error
    assert charged == [10.0], f"the fallback charge went out twice: {charged}"


async def test_a_slower_resume_does_not_move_the_fallback(tmp_path: Path) -> None:
    """No guessing at all: the crashed run was handed both blocks before its deadline, the
    resume -- its question slower to write -- only the first, and the fallback sat elsewhere."""
    db = tmp_path / "source.db"
    charged: list[float] = []
    adapter, registry = billing(charged)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            driving(
                CrashingJournal(db, before_commit_of("bill#0")),
                adapter,
                registry,
                speculation=False,
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    adapter, registry = billing(charged)
    resumed = await asyncio.wait_for(
        driving(SlowToAsk(db), adapter, registry, speculation=False).resume(run_id), timeout=60
    )
    assert resumed.ok, resumed.error
    assert charged == [10.0], f"the fallback charge went out twice: {charged}"


async def test_a_finished_run_whose_deadline_fell_inside_a_guess_replays(tmp_path: Path) -> None:
    """The run finished fine; its deadline fell while a guess was settled, so the next node
    started one position earlier than the replay -- which settled the guess quickly -- put it,
    and found no recorded turn there."""
    charged: list[float] = []

    def graph() -> tuple[PlainAdapter, object]:
        billed, registry = billing(charged)

        @node(name="confirm")
        async def confirm(session: RunSession) -> Decision:
            assert session.model is not None
            await session.model.complete(TURN)
            session.state["confirmed"] = True
            return ToolCall("lookup_plan", {})

        def route(state: Mapping[str, JsonValue]) -> str | None:
            if not state.get("billed"):
                return "bill"
            return None if state.get("confirmed") else "confirm"

        return PlainAdapter.of([*billed.node_fns.values(), confirm], route), registry

    adapter, registry = graph()
    db = tmp_path / "source.db"
    run_id = new_ulid()
    result = await asyncio.wait_for(
        driving(
            SlowToSettleAGuess(db, lambda kind, payload: False), adapter, registry, speculation=True
        ).run(run_id, {}),
        timeout=30,
    )
    assert result.ok, result.error
    adapter, registry = graph()
    replay_journal = Journal(tmp_path / "replay.db")
    replayed = await asyncio.wait_for(
        Scheduler(
            graph=adapter,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            journal=replay_journal,
            buffer=StoreBuffer(journal=replay_journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),  # type: ignore[arg-type]
            target=replay_of(Journal(db), run_id),
            policy=Policy(speculation=False),
        ).run(new_ulid(), {}),
        timeout=60,
    )
    assert replayed.ok, replayed.error


async def test_a_guess_settled_while_its_node_gives_up_is_counted_once(tmp_path: Path) -> None:
    """A cancel inside a guess's squash reached it again from the failure path: counted twice
    in flight and in alpha -- and, let go of part-way, its squash was not journaled."""
    charged: list[float] = []
    adapter, registry = billing(charged)
    journal = SlowToSettleAGuess(tmp_path / "j.db", lambda kind, payload: False)
    driver = driving(journal, adapter, registry, speculation=True)
    run_id = new_ulid()
    result = await asyncio.wait_for(driver.run(run_id, {}), timeout=30)
    assert result.ok, result.error
    observed = [
        e.payload.get("branch_id")
        for e in journal.read(run_id, kinds=["policy_event"])
        if e.payload.get("event") == "alpha_observed"
    ]
    assert driver.budget.inflight_branches == 0
    assert driver.budget.window.samples == 1
    assert len(observed) == 1, observed


async def test_a_journal_from_another_position_rule_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written by a version that placed calls after an incomplete turn differently, a crashed
    run resumed under this one put its fallback charge at another position, and it went out
    twice with the run reporting success. Found by the twentieth review."""
    from specunode.core import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "POSITION_RULE", 1)
    db = tmp_path / "source.db"
    charged: list[float] = []
    adapter, registry = billing(charged)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            driving(
                CrashingJournal(db, before_commit_of("bill#0")),
                adapter,
                registry,
                speculation=False,
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    monkeypatch.undo()
    adapter, registry = billing(charged)
    with pytest.raises(SchedulerError, match="earlier version"):
        await driving(Journal(db), adapter, registry, speculation=False).resume(run_id)
    with pytest.raises(PositionRuleMismatch, match="earlier version"):
        replay_of(Journal(db), run_id)
    assert charged == [10.0]


# -- the twenty-first review: the rewind itself ------------------------------------------------


class LooksUpThenFails:
    """Block 0 (a read) at 10 ms; at 100 ms the model says it is overloaded."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return ModelResponse(model="scripted", content=(TextBlock(text="ok"),))

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(0.01)
        yield ToolUseComplete(
            index=0,
            block=ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "cus-1"}),
        )
        await asyncio.sleep(0.09)
        raise ModelError("overloaded")


def failing_billing(charged: list[float]) -> tuple[PlainAdapter, object]:
    """Ask the model; if it fails or is slower than 0.3 s, charge the standard price."""

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        with contextlib.suppress(TimeoutError, ModelError):
            await asyncio.wait_for(session.call_turn(TURN), timeout=0.3)  # type: ignore[misc]
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([lookup_plan, charge_card])


def failing(journal: Journal, adapter: object, registry: object, *, speculation: bool) -> Scheduler:
    return Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),  # type: ignore[arg-type]
        target=JournaledModel(LooksUpThenFails(), journal, provider="scripted"),
        policy=Policy(speculation=speculation, max_speculation_depth=3),
        predictor=(
            FixedDrafter(ToolCall("lookup_plan", {"customer_id": "cus-2"}))  # type: ignore[arg-type]
            if speculation
            else None
        ),
    )


@pytest.mark.parametrize("resume_speculates", [True, False])
async def test_a_deadline_during_a_failed_turns_cleanup_does_not_keep_its_positions(
    tmp_path: Path, resume_speculates: bool
) -> None:
    """The turn failed with a guess open; the node's deadline fired while the guess was being
    squashed, and the rewind -- after that await -- never ran: the fallback charge sat one
    place on, and went out again on resume under another key. Found by the twenty-first
    review."""
    db = tmp_path / "source.db"
    charged: list[float] = []
    adapter, registry = failing_billing(charged)
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            failing(
                SlowToSettleAGuess(db, before_commit_of("bill#0")),
                adapter,
                registry,
                speculation=True,
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    assert charged == [10.0]
    adapter, registry = failing_billing(charged)
    resumed = await asyncio.wait_for(
        failing(Journal(db), adapter, registry, speculation=resume_speculates).resume(run_id),
        timeout=60,
    )
    assert resumed.ok, resumed.error
    assert charged == [10.0], f"the fallback charge went out twice: {charged}"


async def test_two_turns_at_once_are_refused(tmp_path: Path) -> None:
    """Two call_turns at once took their positions as they happened to interleave -- one way in
    the run, another on the resume -- and a charge went out twice. The second is refused."""
    from specunode.core.scheduler import SchedulerError as Refused

    refused: list[str] = []

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        first = asyncio.create_task(session.call_turn(TURN))  # type: ignore[misc]
        await asyncio.sleep(0)
        try:
            await session.call_turn(TURN)  # type: ignore[misc]
        except Refused as exc:
            refused.append(str(exc))
        with contextlib.suppress(BaseException):
            await first
        session.state["billed"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    journal = Journal(tmp_path / "j.db")
    result = await Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry_of([]),
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry_of([]), max_attempts=1, base_delay_ms=0.5),
        target=JournaledModel(LooksUpThenFails(), journal, provider="scripted"),
        policy=Policy(speculation=False),
    ).run(new_ulid(), {})
    assert result.ok, result.error
    assert refused and "another was under way" in refused[0]


async def test_a_write_while_a_turn_is_under_way_is_refused(tmp_path: Path) -> None:
    """A write made while the node's turn was still being asked took a position the failed
    turn then gave back, and the next write landed on it under the same key. Found by the
    twenty-first review."""
    from specunode.core.scheduler import SchedulerError as Refused

    sent: list[str] = []
    refused: list[str] = []

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def send_sms(text: str) -> JsonValue:
        sent.append(text)
        return {"sms_id": f"sms_{len(sent)}"}

    @node(name="notify")
    async def notify(session: RunSession) -> Decision:
        turn = asyncio.create_task(session.call_turn(TURN))  # type: ignore[misc]
        await asyncio.sleep(0)
        try:
            await session.call_tool("send_sms", {"text": "on it"})
        except Refused as exc:
            refused.append(str(exc))
        with contextlib.suppress(BaseException):
            await turn
        await session.call_tool("send_sms", {"text": "done"})
        session.state["done"] = True
        return ToolCall("send_sms", {})

    adapter = PlainAdapter.of([notify], lambda s: None if s.get("done") else "notify")
    registry = registry_of([lookup_plan, send_sms])
    journal = Journal(tmp_path / "j.db")
    result = await Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
        target=JournaledModel(LooksUpThenFails(), journal, provider="scripted"),
        policy=Policy(speculation=False),
    ).run(new_ulid(), {})
    assert result.ok, result.error
    assert refused and "under way" in refused[0], refused
    assert sent == ["done"]


class SummarisesThenFails:
    """Looks something up at 5 ms, then fails once ``fail_now`` is set."""

    def __init__(self, fail_now: asyncio.Event) -> None:
        self.fail_now = fail_now

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(0.005)
        yield ToolUseComplete(
            index=0, block=ToolUseBlock(id="t0", name="lookup_plan", args={"customer_id": "c"})
        )
        await self.fail_now.wait()
        raise ModelError("overloaded")


class SlowToRetirePlan(CrashingJournal):
    """plan#0's retirement is slow to write; its left-over turn fails while it is written."""

    def __init__(self, path: Path, fail_now: asyncio.Event) -> None:
        super().__init__(path, before_commit_of("bill#0"))
        self.fail_now = fail_now
        self.owners: dict[str, str] = {}

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "branch_forked":
            self.owners[str(payload.get("branch_id"))] = str(payload.get("node_id"))
        if (
            kind == "branch_resolved"
            and payload.get("status") == "retired"
            and self.owners.get(str(payload.get("branch_id"))) == "plan#0"
        ):
            self.fail_now.set()
            await asyncio.sleep(0.1)
        return await super().append_async(run_id, kind, payload)


async def test_a_turn_left_running_does_not_move_a_retired_nodes_position(tmp_path: Path) -> None:
    """A node left a turn running and retired; the turn failed as the retirement was written,
    and the run carried on from the cursor it had moved, not the one journaled -- so the next
    node's charge sat elsewhere on the resume, under another key. Found by the twenty-first
    review."""
    charged: list[str] = []

    def build() -> tuple[PlainAdapter, object]:
        @tool(effect="read")
        async def lookup_plan(customer_id: str) -> JsonValue:
            return {"plan": "basic"}

        @tool(effect="write", idempotent=False)
        async def charge_card(customer_id: str, amount: float) -> JsonValue:
            charged.append(f"charge {amount}")
            return {"charge_id": "ch"}

        @node(name="plan")
        async def plan(session: RunSession) -> Decision:
            asyncio.get_running_loop().create_task(session.call_turn(TURN))  # type: ignore[misc]
            await asyncio.sleep(0.05)
            session.state["planned"] = True
            return ToolCall("lookup_plan", {})

        @node(name="bill")
        async def bill(session: RunSession) -> Decision:
            await session.call_tool("charge_card", {"customer_id": "c", "amount": 10.0})
            session.state["billed"] = True
            return ToolCall("charge_card", {})

        def route(state: Mapping[str, JsonValue]) -> str | None:
            if not state.get("planned"):
                return "plan"
            return None if state.get("billed") else "bill"

        return PlainAdapter.of([plan, bill], route), registry_of([lookup_plan, charge_card])

    def driver(journal: Journal, model: object) -> Scheduler:
        adapter, registry = build()
        return Scheduler(
            graph=adapter,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),  # type: ignore[arg-type]
            target=JournaledModel(model, journal, provider="scripted"),  # type: ignore[arg-type]
            policy=Policy(speculation=False),
        )

    db = tmp_path / "source.db"
    fail_now = asyncio.Event()
    run_id = new_ulid()
    with pytest.raises(Crash):
        await asyncio.wait_for(
            driver(SlowToRetirePlan(db, fail_now), SummarisesThenFails(fail_now)).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    resumed = await asyncio.wait_for(
        driver(Journal(db), SummarisesThenFails(asyncio.Event())).resume(run_id), timeout=60
    )
    assert resumed.ok, resumed.error
    assert charged == ["charge 10.0"], f"the charge went out twice: {charged}"


async def test_a_run_is_judged_by_the_rule_each_failed_turn_was_recorded_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started under an earlier rule and resumed under this one, a run whose failed turn this
    version recorded was refused -- and the earlier version, followed as advised, charged
    twice. A turn is judged by the rule in force when it was recorded. Found by the
    twenty-first review."""
    from specunode.core import scheduler as scheduler_module

    charged: list[float] = []
    db = tmp_path / "source.db"
    run_id = new_ulid()
    # The earlier rule starts the run, and dies as the node asks its first question.
    monkeypatch.setattr(scheduler_module, "POSITION_RULE", 1)
    adapter, registry = failing_billing(charged)
    with pytest.raises(Crash):
        await asyncio.wait_for(
            failing(
                CrashingJournal(db, lambda kind, payload: kind == "model_request"),
                adapter,
                registry,
                speculation=False,
            ).run(run_id, {}),
            timeout=30,
        )
    await bury_the_dead_process()
    monkeypatch.undo()
    # This rule resumes it: the turn fails with a read in it, the fallback is charged, and the
    # process dies before it commits.
    adapter, registry = failing_billing(charged)
    with pytest.raises(Crash):
        await asyncio.wait_for(
            failing(
                CrashingJournal(db, before_commit_of("bill#0")),
                adapter,
                registry,
                speculation=False,
            ).resume(run_id),
            timeout=30,
        )
    await bury_the_dead_process()
    assert charged == [10.0]
    adapter, registry = failing_billing(charged)
    resumed = await asyncio.wait_for(
        failing(Journal(db), adapter, registry, speculation=False).resume(run_id), timeout=60
    )
    assert resumed.ok, resumed.error
    assert charged == [10.0]


async def test_a_failed_whole_answer_under_an_earlier_rule_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn asked with ``complete()`` never takes a position, whatever rule recorded it: a
    run whose only failed turn was one was refused all the same. Found by the twenty-first
    review."""
    from specunode.core import scheduler as scheduler_module

    class CutOff:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            return ModelResponse(
                model="scripted",
                content=(ToolUseBlock(id="t", name="lookup_plan", args={"customer_id": "cus-1"}),),
                stop_reason="max_tokens",
            )

        def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            raise NotImplementedError

    charged: list[float] = []

    @tool(effect="read")
    async def lookup_plan(customer_id: str) -> JsonValue:
        return {"plan": "basic"}

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": "ch"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        assert session.model is not None
        with contextlib.suppress(ModelError):
            await session.model.complete(replace(TURN, stream=False))
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 10.0})
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    registry = registry_of([lookup_plan, charge_card])
    db = tmp_path / "source.db"
    run_id = new_ulid()
    monkeypatch.setattr(scheduler_module, "POSITION_RULE", 1)
    with pytest.raises(Crash):
        await Scheduler(
            graph=adapter,  # type: ignore[arg-type]
            registry=registry,
            journal=(journal := CrashingJournal(db, before_commit_of("bill#0"))),
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(CutOff(), journal, provider="scripted"),
            policy=Policy(speculation=False),
        ).run(run_id, {})
    await bury_the_dead_process()
    monkeypatch.undo()
    replay_of(Journal(db), run_id)  # not refused
    resumed = await Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,
        journal=(again := Journal(db)),
        buffer=StoreBuffer(journal=again, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
        target=JournaledModel(CutOff(), again, provider="scripted"),
        policy=Policy(speculation=False),
    ).resume(run_id)
    assert resumed.ok, resumed.error
    assert charged == [10.0]
