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
    with pytest.raises(SchedulerError, match="position rule 1"):
        await driving(Journal(db), adapter, registry, speculation=False).resume(run_id)
    with pytest.raises(PositionRuleMismatch, match="earlier version"):
        replay_of(Journal(db), run_id)
    assert charged == [10.0]
