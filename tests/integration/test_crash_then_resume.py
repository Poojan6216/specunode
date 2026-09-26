"""A process dies at a chosen journal entry; a new one resumes the run; replay reads it back.

Found by an independent adversarial review of parallel nodes, the agent loop and replay keying,
each with a failing test first. A crash is simulated precisely: the journal raises a
``BaseException`` just before it would durably append one chosen entry, so nothing after that
point is written and nothing is cleaned up. The same ``World`` survives, playing the durable
world a real crash leaves behind, and a fresh ``Scheduler`` resumes on the same database.

The property every test here holds a resume to: nothing that already went out goes out again,
and nothing the committed run did not decide is reproduced.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from examples.incident_agent.agent import build as build_incident
from examples.incident_agent.agent import scripted_replies
from examples.support_agent.agent import build as build_support

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import ToolRegistry
from specunode.core.graph import RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolUseComplete,
    TurnComplete,
    decisions_of,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.journal.replay import ReplayModel, recover
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world


class Crash(BaseException):
    """The process dying: nothing after this point is written, nothing is cleaned up."""


class CrashingJournal(Journal):
    """A journal whose process dies just before it would append one chosen entry."""

    def __init__(
        self, path: Path, should_crash: Callable[[str, Mapping[str, JsonValue]], bool]
    ) -> None:
        super().__init__(path)
        self._should_crash = should_crash
        self.crashed = False

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        # Once dead, dead: a real crash writes nothing more, whatever is still scheduled.
        if self.crashed or self._should_crash(kind, payload):
            self.crashed = True
            raise Crash(f"process died before appending {kind}")
        return await super().append_async(run_id, kind, payload)


async def bury_the_dead_process() -> None:
    """End whatever the crashed run left scheduled, as the process's death would have.

    It shares the test's event loop, so its tasks outlive it here; left alone they are
    finalised later, outside their own context. Their journal writes are refused above.
    """
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    for task in leftovers:
        task.cancel()
    await asyncio.gather(*leftovers, return_exceptions=True)


def before_commit_of(node_id: str) -> Callable[[str, Mapping[str, JsonValue]], bool]:
    """Die just before ``node_id``'s state delta is journaled: after its effects went out."""
    owners: dict[str, str] = {}

    def predicate(kind: str, payload: Mapping[str, JsonValue]) -> bool:
        if kind == "branch_forked":
            owners[str(payload.get("branch_id"))] = str(payload.get("node_id"))
            return False
        return kind == "state_delta_applied" and owners.get(str(payload.get("branch_id"))) == (
            node_id
        )

    return predicate


def scheduler(
    journal: Journal, adapter: object, registry: object, model: object, **policy: Any
) -> Scheduler:
    return Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),  # type: ignore[arg-type]
        target=JournaledModel(model, journal, provider="scripted"),  # type: ignore[arg-type]
        policy=Policy(**{"speculation": False, **policy}),
    )


def replay_of(journal: Journal, run_id: str) -> ReplayModel:
    return ReplayModel(
        journal=journal, run_id=run_id, retired_branches=recover(journal, run_id).retired_branches
    )


# -- replay of a resumed run --------------------------------------------------------------------


def charge(amount: float) -> object:
    return tool_turn(("charge_card", {"customer_id": "cus-1", "amount": amount}))


async def test_replay_of_a_resumed_run_reproduces_the_run_that_was_kept(tmp_path: Path) -> None:
    """The model said 25; the process died before ``decide`` retired; resumed, it said 30.

    Nothing had gone out on the 25 -- ``decide`` only decides -- so the resume asks again rather
    than being served it, and both answers sit in the journal under one node and one position.
    Replay used to serve the dead attempt's, reproducing a charge the committed run never made.
    """
    db = tmp_path / "source.db"
    world = standard_world()
    adapter, registry = build_support(world)
    run_id = new_ulid()
    inputs = {"customer_id": "cus-1"}
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("decide#0")),
            adapter,
            registry,
            ScriptedModel(turns=[charge(25.0)]),  # type: ignore[list-item]
        ).run(run_id, inputs)
    await bury_the_dead_process()

    adapter, registry = build_support(world)
    resumed = await scheduler(
        Journal(db),
        adapter,
        registry,
        ScriptedModel(turns=[charge(30.0)]),  # type: ignore[list-item]
    ).resume(run_id)
    assert resumed.ok, resumed.error
    assert [row["amount"] for row in world.tables["charges"].values()] == [30.0]

    adapter, registry = build_support(standard_world())
    replayed = await scheduler(
        Journal(tmp_path / "replay.db"), adapter, registry, replay_of(Journal(db), run_id)
    ).run(new_ulid(), inputs)
    assert replayed.ok, replayed.error
    assert replayed.state["decided"] == resumed.state["decided"]


def build_billing(world: World) -> tuple[object, object]:
    """One node that asks the model what to charge and charges it, in the same step."""

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        return await world.charge_card(customer_id=customer_id, amount=amount)

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        envelope = RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text="bill cus-1"),)),),
            max_tokens=64,
        )
        decision = decisions_of(await session.model.complete(envelope))[0]
        assert isinstance(decision, ToolCall)
        await session.call_tool(decision.name, dict(decision.args))
        session.state["billed"] = True
        return decision

    def route(state: Mapping[str, JsonValue]) -> str | None:
        return None if state.get("billed") else "bill"

    return PlainAdapter.of([bill], route), registry_of([charge_card])


async def test_a_decision_that_sent_something_is_served_to_the_resume(tmp_path: Path) -> None:
    """The model said 25 and the charge went out; the process died before ``bill`` retired.

    Asked again, the model would say 30. The resume is served the journaled 25 instead, derives
    the charge's key again, and the dedupe table recognises the charge as sent: one charge. A
    resume that asked again charged 30 on top of the 25 -- a second, different call, under a
    key nothing could connect to the first.
    """
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
    assert [row["amount"] for row in world.tables["charges"].values()] == [25.0]

    adapter, registry = build_billing(world)
    changed_its_mind = ScriptedModel(turns=[charge(30.0)])  # type: ignore[list-item]
    resumed = await scheduler(Journal(db), adapter, registry, changed_its_mind).resume(run_id)
    assert resumed.ok, resumed.error
    assert changed_its_mind.calls == 0, "the resume asked for a decision that had sent a charge"
    assert [row["amount"] for row in world.tables["charges"].values()] == [25.0]


async def test_replay_of_a_resumed_conversation_is_not_refused(tmp_path: Path) -> None:
    """A conversation node killed after two turns and resumed records 2 + 4 turns under one key.

    Served in journal order, the replayed node's third request was compared with the resumed
    process's first and refused, although nothing had changed.
    """

    def dies_on_first_restart(registry: ToolRegistry) -> ToolRegistry:
        spec = registry.get("restart_job")
        died = {"once": False}

        async def restart(**kwargs: Any) -> JsonValue:
            if not died["once"]:
                died["once"] = True
                raise Crash("SIGKILL during the first restart")
            return await spec.fn(**kwargs)  # type: ignore[no-any-return]

        out = ToolRegistry()
        for name, declared in registry.declared().items():
            out.register(replace(declared, fn=restart) if name == "restart_job" else declared)
        return out

    db = tmp_path / "loop.db"
    world = standard_world()
    adapter, registry = build_incident(world, "parallel")
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            Journal(db),
            adapter,
            dies_on_first_restart(registry),  # type: ignore[arg-type]
            ScriptedModel(turns=scripted_replies("parallel")),
        ).run(run_id, {})
    await bury_the_dead_process()

    adapter, registry = build_incident(world, "parallel")
    resumed = await scheduler(
        Journal(db), adapter, registry, ScriptedModel(turns=scripted_replies("parallel"))
    ).resume(run_id)
    assert resumed.ok, resumed.error

    adapter, registry = build_incident(standard_world(), "parallel")
    replayed = await scheduler(
        Journal(tmp_path / "replay.db"), adapter, registry, replay_of(Journal(db), run_id)
    ).run(new_ulid(), {})
    assert replayed.ok, f"replay refused a faithful re-run: {replayed.error}"
    assert replayed.state["turns"] == resumed.state["turns"]


# -- a crash around a parallel group ------------------------------------------------------------


def two_lanes_then_finish(world: Any, *, grouped: bool) -> tuple[object, object]:
    """Lane ``a`` makes two writes and ``b`` one, so they end at different positions."""
    from specunode.core.decision import FreeText
    from specunode.core.effects import EffectClass
    from specunode.integrations.plain import PlainAdapter, node, registry_of, tool

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)  # type: ignore[no-any-return]

    @node(name="a")
    async def a(session: Any) -> Any:
        await session.call_tool("post_summary", {"channel": "#a", "text": "1"})
        await session.call_tool("post_summary", {"channel": "#a", "text": "2"})
        session.state["done:a"] = True
        return FreeText.of("a")

    @node(name="b")
    async def b(session: Any) -> Any:
        await session.call_tool("post_summary", {"channel": "#b", "text": "1"})
        session.state["done:b"] = True
        return FreeText.of("b")

    @node(name="finish")
    async def finish(session: Any) -> Any:
        await session.call_tool("post_summary", {"channel": "#finish", "text": "done"})
        session.state["finished"] = True
        return FreeText.of("finish")

    def route(state: Mapping[str, JsonValue]) -> str | list[str] | None:
        # The fan-out router's shape: name whatever is still pending.
        pending = [n for n in ("a", "b") if f"done:{n}" not in state]
        if pending:
            return pending if grouped else pending[0]
        return None if state.get("finished") else "finish"

    adapter = PlainAdapter.of([a, b, finish], route)  # type: ignore[list-item]
    return adapter, registry_of([post_summary])  # type: ignore[list-item]


def posts_to(world: Any, channel: str) -> int:
    return sum(1 for row in world.tables["messages"].values() if row.get("channel") == channel)


@pytest.mark.parametrize("grouped", [False, True], ids=["one-at-a-time", "parallel-group"])
@pytest.mark.parametrize(
    "dies_before_commit_of",
    [
        # Mid-group, before any lane retired.
        "a#0",
        # Mid-group, after ``a`` retired and after ``b``'s post went out. A resume used to ask
        # the router again, which named only ``b`` -- as ``b#1``, at a new position.
        "b#0",
        # After the whole group, in the node that follows it. The last lane journaled its own
        # cursor rather than the group's, so ``finish`` resumed at a lower position.
        "finish#0",
    ],
)
async def test_a_crash_around_a_parallel_group_sends_nothing_twice(
    tmp_path: Path, grouped: bool, dies_before_commit_of: str
) -> None:
    world = standard_world()
    adapter, registry = two_lanes_then_finish(world, grouped=grouped)
    run_id = new_ulid()
    db = tmp_path / "run.db"
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of(dies_before_commit_of)),
            adapter,
            registry,
            ScriptedModel(turns=[]),
        ).run(run_id, {})
    await bury_the_dead_process()

    adapter, registry = two_lanes_then_finish(world, grouped=grouped)  # a new process
    resumed = await scheduler(Journal(db), adapter, registry, ScriptedModel(turns=[])).resume(
        run_id
    )
    assert resumed.ok, resumed.error
    sent = {channel: posts_to(world, channel) for channel in ("#a", "#b", "#finish")}
    assert sent == {"#a": 2, "#b": 1, "#finish": 1}, sent


async def test_recovery_reports_the_group_a_crash_interrupted(tmp_path: Path) -> None:
    world = standard_world()
    adapter, registry = two_lanes_then_finish(world, grouped=True)
    run_id = new_ulid()
    db = tmp_path / "run.db"
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("b#0")), adapter, registry, ScriptedModel(turns=[])
        ).run(run_id, {})
    await bury_the_dead_process()
    group = recover(Journal(db), run_id).open_group
    assert group is not None
    assert [node_id for _name, _path, node_id in group.lanes] == ["a#0", "b#0"]
    assert group.retired == frozenset({"a#0"})
    assert group.base_state == {}, "the lanes forked from the state before either committed"
    assert group.claimed == {"done:a": "a#0"}


# -- a crash while a write is in flight ---------------------------------------------------------


@pytest.mark.parametrize("when", ["request_lost", "reply_lost"])
@pytest.mark.parametrize("at", range(1, 8))
async def test_a_crash_at_any_write_never_sends_it_twice(at: int, when: str) -> None:
    """Pulled at every write, before the request left and after its reply was lost: the
    runtime never sends an effect twice. Without a way to ask the upstream it stops for a
    human; with one it finishes, each effect in the world once, on the upstream's own answer."""
    from bench.offline.run_crash_safety import one

    held = await one("specunode", at, when)
    assert held["duplicated"] == {} and held["outcome"] == "held", held
    assert "dead-lettered" in str(held["why"])
    asked = await one("specunode_reconcile", at, when)
    assert asked["outcome"] == "exact", asked
    assert asked["receipts_that_lie"] == []


async def test_an_upstream_that_cannot_be_asked_is_a_dead_letter_not_a_guess(
    tmp_path: Path,
) -> None:
    """If asking fails, the answer is unknown -- and unknown is a human's call, as before."""
    from bench.offline.run_crash_safety import Plug, SpecuNodeReconcile

    class Unreachable(SpecuNodeReconcile):
        def _asker(self, tool_name: str, reply: Any) -> Any:
            async def reconcile(key: str, args: Mapping[str, JsonValue]) -> JsonValue:
                raise ConnectionError("the upstream's lookup API is down")

            return reconcile

    world = standard_world()
    plug = Plug(at=1, when="reply_lost")
    runner = Unreachable(world, plug, tmp_path)
    with pytest.raises(BaseException, match="took effect"):
        await runner.first()
    await bury_the_dead_process()
    assert await runner.restart() is False
    assert "the upstream's lookup API is down" in str(runner.why) or "dead-lettered" in str(
        runner.why
    )
    assert len(world.tables["charges"]) == 1, "the charge was sent again on a guess"


async def test_a_run_a_resume_finished_is_not_reported_as_still_resumable(tmp_path: Path) -> None:
    """The dead process's attempt at the interrupted node was left "confirmed but not retired",
    so ``specunode status`` told the operator a finished run was still resumable."""
    from bench.offline.run_crash_safety import Plug, SpecuNodeReconcile

    world = standard_world()
    runner = SpecuNodeReconcile(world, Plug(at=1, when="reply_lost"), tmp_path)
    with pytest.raises(BaseException, match="took effect"):
        await runner.first()
    await bury_the_dead_process()
    interrupted = recover(Journal(tmp_path / "journal.db"), runner.run_id)
    assert interrupted.resumable and interrupted.confirmed_not_retired, "nothing to resume?"
    assert await runner.restart() is True
    after = recover(Journal(tmp_path / "journal.db"), runner.run_id)
    assert after.finished
    assert after.confirmed_not_retired == ()
    assert not after.resumable


# -- a turn that failed, and was asked again ---------------------------------------------------

SMALL = RequestEnvelope(
    model="scripted",
    max_tokens=64,
    stream=True,
    messages=(Message(role="user", content=(TextBlock(text="Charge cus-1 for the plan"),)),),
)
ROOMY = replace(SMALL, max_tokens=4096)


class FailsTheFirstQuestion:
    """Fails the question asked with little room -- cut off mid-call, or overloaded -- and
    answers the one asked with more with a charge of ``amount``."""

    def __init__(self, amount: float, failure: str) -> None:
        self.amount = amount
        self.failure = failure
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(envelope.max_tokens)
        if envelope.max_tokens != 64:
            reply = charge(self.amount)
        elif self.failure == "overloaded":
            raise ModelError("overloaded_error")
        else:  # "amount": 25.0 cut off after the 2
            reply = replace(charge(2.0), stop_reason="max_tokens")  # type: ignore[type-var]
        yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[attr-defined]
        yield TurnComplete(response=reply)  # type: ignore[arg-type]


def asks_again() -> tuple[PlainAdapter, ToolRegistry, list[float]]:
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await session.call_turn(SMALL)  # type: ignore[misc]
        except ModelError:  # as the refusal says: give the model room, and ask again
            await session.call_turn(ROOMY)  # type: ignore[misc]
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card]), charged


@pytest.mark.parametrize("failure", ["cut_off", "overloaded"])
async def test_a_turn_that_failed_and_was_asked_again_is_served_as_it_went(
    tmp_path: Path, failure: str
) -> None:
    """The node's first question failed -- a reply cut off mid-call, or the model overloaded
    -- so it asked again, with more room, and the charge went out; then the process died.
    The failed question had no outcome on disk, so on resume it matched nothing, and every
    question the node asked after it went to the live model: one that had changed its mind
    charged a second time. The failure is recorded now, and served again. Found by the
    twelfth review."""
    db = tmp_path / "source.db"
    adapter, registry, charged = asks_again()
    run_id = new_ulid()
    with pytest.raises(Crash):
        await scheduler(
            CrashingJournal(db, before_commit_of("bill#0")),
            adapter,
            registry,
            FailsTheFirstQuestion(25.0, failure),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]

    changed_its_mind = FailsTheFirstQuestion(30.0, failure)
    resumed = await scheduler(Journal(db), adapter, registry, changed_its_mind).resume(run_id)
    assert resumed.ok, resumed.error
    assert changed_its_mind.asked == [], "the resume asked the model what the journal answers"
    assert charged == [25.0], "a second, different charge went out"


async def test_replay_reproduces_a_turn_that_failed(tmp_path: Path) -> None:
    """Replay refused the faithful re-run of a node that asked again: its first question was
    matched with the answer to the second."""
    journal = Journal(tmp_path / "source.db")
    adapter, registry, charged = asks_again()
    result = await scheduler(
        journal, adapter, registry, FailsTheFirstQuestion(25.0, "cut_off")
    ).run(new_ulid(), {})
    assert result.ok, result.error

    adapter, registry, _again = asks_again()
    replayed = await scheduler(
        Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
    ).run(new_ulid(), {})
    assert replayed.ok, replayed.error
    assert charged == [25.0] and _again == [25.0]


# -- a turn the node stopped waiting for -------------------------------------------------------


class AnswersLate:
    """Asked with little room, it is slow (``slow``) or answers at once; asked with more, it
    answers with a charge of ``amount``."""

    def __init__(self, amount: float, *, slow: bool) -> None:
        self.amount = amount
        self.slow = slow
        self.asked: list[int | None] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.asked.append(envelope.max_tokens)
        if envelope.max_tokens == 64:
            if self.slow:
                await asyncio.Event().wait()
            reply = charge(150.0)
        else:
            reply = charge(self.amount)
        yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[attr-defined]
        yield TurnComplete(response=reply)  # type: ignore[arg-type]


class SlowToWriteAnswers(CrashingJournal):
    """The first answer takes a while to reach the disk -- an fsync under load."""

    def __init__(self, path: Path, should_crash: Callable[[str, Mapping[str, JsonValue]], bool]):
        super().__init__(path, should_crash)
        self.slowed = False

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        if kind == "model_response" and not self.slowed:
            self.slowed = True
            await asyncio.sleep(0.4)
        return await super().append_async(run_id, kind, payload)


def times_out() -> tuple[PlainAdapter, ToolRegistry, list[float]]:
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        try:
            await asyncio.wait_for(session.call_turn(SMALL), timeout=0.2)  # type: ignore[misc]
        except TimeoutError:  # the model is slow: ask again, with more room
            await session.call_turn(ROOMY)  # type: ignore[misc]
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    adapter = PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill")
    return adapter, registry_of([charge_card]), charged


@pytest.mark.parametrize("where", ["while_the_model_thinks", "while_the_answer_is_written"])
async def test_a_turn_the_node_stopped_waiting_for_is_served_as_one(
    tmp_path: Path, where: str
) -> None:
    """The node's timeout fires on its first question -- while the model is slow, or while the
    answer is being written to disk -- and it asks again; the charge goes out, and the process
    dies. A timed-out turn left no outcome, so a resume sent every later question to the live
    model; and an answer written after the node gave up was on disk as though the node had
    seen it, and a resume acted on it. Either way a second, different charge. The turn is
    recorded as cancelled, and served as one that never answers. Found by the thirteenth
    review."""
    db = tmp_path / "source.db"
    adapter, registry, charged = times_out()
    run_id = new_ulid()
    slow_model = where == "while_the_model_thinks"
    journal_class = CrashingJournal if slow_model else SlowToWriteAnswers
    with pytest.raises(Crash):
        await scheduler(
            journal_class(db, before_commit_of("bill#0")),
            adapter,
            registry,
            AnswersLate(25.0, slow=slow_model),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]
    cancelled = [
        e.payload
        for e in Journal(db).read(run_id, kinds=["model_response"])
        if e.payload.get("cancelled")
    ]
    assert cancelled, "the question the node gave up on has no outcome"

    changed_its_mind = AnswersLate(30.0, slow=False)
    resumed = await scheduler(Journal(db), adapter, registry, changed_its_mind).resume(run_id)
    assert resumed.ok, resumed.error
    assert changed_its_mind.asked == [], "the resume asked the model what the journal answers"
    assert charged == [25.0], "a second, different charge went out"


async def test_replay_waits_out_a_turn_the_node_stopped_waiting_for(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "source.db")
    adapter, registry, charged = times_out()
    result = await scheduler(journal, adapter, registry, AnswersLate(25.0, slow=True)).run(
        new_ulid(), {}
    )
    assert result.ok, result.error
    adapter, registry, again = times_out()
    replayed = await scheduler(
        Journal(tmp_path / "replay.db"), adapter, registry, replay_of(journal, result.run_id)
    ).run(new_ulid(), {})
    assert replayed.ok, replayed.error
    assert charged == [25.0] and again == [25.0]


async def test_a_client_error_reaches_the_node_as_a_model_error(tmp_path: Path) -> None:
    """A served failure is a ModelError, and the live one was the client's own error: a node
    that caught the client's type ran differently on resume and in replay. Both are a
    ModelError now, caused by the client's error. Found by the thirteenth review."""

    class Resets:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            raise ConnectionError("connection reset by peer")

        def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            raise NotImplementedError

    seen: list[BaseException] = []

    @node(name="ask")
    async def ask(session: RunSession) -> Decision:
        try:
            await session.model.complete(SMALL)  # type: ignore[union-attr]
        except ModelError as exc:
            seen.append(exc)
        session.state["asked"] = True
        return ToolCall("noop", {})

    adapter = PlainAdapter.of([ask], lambda s: None if s.get("asked") else "ask")
    result = await scheduler(Journal(tmp_path / "j.db"), adapter, registry_of([]), Resets()).run(
        new_ulid(), {}
    )
    assert result.ok, result.error
    assert len(seen) == 1 and isinstance(seen[0].__cause__, ConnectionError)


async def test_a_resume_that_keeps_waiting_where_the_run_gave_up_stops_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node asked the model, found the answer in its cache meanwhile, cancelled the call and
    charged; the process died. On resume the cache had expired and the node waited for the
    call -- served as one that never answers, with nothing to end it, the resume hung without
    a word. It gives up once it has waited as the recorded run did, and a margin, and says why.
    Found by the fourteenth review."""
    from specunode.core import model as model_module
    from specunode.core.model import TurnAbandoned

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.1)
    cache = {"hit": True}
    charged: list[float] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        charged.append(amount)
        return {"charge_id": f"ch_{len(charged)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        asking = asyncio.create_task(session.call_turn(SMALL))  # type: ignore[misc]
        await asyncio.sleep(0.05)
        if cache["hit"]:
            asking.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asking
            await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 25.0})
        else:
            await asking
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
            AnswersLate(25.0, slow=True),
        ).run(run_id, {})
    await bury_the_dead_process()
    assert charged == [25.0]

    cache["hit"] = False
    asked_live = AnswersLate(30.0, slow=False)
    resumed = await asyncio.wait_for(
        scheduler(Journal(db), adapter, registry, asked_live).resume(run_id), timeout=10
    )
    assert not resumed.ok and TurnAbandoned.__name__ in (resumed.error or ""), resumed.error
    assert asked_live.asked == [] and charged == [25.0]
