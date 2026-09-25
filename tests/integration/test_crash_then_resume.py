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
from collections.abc import Callable, Mapping
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
from specunode.core.effects import ToolRegistry
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.journal.replay import ReplayModel, recover
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import standard_world


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

    Both answers sit in the journal under one node and one position. Replay used to serve the
    dead attempt's, reproducing a charge the committed run never made.
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
