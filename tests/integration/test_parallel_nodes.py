"""Parallel nodes: independent work side by side, and not one guarantee traded for it.

A graph's router may name several nodes at once (:class:`Parallel`). Their bodies overlap in
time -- which is the entire point when each one waits seconds on a model -- and everything a
reader relies on stays as it was: the same program positions and idempotency keys whether the
bodies overlap or not, effects reaching the world in the order the router named the nodes
however they happened to finish, a write conflict refused before anything is sent, and a
failing node leaving no effect behind and no fork without a resolution.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from examples.fanout_agent.agent import KeyedScriptedModel
from examples.fanout_agent.agent import build as build_fanout

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText
from specunode.core.effects import EffectClass
from specunode.core.graph import NodeRef, Parallel, RunSession
from specunode.core.model import JournaledModel, RequestEnvelope, StreamEvent, TextBlock
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger, render_ledger
from specunode.testing.models import ScriptedModel
from specunode.testing.world import World, standard_world


async def run_graph(
    tmp_path: Path,
    adapter: object,
    registry: object,
    *,
    parallel: bool = True,
    model: object | None = None,
    db: str,
    reducers: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> tuple[RunResult, Journal, str]:
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=adapter,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),  # type: ignore[arg-type]
        target=JournaledModel(model or ScriptedModel(turns=[]), journal, provider="scripted"),  # type: ignore[arg-type]
        policy=Policy(speculation=False, parallel_nodes=parallel),
        reducers=dict(reducers or {}),
    )
    run_id = run_id or new_ulid()
    return await scheduler.run(run_id, {}), journal, run_id


def every_fork_resolved(journal: Journal, run_id: str) -> None:
    forked = {e.payload["branch_id"] for e in journal.read(run_id, kinds=["branch_forked"])}
    resolved = {e.payload["branch_id"] for e in journal.read(run_id, kinds=["branch_resolved"])}
    assert forked <= resolved, f"{len(forked - resolved)} fork(s) never resolved"
    assert [e.kind for e in journal.read(run_id)][-1] == "run_finished"
    assert journal.verify_chain(run_id).ok


# -- a graph whose lanes write, finishing in the reverse of the order they were named ----------


def writers(
    world: World, *, same_key: bool = False, fail: str | None = None, state_first: str = ""
) -> tuple[object, object]:
    """Three lanes that each post to their own channel; ``a`` is slowest, ``c`` fastest.

    A lane named in ``state_first`` writes its state before it posts; the others write it
    after the post returns, which no check can see until that post has gone out.
    """

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)

    def lane(name: str, delay: float) -> object:
        @node(name=name, emits="tool_call")
        async def body(session: RunSession) -> Decision:
            await asyncio.sleep(delay)
            if name == fail:
                raise RuntimeError(f"{name} fell over")
            key = "shared" if same_key else f"done:{name}"
            if name in state_first:
                session.state[key] = name
            await session.call_tool("post_summary", {"channel": f"#{name}", "text": name})
            session.state[key] = name
            return FreeText.of(name)

        return body

    @node(name="finish", emits="tool_call")
    async def finish(session: RunSession) -> Decision:
        session.state["finished"] = True
        return FreeText.of("finished")

    def route(state: Mapping[str, JsonValue]) -> str | list[str] | None:
        if not any(k.startswith("done:") or k == "shared" for k in state):
            return ["a", "b", "c"]
        return None if state.get("finished") else "finish"

    adapter = PlainAdapter.of([lane("a", 0.12), lane("b", 0.06), lane("c", 0.0), finish], route)
    return adapter, registry_of([post_summary])  # type: ignore[arg-type]


def posted(world: World) -> list[str]:
    """The channels the world received posts on, in the order it received them."""
    return [str(row["channel"]) for row in world.tables["messages"].values()]


def discarded(journal: Journal, run_id: str) -> int:
    return sum(
        int(e.payload.get("count", 0)) for e in journal.read(run_id, kinds=["effect_discarded"])
    )


# -- the fan-out example: model-bound lanes ------------------------------------------------------


async def test_independent_nodes_overlap_their_model_calls(tmp_path: Path) -> None:
    """The point of the feature, counted rather than timed."""
    world = standard_world()
    adapter, registry = build_fanout(world)
    model = KeyedScriptedModel(think_ms=50.0)
    result, journal, run_id = await run_graph(
        tmp_path, adapter, registry, model=model, db="overlap.db"
    )
    assert result.ok, result.error
    assert model.high_water == 3, f"at most {model.high_water} model call(s) were ever in flight"
    every_fork_resolved(journal, run_id)

    world = standard_world()
    adapter, registry = build_fanout(world)
    serial = KeyedScriptedModel(think_ms=50.0)
    result, _, _ = await run_graph(
        tmp_path, adapter, registry, model=serial, parallel=False, db="serial.db"
    )
    assert result.ok, result.error
    assert serial.high_water == 1, "with parallel_nodes off the bodies overlapped anyway"


async def test_side_by_side_and_one_at_a_time_produce_the_same_run(tmp_path: Path) -> None:
    """Only the wall clock may differ: same keys, same positions, same ledger, same world."""
    run_id = new_ulid()
    outcomes = []
    for parallel in (True, False):
        world = standard_world()
        adapter, registry = build_fanout(world)
        result, journal, _ = await run_graph(
            tmp_path,
            adapter,
            registry,
            model=KeyedScriptedModel(),
            parallel=parallel,
            db=f"same-{parallel}.db",
            run_id=run_id,
        )
        assert result.ok, result.error
        keys = [e.payload["nkey"] for e in journal.read(run_id, kinds=["effect_dispatched"])]
        forks = [
            (e.payload.get("node_id"), e.payload.get("step"))
            for e in journal.read(run_id, kinds=["branch_forked"])
        ]
        ledger = render_ledger(build_ledger(journal, run_id), normalised=True)
        outcomes.append((keys, forks, ledger, world.tables["messages"], result.state))
    assert outcomes[0][0], "nothing was dispatched, so the keys prove nothing"
    assert outcomes[0] == outcomes[1]


class AnsweringInReverse(KeyedScriptedModel):
    """Asked about etl-1, etl-2, etl-3 in that order; answers etl-3 first and etl-1 last."""

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        asked = "".join(
            block.text
            for message in envelope.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        await asyncio.sleep(0.04 if "etl-1" in asked else 0.02 if "etl-2" in asked else 0.0)
        async for event in super().stream(envelope):
            yield event


async def test_a_parallel_run_replays_against_its_own_journal(tmp_path: Path) -> None:
    """Three nodes ask the model from one program position, and their replies were recorded in
    the reverse of the order replay asks for them. Replay must hand each node its own."""
    from specunode.journal.replay import ReplayModel, recover

    world = standard_world()
    adapter, registry = build_fanout(world)
    result, journal, run_id = await run_graph(
        tmp_path, adapter, registry, model=AnsweringInReverse(), db="recorded.db"
    )
    assert result.ok, result.error
    replay = ReplayModel(
        journal=journal, run_id=run_id, retired_branches=recover(journal, run_id).retired_branches
    )
    replayed_world = standard_world()
    adapter, registry = build_fanout(replayed_world)
    replayed, _, _ = await run_graph(tmp_path, adapter, registry, model=replay, db="replayed.db")
    assert replayed.ok, replayed.error
    assert replayed.state == result.state
    assert replayed_world.tables["messages"] == world.tables["messages"]


# -- ordering, conflicts and failure -------------------------------------------------------------


async def test_effects_reach_the_world_in_the_order_the_router_named_the_nodes(
    tmp_path: Path,
) -> None:
    """``c`` is ready first and ``a`` last; the world still sees a, b, c."""
    world = standard_world()
    adapter, registry = writers(world)
    result, journal, run_id = await run_graph(tmp_path, adapter, registry, db="order.db")
    assert result.ok, result.error
    assert posted(world) == ["#a", "#b", "#c"]
    every_fork_resolved(journal, run_id)


@pytest.mark.parametrize(
    ("state_first", "sent"),
    [
        # Every lane wrote the key before its post: refused before anything leaves.
        ("abc", []),
        # ``a`` commits the key only after its post; ``b`` had written it before its own, so
        # ``b`` is refused before its post leaves.
        ("b", ["#a"]),
        # Nobody writes the key until their post has returned. ``b``'s clash cannot be seen
        # until its post is out; ``c``'s post is still held back.
        ("", ["#a", "#b"]),
    ],
)
async def test_two_nodes_writing_one_key_are_refused_as_early_as_it_can_be_seen(
    tmp_path: Path, state_first: str, sent: list[str]
) -> None:
    """Applied on top of each other, the later write would silently overwrite the earlier."""
    world = standard_world()
    adapter, registry = writers(world, same_key=True, state_first=state_first)
    result, journal, run_id = await run_graph(tmp_path, adapter, registry, db="conflict.db")
    assert not result.ok
    assert "both write state key 'shared'" in str(result.error)
    assert posted(world) == sent
    assert discarded(journal, run_id) == 3 - len(sent), "a held post was sent or left staged"
    every_fork_resolved(journal, run_id)


async def test_a_declared_reducer_is_how_two_nodes_may_share_a_key(tmp_path: Path) -> None:
    world = standard_world()
    adapter, registry = writers(world, same_key=True)
    result, journal, run_id = await run_graph(
        tmp_path, adapter, registry, reducers={"shared": "last_write"}, db="reduced.db"
    )
    assert result.ok, result.error
    assert result.state["shared"] == "c", "the reducer ran in declared order"
    every_fork_resolved(journal, run_id)


async def test_a_failing_node_leaves_no_effect_from_its_siblings(tmp_path: Path) -> None:
    """None of the group has retired, so none of its writes may leave -- including the siblings'."""
    world = standard_world()
    adapter, registry = writers(world, fail="b")
    result, journal, run_id = await run_graph(tmp_path, adapter, registry, db="failed.db")
    assert not result.ok
    assert "node b failed" in str(result.error)
    assert posted(world) == []
    assert discarded(journal, run_id) == 2, "the two siblings' staged writes were not discarded"
    every_fork_resolved(journal, run_id)


async def test_a_node_that_read_what_an_earlier_one_changed_is_refused(tmp_path: Path) -> None:
    """Nodes named together must be independent. ``look`` read etl-2 while ``fix`` was about to
    restart it; by ``look``'s retirement the world no longer holds what it read. It is refused,
    not retired on a stale value -- and the message says nothing of it was sent, because
    nothing was."""
    world = standard_world()

    @tool(effect=EffectClass.READ, witness=True, forward_keys="job:{args.pipeline_id}")
    async def get_pipeline_status(pipeline_id: str) -> JsonValue:
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    @tool(effect=EffectClass.WRITE, idempotent=True, forward_keys="job:{args.job_id}")
    async def restart_job(job_id: str) -> JsonValue:
        return await world.restart_job(job_id=job_id)

    @node(name="fix", emits="tool_call")
    async def fix(session: RunSession) -> Decision:
        await session.call_tool("restart_job", {"job_id": "etl-2"})
        session.state["fixed"] = True
        return FreeText.of("fixed")

    @node(name="look", emits="tool_call")
    async def look(session: RunSession) -> Decision:
        seen = await session.call_tool("get_pipeline_status", {"pipeline_id": "etl-2"})
        session.state["seen"] = seen
        return FreeText.of("looked")

    def route(state: Mapping[str, JsonValue]) -> list[str] | None:
        return None if "fixed" in state else ["fix", "look"]

    adapter = PlainAdapter.of([fix, look], route)  # type: ignore[list-item]
    registry = registry_of([get_pipeline_status, restart_job])  # type: ignore[list-item]
    result, journal, run_id = await run_graph(tmp_path, adapter, registry, db="stale.db")
    assert not result.ok
    assert "node look was refused at retirement and nothing it staged was sent" in str(
        result.error
    ), result.error
    assert "stale" in str(result.error)
    assert [m.tool for m in world.mutations] == ["restart_job"], "only fix's write went out"
    every_fork_resolved(journal, run_id)


def test_the_router_may_name_one_node_or_several() -> None:
    world = standard_world()
    adapter, _ = writers(world)
    assert isinstance(adapter.next({}), Parallel)  # type: ignore[attr-defined]
    assert adapter.next({"done:a": 1}) == NodeRef(name="finish")  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        Parallel(nodes=(NodeRef(name="a"),))


# -- found by an independent adversarial review, each with a failing test first -----------------


def pending_tasks() -> list[str]:
    return sorted(
        task.get_coro().__qualname__  # type: ignore[union-attr]
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    )


@pytest.mark.parametrize(
    ("key", "reducer", "initial", "values"),
    [
        # A node writes what it contributes; ``append`` adds it to what is committed.
        ("log", "append", [], {"a": ["a"], "b": ["b"]}),
        ("log", "append", ["x"], {"a": ["a"], "b": ["b"]}),
        # The later lane in declared order wins, whole -- not a value neither lane wrote.
        ("cfg", "last_write", {"p": 1, "q": 1}, {"a": {"p": 2, "q": 1}, "b": {"p": 1, "q": 3}}),
    ],
    ids=["append-empty", "append-seeded", "last_write-object"],
)
async def test_a_reducer_combines_lanes_as_it_combines_nodes_run_one_after_another(
    tmp_path: Path, key: str, reducer: str, initial: JsonValue, values: Mapping[str, JsonValue]
) -> None:
    """A lane's delta was replayed on top of a sibling's commit, positions and all."""
    final = {}
    for grouped in (False, True):
        world = standard_world()

        @tool(effect=EffectClass.WRITE, idempotent=False)
        async def post_summary(channel: str, text: str) -> JsonValue:
            return await world.post_summary(channel=channel, text=text)  # noqa: B023

        def lane(name: str) -> object:
            @node(name=name)
            async def body(session: RunSession) -> Decision:
                await session.call_tool("post_summary", {"channel": f"#{name}", "text": name})
                session.state[key] = values[name]
                session.state[f"done:{name}"] = True
                return FreeText.of(name)

            return body

        def route(state: Mapping[str, JsonValue], grouped: bool = grouped) -> object:
            pending = [n for n in ("a", "b") if f"done:{n}" not in state]
            return (pending if grouped else pending[0]) if pending else None

        adapter = PlainAdapter.of([lane("a"), lane("b")], route)  # type: ignore[list-item,arg-type]
        journal = Journal(tmp_path / f"{key}-{grouped}.db")
        scheduler = Scheduler(
            graph=adapter,
            registry=registry_of([post_summary]),  # type: ignore[list-item]
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry_of([post_summary]), base_delay_ms=0.5),  # type: ignore[list-item]
            target=JournaledModel(ScriptedModel(turns=[]), journal, provider="scripted"),
            policy=Policy(speculation=False),
            reducers={key: reducer},
        )
        result = await scheduler.run(new_ulid(), {key: initial})
        assert result.ok, result.error
        final[grouped] = result.state[key]
    assert final[True] == final[False]


async def test_a_clash_found_at_commit_closes_the_lane_and_a_resume_sends_nothing_twice(
    tmp_path: Path,
) -> None:
    """``b`` writes the key only after its post returned, so the clash is found after the post.

    The lane used to stay confirmed forever, with an error that did not say its post was out,
    and a resume dropped it and the lane after it and reported success.
    """
    world = standard_world()
    adapter, registry = writers(world, same_key=True)
    result, journal, run_id = await run_graph(tmp_path, adapter, registry, db="clash.db")
    assert not result.ok
    assert "both write state key 'shared'" in str(result.error)
    assert "1 of its effects had already been dispatched" in str(result.error)
    every_fork_resolved(journal, run_id)
    lane_b = next(
        e.payload["branch_id"]
        for e in journal.read(run_id, kinds=["branch_forked"])
        if e.payload["node_id"] == "b#0"
    )
    statuses = [
        e.payload["status"]
        for e in journal.read(run_id, kinds=["branch_resolved"])
        if e.payload["branch_id"] == lane_b
    ]
    assert statuses[-1] == "faulted", f"lane b's lifecycle ended at {statuses}"
    assert posted(world) == ["#a", "#b"]

    def resume(reducers: Mapping[str, str]) -> Scheduler:
        again = Journal(tmp_path / "clash.db")
        return Scheduler(
            graph=adapter,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            journal=again,
            buffer=StoreBuffer(journal=again, run_id=""),
            dispatcher=Dispatcher(registry=registry, base_delay_ms=0.5),  # type: ignore[arg-type]
            target=JournaledModel(ScriptedModel(turns=[]), again, provider="scripted"),
            policy=Policy(speculation=False),
            reducers=dict(reducers),
        )

    # Resumed as it is, it re-runs b and c under their own keys and refuses the same clash.
    refused = await resume({}).resume(run_id)
    assert not refused.ok and "both write state key 'shared'" in str(refused.error)
    assert posted(world) == ["#a", "#b"], "the resume sent b's post again"
    # With the clash resolved -- a reducer declared -- the resume finishes the group.
    finished = await resume({"shared": "last_write"}).resume(run_id)
    assert finished.ok, finished.error
    assert finished.state["shared"] == "c"
    assert posted(world) == ["#a", "#b", "#c"]


@pytest.mark.parametrize("failure", ["dead-letter", "reducer-refuses"])
async def test_a_lane_that_fails_at_retirement_strands_nothing(
    tmp_path: Path, failure: str
) -> None:
    """Only a state clash used to be caught; anything else escaped with later lanes parked."""
    from specunode.buffer.dispatcher import ToolDispatchError

    world = standard_world()

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def page_oncall(team: str) -> JsonValue:
        raise ToolDispatchError("pager is down", sent="no", retriable=False)

    @node(name="a")
    async def a(session: RunSession) -> Decision:
        if failure == "dead-letter":
            await session.call_tool("page_oncall", {"team": "data"})
        else:
            await session.call_tool("post_summary", {"channel": "#a", "text": "a"})
            session.state["level"] = 5
        return FreeText.of("a")

    @node(name="b")
    async def b(session: RunSession) -> Decision:
        await session.call_tool("post_summary", {"channel": "#b", "text": "b"})
        session.state["level"] = "high"  # ``max`` cannot order it against 5
        return FreeText.of("b")

    @node(name="c")
    async def c(session: RunSession) -> Decision:
        await session.call_tool("post_summary", {"channel": "#c", "text": "c"})
        session.state["c"] = True
        return FreeText.of("c")

    def route(state: Mapping[str, JsonValue]) -> list[str] | None:
        return None if "c" in state else ["a", "b", "c"]

    adapter = PlainAdapter.of([a, b, c], route)  # type: ignore[list-item]
    registry = registry_of([post_summary, page_oncall])  # type: ignore[list-item]
    result, journal, run_id = await run_graph(
        tmp_path, adapter, registry, reducers={"level": "max"}, db=f"{failure}.db"
    )
    assert not result.ok
    assert ("dead-lettered" if failure == "dead-letter" else "ReducerError") in str(result.error)
    every_fork_resolved(journal, run_id)
    assert pending_tasks() == []
    assert "#c" not in posted(world), "a lane after the failure was sent"


async def test_a_lane_whose_finally_writes_does_not_hang_an_abandoned_group(
    tmp_path: Path,
) -> None:
    """Hold something, always give it back: the ``finally`` wrote into a buffer nobody drained."""
    world = standard_world()

    @tool(effect=EffectClass.WRITE, idempotent=True, forward_keys="job:{args.job_id}")
    async def reserve_capacity(job_id: str, units: int) -> JsonValue:
        return await world.reserve_capacity(job_id=job_id, units=units)

    @tool(effect=EffectClass.WRITE, idempotent=True, forward_keys="job:{args.job_id}")
    async def release_capacity(job_id: str, units: int) -> JsonValue:
        return await world.release_capacity(job_id=job_id, units=units)

    @node(name="lease")
    async def lease(session: RunSession) -> Decision:
        try:
            await session.call_tool("reserve_capacity", {"job_id": "etl-2", "units": 1})
            session.state["lease"] = True
            return FreeText.of("lease")
        finally:
            await session.call_tool("release_capacity", {"job_id": "etl-2", "units": 1})

    @node(name="y")
    async def y(session: RunSession) -> Decision:
        await asyncio.sleep(0.02)
        raise RuntimeError("y fell over")

    def route(state: Mapping[str, JsonValue]) -> list[str] | None:
        return None if "lease" in state else ["lease", "y"]

    adapter = PlainAdapter.of([lease, y], route)  # type: ignore[list-item]
    registry = registry_of([reserve_capacity, release_capacity])  # type: ignore[list-item]
    result, journal, run_id = await asyncio.wait_for(
        run_graph(tmp_path, adapter, registry, db="lease.db"), timeout=10
    )
    assert not result.ok and "node y failed" in str(result.error)
    assert world.mutations == [], "a revoked lane's write was sent"
    every_fork_resolved(journal, run_id)


@pytest.mark.parametrize("torn_down_by", ["a sibling failing", "its own write dead-lettering"])
async def test_a_turn_torn_down_while_settling_leaves_nothing_running(
    tmp_path: Path, torn_down_by: str
) -> None:
    """The read a turn issued early for a later block kept running after the run returned."""
    from specunode.buffer.dispatcher import ToolDispatchError
    from specunode.core.model import Message
    from specunode.testing.models import tool_turn

    world = standard_world()
    write_fails = torn_down_by == "its own write dead-lettering"

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        if write_fails:
            raise ToolDispatchError("upstream refused", sent="no", retriable=False)
        return await world.post_summary(channel=channel, text=text)

    @tool(effect=EffectClass.READ, witness=True)
    async def slow_status(pipeline_id: str) -> JsonValue:
        await asyncio.sleep(0.3)
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    @node(name="x")
    async def x(session: RunSession) -> Decision:
        assert session.call_turn is not None
        await session.call_turn(
            RequestEnvelope(
                model="scripted",
                messages=(Message(role="user", content=(TextBlock(text="go"),)),),
                stream=True,
            )
        )
        session.state["x"] = True
        return FreeText.of("x")

    @node(name="y")
    async def y(session: RunSession) -> Decision:
        await asyncio.sleep(0.05)
        raise RuntimeError("y fell over")

    def route(state: Mapping[str, JsonValue]) -> str | list[str] | None:
        return None if "x" in state else ("x" if write_fails else ["x", "y"])

    adapter = PlainAdapter.of([x, y], route)  # type: ignore[list-item]
    registry = registry_of([post_summary, slow_status])  # type: ignore[list-item]
    model = ScriptedModel(
        turns=[
            tool_turn(
                ("post_summary", {"channel": "#ops", "text": "restarting"}),
                ("slow_status", {"pipeline_id": "etl-2"}),
            )
        ]
    )
    result, journal, run_id = await run_graph(
        tmp_path, adapter, registry, model=model, db="settle.db"
    )
    assert not result.ok
    assert pending_tasks() == [], "something the turn started outlived the run"
    await asyncio.sleep(0.4)
    assert world.reads == [], "a read reached upstream after the run returned"
    assert [e.kind for e in journal.read(run_id)][-1] == "run_finished"


@pytest.mark.parametrize(
    ("chosen", "complaint"),
    [({"a", "b"}, "list or tuple"), ([], "return None"), (["a", "a"], "twice")],
    ids=["a set, whose order changes per process", "nothing", "a node twice"],
)
def test_a_router_names_its_group_in_an_order_that_survives_a_resume(
    chosen: object, complaint: str
) -> None:
    world = standard_world()
    adapter, _ = writers(world)
    router = PlainAdapter.of(list(adapter.node_fns.values()), lambda state: chosen)  # type: ignore[attr-defined,arg-type]
    with pytest.raises((TypeError, ValueError), match=complaint):
        router.next({})


@pytest.mark.parametrize("grouped", [False, True], ids=["one node", "a group"])
async def test_a_run_cancelled_from_outside_leaves_none_of_its_nodes_running(
    tmp_path: Path, grouped: bool
) -> None:
    """A timeout or a shutdown cancels ``run()``. Its nodes used to keep running without it,
    making calls for a run that was over."""
    world = standard_world()

    @tool(effect=EffectClass.READ, witness=True)
    async def slow_status(pipeline_id: str) -> JsonValue:
        await asyncio.sleep(0.5)
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    def lane(name: str) -> object:
        @node(name=name)
        async def body(session: RunSession) -> Decision:
            await asyncio.sleep(0.05)
            await session.call_tool("slow_status", {"pipeline_id": "etl-1"})
            session.state[name] = True
            return FreeText.of(name)

        return body

    def route(state: Mapping[str, JsonValue]) -> str | list[str] | None:
        return None if "a" in state else (["a", "b"] if grouped else "a")

    adapter = PlainAdapter.of([lane("a"), lane("b")], route)  # type: ignore[list-item]
    registry = registry_of([slow_status])  # type: ignore[list-item]
    running = asyncio.ensure_future(run_graph(tmp_path, adapter, registry, db="cancel.db"))
    await asyncio.sleep(0.02)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await asyncio.sleep(0)
    assert pending_tasks() == [], "a node outlived the run that was cancelled"
    await asyncio.sleep(0.6)
    assert world.reads == [], "a cancelled run's node still reached upstream"
