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
