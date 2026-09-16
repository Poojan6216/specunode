"""Sub-graphs, streaming and interrupts (spec tasks 8.1, 8.2 and 8.3).

Three ways a real pipeline differs from the example in the README, and each one has a way of
going quietly wrong.

A sub-graph wrapped as one opaque branch would make the leak invariant meaningless inside it:
every inner effect would be attributed to the container, and a wrong one could not be told from
a right one. So the inner nodes are substituted and the container is not.

Streaming a speculative branch's output would show a user text the run then decided against,
and there is no taking it back once it is on their screen.

An interrupt inside a speculative branch would ask a human to approve work that may be
discarded. A speculative branch does not run node bodies at all unless the developer opted the
node in, so the question does not arise by default -- and that default is what this asserts.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph", reason="the LangGraph integration is an optional extra")

from examples.support_agent.langgraph_agent import build_registry

from specunode.buffer.dispatcher import Dispatcher
from specunode.core.effects import EffectClass, ToolSpec
from specunode.core.graph import routed
from specunode.core.model import JournaledModel
from specunode.ids import new_ulid
from specunode.integrations.langgraph import LangGraphAdapter, wrap
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world


class NestedState(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    done: bool


def build_nested(world: World) -> Any:
    """An outer graph with a sub-graph inside it, each node doing one real write."""
    from langgraph.graph import END, StateGraph

    registry = build_registry(world)
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    registry.register(
        ToolSpec(name="post_summary", effect=EffectClass.WRITE, fn=world.post_summary)
    )
    restart = routed(registry.get("restart_job"))
    summarise = routed(registry.get("post_summary"))

    async def inner_restart(state: NestedState) -> NestedState:
        await restart(job_id="etl-1")
        return {"log": ["inner_restart"]}

    async def inner_summarise(state: NestedState) -> NestedState:
        await summarise(channel="#ops", text="restarted")
        return {"log": ["inner_summarise"], "done": True}

    inner = StateGraph(NestedState)
    inner.add_node("inner_restart", inner_restart)
    inner.add_node("inner_summarise", inner_summarise)
    inner.set_entry_point("inner_restart")
    inner.add_edge("inner_restart", "inner_summarise")
    inner.add_edge("inner_summarise", END)

    async def outer_start(state: NestedState) -> NestedState:
        return {"log": ["outer_start"]}

    outer = StateGraph(NestedState)
    outer.add_node("outer_start", outer_start)
    outer.add_node("sub", inner.compile())
    outer.set_entry_point("outer_start")
    outer.add_edge("outer_start", "sub")
    outer.add_edge("sub", END)
    return outer.compile(), registry


def wrapped(tmp_path: Path, db: str = "nested.db") -> tuple[Any, World, Journal, str]:
    world = standard_world()
    compiled, registry = build_nested(world)
    journal = Journal(tmp_path / db)
    run_id = new_ulid()
    graph = wrap(
        compiled,
        registry=registry,
        journal=journal,
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(("restart_job", {"job_id": "etl-1"}), turn=0)]),
            journal,
            provider="scripted",
        ),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        run_id=run_id,
    )
    return graph, world, journal, run_id


# -- sub-graphs (8.3) ---------------------------------------------------------------------


def test_a_subgraphs_inner_nodes_are_substituted_not_its_container(tmp_path: Path) -> None:
    world = standard_world()
    compiled, _registry = build_nested(world)
    adapter = LangGraphAdapter(compiled=compiled)

    assert adapter.subgraphs == 1
    assert "sub/inner_restart" in adapter.installed_nodes
    assert "sub/inner_summarise" in adapter.installed_nodes
    assert "sub" not in adapter.installed_nodes, "the container does no work and gets no branch"
    assert "outer_start" in adapter.installed_nodes


def test_a_nested_node_carries_its_parents_path(tmp_path: Path) -> None:
    """Otherwise two sub-graphs with a node of the same name derive the same idempotency key."""
    world = standard_world()
    compiled, _registry = build_nested(world)
    refs = {ref.structural_id for ref in LangGraphAdapter(compiled=compiled).nodes()}
    assert "sub/inner_restart" in refs


async def test_each_inner_node_retires_its_own_branch(tmp_path: Path) -> None:
    graph, world, journal, run_id = wrapped(tmp_path)
    result = await graph.run({})
    assert result.ok, result.error

    forks = list(journal.read(run_id, kinds=["branch_forked"]))
    node_ids = {str(entry.payload["node_id"]).split("#")[0] for entry in forks}
    assert node_ids == {"outer_start", "sub/inner_restart", "sub/inner_summarise"}
    assert [m.tool for m in world.mutations] == ["restart_job", "post_summary"]


async def test_the_leak_invariant_holds_inside_a_subgraph(tmp_path: Path) -> None:
    """The reason the container is not wrapped: attribution has to reach the inner effects."""
    graph, world, journal, run_id = wrapped(tmp_path, db="leak.db")
    await graph.run({})

    retired = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    touched = {m.branch_id for m in world.mutations} - {"<external>"}
    assert touched and touched <= retired


# -- streaming (8.2) ----------------------------------------------------------------------


async def test_streaming_yields_only_what_reached_the_world(tmp_path: Path) -> None:
    """A speculative branch's output may be squashed; a user's screen cannot be un-written."""
    graph, world, _journal, _run_id = wrapped(tmp_path, db="stream.db")
    chunks = [chunk async for chunk in graph.astream({})]

    effects = [c for c in chunks if "effect" in c]
    assert [c["effect"] for c in effects] == [m.tool for m in world.mutations]
    assert all(c["status"] == "DISPATCHED" for c in effects)
    assert "state" in chunks[-1]


# -- interrupts (8.1) ---------------------------------------------------------------------


def test_no_node_is_speculable_unless_it_opted_in(tmp_path: Path) -> None:
    """Which is what keeps an interrupt out of a speculative branch by default.

    A node body is unbounded code -- it can prompt a human, write a file, open a socket -- so
    speculation does not run one until the developer says it is safe to run and discard.
    """
    world = standard_world()
    compiled, _registry = build_nested(world)
    capabilities = LangGraphAdapter(compiled=compiled).capabilities()
    assert capabilities.speculable_nodes == frozenset()
    assert capabilities.supports_interrupt


async def test_an_opt_in_is_explicit_and_per_node(tmp_path: Path) -> None:
    world = standard_world()
    compiled, _registry = build_nested(world)
    adapter = LangGraphAdapter(compiled=compiled, speculable=frozenset({"outer_start"}))
    assert adapter.capabilities().speculable_nodes == frozenset({"outer_start"})
    assert "sub/inner_restart" not in adapter.capabilities().speculable_nodes
