"""The LangGraph integration (spec task 2.3).

Task 2.3's Verify, stated exactly: the support example runs unchanged under vanilla LangGraph
and under ``wrap()``; both produce the same final state; only the wrapped one produces a ledger.

The "unchanged" clause is the substance. The graph file builds nodes and edges like any
LangGraph app, and nothing in it mentions speculation, staging or branches. Wrapping
substitutes each node's bound runnable so the body runs inside a branch -- LangGraph keeps
deciding what runs next and keeps reducing state, which is why the final states have to match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="the LangGraph integration is an optional extra")

from examples.support_agent.langgraph_agent import (
    build_graph,
    build_registry,
)

from specunode.buffer.dispatcher import Dispatcher
from specunode.core.graph import RoutingIsInternal
from specunode.core.model import JournaledModel
from specunode.ids import new_ulid
from specunode.integrations.langgraph import (
    LangGraphAdapter,
    probe,
    wrap,
)
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import standard_world

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})


def scripted() -> ScriptedModel:
    return ScriptedModel(turns=[tool_turn(CHARGE, turn=0)])


async def test_the_same_graph_runs_unwrapped(tmp_path: Path) -> None:
    """Nothing in the graph file requires a runtime; routed tools call straight through."""
    world = standard_world()
    compiled = build_graph(world, scripted())
    state = await compiled.ainvoke({"customer_id": "cus-1"})
    assert state["receipted"] is True
    assert state["log"] == ["lookup", "decide", "charge"]
    # Unwrapped, the writes happen immediately and nothing attributes them to a branch.
    assert [m.tool for m in world.mutations] == ["charge_card", "send_receipt"]
    assert {m.branch_id for m in world.mutations} == {"<external>"}


async def test_the_same_graph_runs_wrapped_and_reaches_the_same_state(tmp_path: Path) -> None:
    world = standard_world()
    unwrapped_world = standard_world()
    unwrapped_state = await build_graph(unwrapped_world, scripted()).ainvoke(
        {"customer_id": "cus-1"}
    )

    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
    )
    result = await graph.run({"customer_id": "cus-1"})

    assert result.ok, result.error
    assert result.state["receipted"] == unwrapped_state["receipted"]
    assert result.state["log"] == unwrapped_state["log"]


async def test_only_the_wrapped_run_produces_a_ledger(tmp_path: Path) -> None:
    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
    )
    result = await graph.run({"customer_id": "cus-1"})
    assert [row.call.name for row in result.ledger.rows] == ["charge_card", "send_receipt"]
    assert all(row.status == "DISPATCHED" for row in result.ledger.rows)


async def test_the_wrapped_run_attributes_every_effect_to_a_retired_branch(
    tmp_path: Path,
) -> None:
    """Hard Rule 3 through a framework that owns its own loop."""
    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    run_id = new_ulid()
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        run_id=run_id,
    )
    await graph.run({"customer_id": "cus-1"}, run_id=run_id)

    retired = {
        entry.payload["branch_id"]
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    touched = {m.branch_id for m in world.mutations} - {"<external>"}
    assert touched and touched <= retired


async def test_the_receipt_carries_the_real_charge_id_through_langgraph(tmp_path: Path) -> None:
    """The node reads its own write's result inside a framework node body."""
    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
    )
    result = await graph.run({"customer_id": "cus-1"})
    receipt = next(row for row in result.ledger.rows if row.call.name == "send_receipt")
    assert receipt.call.args["charge_id"] == "chg-1"
    assert "handle:" not in str(receipt.call.args)


async def test_a_node_runs_under_its_own_branch(tmp_path: Path) -> None:
    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    run_id = new_ulid()
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        run_id=run_id,
    )
    await graph.run({"customer_id": "cus-1"}, run_id=run_id)
    forks = list(journal.read(run_id, kinds=["branch_forked"]))
    assert len(forks) == 3, "one branch per node: lookup, decide, charge"
    assert {str(f.payload["node_id"]).split("#")[0] for f in forks} == {
        "lookup",
        "decide",
        "charge",
    }


# -- the seam this integration depends on ------------------------------------------------


def test_the_probe_confirms_the_substitution_point_exists() -> None:
    world = standard_world()
    result = probe(build_graph(world, scripted()))
    assert result.usable, "compiled.nodes[*].bound is the seam; without it nothing is wrapped"
    assert result.version


def test_routing_stays_internal_to_the_framework() -> None:
    """Re-deriving Pregel's routing outside it would break the same-final-state promise."""
    world = standard_world()
    adapter = LangGraphAdapter(compiled=build_graph(world, scripted()))
    assert adapter.capabilities().drives_itself
    with pytest.raises(RoutingIsInternal):
        adapter.next({})


def test_the_adapter_reports_the_langgraph_version_it_ran_against() -> None:
    world = standard_world()
    adapter = LangGraphAdapter(compiled=build_graph(world, scripted()))
    assert adapter.capabilities().framework == "langgraph"
    assert adapter.capabilities().version


async def test_the_buffer_and_the_ledger_read_the_same_run(tmp_path: Path) -> None:
    """One source of truth for the run id.

    When wrap() minted a run id for the store buffer and run() minted another for the
    scheduler, effects were journaled under one run while the ledger was built from the other.
    The run reported ok, the ledger was empty, and the world had nonetheless been changed --
    a silent, confident, wrong artifact, which is the exact shape this project exists to avoid.
    """
    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
    )
    result = await graph.run({"customer_id": "cus-1"})

    assert world.mutations, "the world was changed"
    assert result.ledger.rows, "so the ledger must not be empty"
    assert len(result.ledger.rows) == len(world.mutations)
    assert graph.scheduler.buffer.run_id == result.run_id
