"""The LangGraph integration (spec task 2.3).

Task 2.3's Verify, stated exactly: the support example runs unchanged under vanilla LangGraph
and under ``wrap()``; both produce the same final state; only the wrapped one produces a ledger.

The "unchanged" clause is the substance. The graph file builds nodes and edges like any
LangGraph app, and nothing in it mentions speculation, staging or branches. Wrapping
substitutes each node's bound runnable so the body runs inside a branch -- LangGraph keeps
deciding what runs next and keeps reducing state, which is why the final states have to match.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, TypedDict

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
    sent = list(journal.read(result.run_id, kinds=["effect_dispatched"]))
    assert len(sent) == len(world.mutations), "an effect was journaled under another run"


async def test_two_runs_of_one_wrapped_graph_at_once_stay_apart(tmp_path: Path) -> None:
    """One wrapped graph, two requests at once -- how a web handler calls it. ``wrap()`` built
    one Scheduler for the graph, and a Scheduler keeps its run on itself: the second run took
    over the first, both runs' charges were journaled under the second, and the first never
    finished. Found by the eleventh review."""
    import asyncio
    import re

    from specunode.core.model import ModelResponse, RequestEnvelope, TextBlock

    class ChargesTheCustomerAskedAbout:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            block = envelope.messages[-1].content[0]
            assert isinstance(block, TextBlock)
            found = re.search(r"cus-\d+", block.text)
            assert found is not None, block.text
            await asyncio.sleep(0.01)  # both runs are in a model turn at once
            return tool_turn(("charge_card", {"customer_id": found.group(0), "amount": 25.0}))

        async def stream(self, envelope: RequestEnvelope) -> object:
            raise NotImplementedError

    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    model = ChargesTheCustomerAskedAbout()
    graph = wrap(
        build_graph(world, model),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        target=JournaledModel(model, journal, provider="scripted"),  # type: ignore[arg-type]
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
    )
    runs = {"cus-1": new_ulid(), "cus-2": new_ulid()}
    results = await asyncio.gather(
        *(graph.run({"customer_id": customer}, run_id=run) for customer, run in runs.items())
    )
    for (customer, run), result in zip(runs.items(), results, strict=True):
        assert result.ok, result.error
        assert result.run_id == run
        assert [(row.call.name, row.call.args["customer_id"]) for row in result.ledger.rows] == [
            ("charge_card", customer),
            ("send_receipt", customer),
        ]
        assert len(list(journal.read(run, kinds=["run_finished"]))) == 1
    assert sorted(m.tool for m in world.mutations) == ["charge_card"] * 2 + ["send_receipt"] * 2


async def test_a_graph_with_a_checkpointer_runs_with_its_config(tmp_path: Path) -> None:
    """``ainvoke`` and ``run`` took no LangGraph config, so a graph compiled with a checkpointer
    failed at once: "Checkpointer requires ... thread_id". Found by the twelfth review."""
    from langgraph.checkpoint.memory import InMemorySaver

    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted(), checkpointer=InMemorySaver()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
    )
    config = {"configurable": {"thread_id": "support-1"}}
    result = await graph.run({"customer_id": "cus-1"}, config=config)
    assert result.ok, result.error
    assert [m.tool for m in world.mutations] == ["charge_card", "send_receipt"]


async def test_a_langgraph_run_says_it_cannot_be_resumed(tmp_path: Path) -> None:
    """The docs described continuing a crashed LangGraph run from its checkpointer; ``run``
    refused the id ("resume it") and ``resume`` failed on LangGraph's internal routing. Neither
    works in this version, and both now say so. Found by the twelfth review."""
    from specunode.core.scheduler import SchedulerError

    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(scripted(), journal, provider="scripted"),
    )
    run_id = new_ulid()
    await graph.run({"customer_id": "cus-1"}, run_id=run_id)
    with pytest.raises(SchedulerError, match="cannot be resumed in this version"):
        await graph.run({"customer_id": "cus-1"}, run_id=run_id)
    with pytest.raises(SchedulerError, match="cannot be resumed in this version"):
        await graph.new_scheduler().resume(run_id)


async def test_one_wrapped_graph_runs_any_number_of_times(tmp_path: Path) -> None:
    """``wrap(run_id=...)`` gave every call one run id, and a run id starts one run: the second
    call was refused. Each call is its own run. Found by the twelfth review."""
    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, scripted()),
        registry=registry,
        journal=journal,
        target=JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE)] * 2), journal),
    )
    first = await graph.run({"customer_id": "cus-1"})
    second = await graph.run({"customer_id": "cus-1"})
    assert first.ok and second.ok and first.run_id != second.run_id


async def test_a_failed_langgraph_run_is_not_reported_resumable(tmp_path: Path) -> None:
    """``status`` said a failed LangGraph run was resumable, and ``resume`` refuses every one.
    Found by the sixteenth review."""
    from specunode.journal.replay import recover

    world = standard_world()
    journal = Journal(tmp_path / "journal.db")
    registry = build_registry(world)
    graph = wrap(
        build_graph(world, ScriptedModel(turns=[])),
        registry=registry,
        journal=journal,
        target=JournaledModel(ScriptedModel(turns=[]), journal, provider="scripted"),
    )
    result = await graph.run({"customer_id": "cus-1"})
    assert not result.ok
    recovery = recover(journal, result.run_id)
    assert recovery.finished and recovery.failed and not recovery.resumable


async def test_nodes_run_side_by_side_keep_their_visits(tmp_path: Path) -> None:
    """Two nodes of one superstep each counted their visit on the run's cursor as they began,
    and the one that retired last put back its own cursor -- from before its sibling's visit. The
    sibling's next visit minted the same node id at the same position: its write had the same key
    as its first, and was deduped away. Found by the twenty-sixth review."""
    import asyncio
    from typing import Any

    from langgraph.graph import END, START, StateGraph

    from specunode.buffer.store_buffer import StoreBuffer
    from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
    from specunode.core.graph import routed
    from specunode.core.policy import Policy
    from specunode.core.scheduler import Scheduler
    from specunode.integrations.langgraph import LangGraphAdapter

    State = SideBySideState
    sent: list[str] = []

    async def notify_fn(who: str) -> dict[str, Any]:
        sent.append(who)
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="notify", effect=EffectClass.WRITE, fn=notify_fn, idempotent=False)
    )
    notify = routed(registry.get("notify"))

    async def a(state: State) -> State:
        await asyncio.sleep(0.2)  # retires last
        return {"log": ["a"]}

    async def b(state: State) -> State:
        await asyncio.sleep(0.01)
        await notify(who="ops")  # the same call on both visits
        return {"log": ["b"]}

    async def c(state: State) -> State:
        return {"loops": state.get("loops", 0) + 1, "log": ["c"]}

    graph = StateGraph(State)
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_node("c", c)
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", "c")
    graph.add_edge("b", "c")
    graph.add_conditional_edges("c", lambda s: "b" if s.get("loops", 0) < 2 else END)

    journal = Journal(tmp_path / "lg.db")
    run_id = new_ulid()
    result = await asyncio.wait_for(
        Scheduler(
            graph=LangGraphAdapter(compiled=graph.compile()),
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(ScriptedModel(turns=[]), journal, provider="scripted"),
            policy=Policy(speculation=False),
        ).run(run_id, {"loops": 0, "log": []}),
        timeout=30,
    )
    assert result.ok, result.error
    nodes = [e.payload.get("node_id") for e in journal.read(run_id, kinds=["branch_forked"])]
    assert len(nodes) == len(set(nodes)), f"a node id minted twice: {nodes}"
    assert sent == ["ops", "ops"], f"a write was deduped away: {sent}"


class SideBySideState(TypedDict, total=False):
    loops: int
    log: Annotated[list[str], operator.add]
