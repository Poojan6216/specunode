"""The same support agent, written as an ordinary LangGraph ``StateGraph``.

This file is the point of spec task 2.3: it is a normal graph. It builds nodes and edges the
way any LangGraph app does, and it runs unchanged whether or not SpecuNode is wrapping it --
unwrapped, the tools call straight through and the model is the one bound here; wrapped, the
same calls are classified, staged and journaled, and the run produces a ledger.

The two things the developer does do are the two Hard Rule 2 asks for: declare each tool's
effect class, and reach the tools through ``routed`` so a runtime can see them. Neither is a
change to the graph's shape, and neither has any effect when nothing is wrapping it.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from specunode.canonical import JsonValue
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec, forward_keys_from_template
from specunode.core.graph import current_session, routed
from specunode.core.model import (
    Message,
    ModelClient,
    RequestEnvelope,
    TextBlock,
    ToolDef,
    decisions_of,
)
from specunode.testing.world import World

TOOL_DEFS = (
    ToolDef(name="charge_card", description="Charge a customer's card", input_schema={}),
    ToolDef(name="send_receipt", description="Email a receipt", input_schema={}),
)


def target_model() -> str:
    """Which model this example asks for.

    ``"scripted"`` by default, so the tests and demos stay hermetic and deterministic. The
    online latency bench sets ``SPECUNODE_MODEL`` to a real id; nothing else does, and the
    runtime never rewrites it -- Hard Rule 13 makes the request the unit of identity, so a
    scheduler that substituted a model would make the journal's record of what was asked
    untrue. Read at call time rather than at import, because the bench sets it after this
    module is imported.
    """
    return os.environ.get("SPECUNODE_MODEL", "scripted")


class SupportState(TypedDict, total=False):
    customer_id: str
    customer: Any
    decided: Any
    receipted: bool
    log: Annotated[list[str], operator.add]


def build_registry(world: World) -> ToolRegistry:
    """Declare what each tool does to the world. Never inferred (Hard Rule 2)."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup_customer",
            effect=EffectClass.READ,
            fn=world.lookup_customer,
            witness=True,
            forward_keys=forward_keys_from_template("customer:{args.customer_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="charge_card",
            effect=EffectClass.WRITE,
            fn=world.charge_card,
            idempotent=False,
            forward_keys=forward_keys_from_template("customer:{args.customer_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="send_receipt",
            effect=EffectClass.WRITE,
            fn=world.send_receipt,
            forward_keys=forward_keys_from_template("customer:{args.customer_id}"),
        )
    )
    return registry


def build_graph(world: World, model: ModelClient) -> Any:
    """An ordinary compiled StateGraph. Nothing here knows about speculation."""
    from langgraph.graph import END, StateGraph

    registry = build_registry(world)
    lookup_customer = routed(registry.get("lookup_customer"))
    charge_card = routed(registry.get("charge_card"))
    send_receipt = routed(registry.get("send_receipt"))

    def _model() -> ModelClient:
        session = current_session()
        return session.model if session is not None and session.model is not None else model

    async def lookup(state: SupportState) -> SupportState:
        customer = await lookup_customer(customer_id=state.get("customer_id", "cus-1"))
        return {"customer": customer, "log": ["lookup"]}

    async def decide(state: SupportState) -> SupportState:
        envelope = RequestEnvelope(
            model=target_model(),
            system=(TextBlock(text="You handle refunds and charges for a support desk."),),
            messages=(
                Message(
                    role="user",
                    content=(TextBlock(text=f"Customer record: {state.get('customer')}"),),
                ),
            ),
            tools=TOOL_DEFS,
            max_tokens=256,
        )
        decision = decisions_of(await _model().complete(envelope))[0]
        args: Mapping[str, JsonValue] = decision.args if isinstance(decision, ToolCall) else {}
        return {"decided": dict(args), "log": ["decide"]}

    async def charge(state: SupportState) -> SupportState:
        decided = state.get("decided") or {}
        customer_id = str(decided.get("customer_id", "cus-1"))
        amount = float(decided.get("amount", 10.0))
        ack = await charge_card(customer_id=customer_id, amount=amount)
        charge_id = ack.get("charge_id") if isinstance(ack, Mapping) else None
        await send_receipt(customer_id=customer_id, charge_id=str(charge_id))
        return {"receipted": True, "log": ["charge"]}

    graph = StateGraph(SupportState)
    graph.add_node("lookup", lookup)
    graph.add_node("decide", decide)
    graph.add_node("charge", charge)
    graph.set_entry_point("lookup")
    graph.add_edge("lookup", "decide")
    graph.add_edge("decide", "charge")
    graph.add_edge("charge", END)
    return graph.compile()
