"""A support agent: look the customer up, decide, charge the card, send the receipt.

Demo 1's workload. It is deliberately the uncomfortable shape -- the write is a card charge,
it is not idempotent, and the node that issues it immediately reads the charge id back out to
send a receipt. That last part is the case a store buffer has to get right: the node cannot be
handed a real result before the branch retires, and it cannot be left waiting for one either.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText, ToolCall
from specunode.core.effects import EffectClass
from specunode.core.graph import RunSession
from specunode.core.model import (
    Message,
    ModelError,
    RequestEnvelope,
    TextBlock,
    ToolDef,
    decisions_of,
)
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.testing.world import World

#: Real JSON Schema, not ``{}``. A tool's schema is what tells the model how to fill the call,
#: and an empty one is rejected outright by the Messages API
#: (``tools.0.custom.input_schema.type: Field required``). These examples ran only against a
#: scripted model, which never looked, so the placeholder survived until the first real call.
#: Room for the answer. 256 was enough for a scripted model, which emits a tool call and
#: nothing else; a current model thinks before it answers and spent the whole budget doing so,
#: returning prose with ``stop_reason: "max_tokens"`` and no tool call at all.
MAX_TOKENS = 4096

TOOL_DEFS = (
    ToolDef(
        name="charge_card",
        description="Charge a customer's card",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["customer_id", "amount"],
        },
    ),
    ToolDef(
        name="send_receipt",
        description="Email a receipt",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "charge_id": {"type": "string"}},
            "required": ["customer_id", "charge_id"],
        },
    ),
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


def build_tools(world: World) -> list[object]:
    """Declare the world's tools with the effect classes this agent asserts for them."""

    @tool(effect=EffectClass.READ, witness=True, forward_keys="customer:{args.customer_id}")
    async def lookup_customer(customer_id: str) -> JsonValue:
        return await world.lookup_customer(customer_id=customer_id)

    # Not idempotent, and declared so. A tool that has not promised a repeat delivery is
    # harmless must not be redispatched after an ambiguous crash; the honest outcome there is
    # a dead letter and a human, not a second charge.
    @tool(effect=EffectClass.WRITE, idempotent=False, forward_keys="customer:{args.customer_id}")
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        return await world.charge_card(customer_id=customer_id, amount=amount)

    @tool(effect=EffectClass.WRITE, forward_keys="customer:{args.customer_id}")
    async def send_receipt(customer_id: str, charge_id: str) -> JsonValue:
        return await world.send_receipt(customer_id=customer_id, charge_id=charge_id)

    return [lookup_customer, charge_card, send_receipt]


@node(name="lookup", emits="tool_call")
async def lookup(session: RunSession) -> Decision:
    customer_id = str(session.state.get("customer_id", "cus-1"))
    result = await session.call_tool("lookup_customer", {"customer_id": customer_id})
    session.state["customer"] = result
    return ToolCall("lookup_customer", {"customer_id": customer_id})


@node(name="decide", emits="tool_call")
async def decide(session: RunSession) -> Decision:
    """Ask the model what to do. Its answer is the ground truth a branch resolves against."""
    if session.model is None:  # pragma: no cover - the scheduler always binds one
        raise ModelError("no model was bound for this run")
    envelope = RequestEnvelope(
        model=target_model(),
        system=(TextBlock(text="You handle refunds and charges for a support desk."),),
        messages=(
            Message(
                role="user",
                content=(
                    # An actual instruction, not a data dump. The first real run of this
                    # app showed the model a customer record and nothing else; it
                    # declined, correctly -- a record is not a request to charge anyone --
                    # and the workload measured a refusal. A scripted model never noticed,
                    # because it plays its turn whatever the prompt says.
                    TextBlock(
                        text=(
                            f"Customer record: {session.state.get('customer')}\n"
                            "Charge this customer 25.00 for their plan renewal, then send "
                            "them a receipt. Use the tools; do not ask for confirmation."
                        )
                    ),
                ),
            ),
        ),
        tools=TOOL_DEFS,
        max_tokens=MAX_TOKENS,
    )
    decision = decisions_of(await session.model.complete(envelope))[0]
    if isinstance(decision, ToolCall):
        session.state["decided"] = dict(decision.args)
    else:
        # A turn with no tool call is a FreeText barrier, and this node's router keys on
        # ``decided``. Without this branch the router sent the run straight back here and the
        # app re-asked the same question forever -- one model call per lap, which against a
        # real provider is an unbounded bill rather than a hang. It happens: a model that
        # thinks by default can spend its whole ``max_tokens`` budget before reaching a call.
        session.state["decided"] = None
        if isinstance(decision, FreeText):
            # The prose itself lives in the journal; the decision carries only its hash.
            session.state["declined"] = decision.content_hash
    return decision


@node(name="charge", emits="tool_call")
async def charge(session: RunSession) -> Decision:
    """Charge, then receipt the charge id the first call returned.

    The second call's arguments come from the first call's result, which does not exist until
    the branch retires and the store buffer drains. The node parks here rather than receiving
    a placeholder, so the receipt carries the real id and never a synthetic one.
    """
    decided = session.state.get("decided")
    args: Mapping[str, JsonValue] = decided if isinstance(decided, Mapping) else {}
    customer_id = str(args.get("customer_id", "cus-1"))
    amount = float(args.get("amount", 10.0))  # type: ignore[arg-type]

    ack = await session.call_tool("charge_card", {"customer_id": customer_id, "amount": amount})
    charge_id = ack.get("charge_id") if isinstance(ack, Mapping) else None
    await session.call_tool(
        "send_receipt", {"customer_id": customer_id, "charge_id": str(charge_id)}
    )
    session.state["receipted"] = True
    return ToolCall("send_receipt", {"customer_id": customer_id, "charge_id": str(charge_id)})


ORDER = ("lookup", "decide", "charge")


def route(state: Mapping[str, JsonValue]) -> str | None:
    if "customer" not in state:
        return "lookup"
    if "decided" not in state:
        return "decide"
    # A model that answered in prose asked for nothing, so nothing is sent. The fallback
    # arguments below this line exist for a decision the model *made*; reaching them with no
    # decision at all would put an effect in the world on arguments this app invented, which
    # is precisely what the rest of the project exists to prevent.
    if state.get("decided") is None:
        return None
    if "receipted" not in state:
        return "charge"
    return None


def build(world: World) -> tuple[PlainAdapter, object]:
    """The graph and the registry its tools are declared in."""
    tools = build_tools(world)
    adapter = PlainAdapter.of([lookup, decide, charge], route)
    return adapter, registry_of(tools)  # type: ignore[arg-type]
