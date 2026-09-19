"""A research agent: search, read, look the requester up, decide, then mail and post.

The third sample app, and the read-heavy end of the range. ``support_agent`` reaches a write
in its second node and ``ops_agent`` opens with two reads; this one runs four reads across two
nodes before any model turn that could stage anything. That matters because read-only stretches
are the *only* place model latency is hidden at all -- everywhere else speculation hides tool
latency and nothing more. A workload suite that never contains one cannot show the difference.

It is also the only sample app with an IRREVERSIBLE effect. ``send_email`` has no compensator
because there is no action that unsends an email, so under the default policy it is a barrier
rather than something the store buffer will hold: the branch must retire before it is even
staged. ``policy.stage_irreversible`` can change that, and attack 7.9 measures what changing it
costs. Having the default exercised by a passing workload is what keeps the attack honest --
otherwise the only evidence about irreversible effects would come from the run designed to
break them.

``search_docs`` is a read with no row behind it: it scans a table and returns matching ids.
There is no single version counter that would tell a retirement check whether the result went
stale, so like ``fetch_runbook`` it is unwitnessed, and for a structurally different reason.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
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

TOOL_DEFS = (
    ToolDef(name="send_email", description="Email the findings to the requester", input_schema={}),
    ToolDef(name="post_summary", description="Post the findings to a channel", input_schema={}),
)

#: The sections this agent always reads, in this order. Fixed rather than model-chosen: the
#: workload's job is to present a read-only stretch of known length, and a varying one would
#: make "how much of this workload is speculable" a property of the script instead.
SECTIONS = ("restart", "escalate")


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

    # Unwitnessed for a different reason than fetch_runbook: this reads no single row, so
    # there is no version counter that re-checking at retirement could compare against.
    @tool(effect=EffectClass.READ, witness=False)
    async def search_docs(query: str) -> JsonValue:
        return await world.search_docs(query=query)

    @tool(effect=EffectClass.READ, witness=False)
    async def fetch_runbook(section: str) -> JsonValue:
        return await world.fetch_runbook(section=section)

    @tool(effect=EffectClass.READ, witness=True, forward_keys="customer:{args.customer_id}")
    async def lookup_customer(customer_id: str) -> JsonValue:
        return await world.lookup_customer(customer_id=customer_id)

    # No compensator, and none is possible. Under the default policy this is a barrier: the
    # branch retires first and the effect is dispatched after, rather than being held.
    @tool(effect=EffectClass.IRREVERSIBLE, idempotent=False, forward_keys="customer:{args.to}")
    async def send_email(to: str, subject: str, body: str) -> JsonValue:
        return await world.send_email(to=to, subject=subject, body=body)

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)

    return [search_docs, fetch_runbook, lookup_customer, send_email, post_summary]


@node(name="search", emits="tool_call")
async def search(session: RunSession) -> Decision:
    """One search, then a fetch per section. Three reads, no writes, no model turn."""
    query = str(session.state.get("query", "Restart"))
    hits = await session.call_tool("search_docs", {"query": query})
    session.state["hits"] = hits
    pages: list[JsonValue] = []
    for section in SECTIONS:
        pages.append(await session.call_tool("fetch_runbook", {"section": section}))
    session.state["pages"] = pages
    return ToolCall("fetch_runbook", {"section": SECTIONS[-1]})


@node(name="identify", emits="tool_call")
async def identify(session: RunSession) -> Decision:
    """The one witnessed read in this workload, so staleness is detectable for exactly one."""
    customer_id = str(session.state.get("customer_id", "cus-3"))
    session.state["requester"] = await session.call_tool(
        "lookup_customer", {"customer_id": customer_id}
    )
    return ToolCall("lookup_customer", {"customer_id": customer_id})


@node(name="decide", emits="tool_call")
async def decide(session: RunSession) -> Decision:
    """The model's real decision, taken with every read result already in the prompt."""
    if session.model is None:  # pragma: no cover - the scheduler always binds one
        raise ModelError("no model was bound for this run")
    envelope = RequestEnvelope(
        model=target_model(),
        system=(TextBlock(text="You research operational questions and report findings."),),
        messages=(
            Message(
                role="user",
                content=(
                    TextBlock(
                        text=(
                            f"Hits: {session.state.get('hits')}\n"
                            f"Pages: {session.state.get('pages')}\n"
                            f"Requester: {session.state.get('requester')}"
                        )
                    ),
                ),
            ),
        ),
        tools=TOOL_DEFS,
        max_tokens=256,
    )
    decision = decisions_of(await session.model.complete(envelope))[0]
    if isinstance(decision, ToolCall):
        session.state["decided"] = dict(decision.args)
    return decision


@node(name="report", emits="tool_call")
async def report(session: RunSession) -> Decision:
    """Mail the requester, then post the same finding to a channel.

    The email is irreversible and the post is an ordinary write, so this node stages one of its
    two effects and barriers on the other. Both still appear in the ledger in dispatch order,
    which is the property the equivalence relation compares.
    """
    decided = session.state.get("decided")
    args: Mapping[str, JsonValue] = decided if isinstance(decided, Mapping) else {}
    to = str(args.get("to", "cus-3"))
    subject = str(args.get("subject", "Findings"))
    body = str(args.get("body", "See the runbook."))

    await session.call_tool("send_email", {"to": to, "subject": subject, "body": body})
    text = f"emailed {to}: {subject}"
    await session.call_tool("post_summary", {"channel": "#research", "text": text})
    session.state["reported"] = True
    return ToolCall("post_summary", {"channel": "#research", "text": text})


ORDER = ("search", "identify", "decide", "report")


def route(state: Mapping[str, JsonValue]) -> str | None:
    if "pages" not in state:
        return "search"
    if "requester" not in state:
        return "identify"
    if "decided" not in state:
        return "decide"
    if "reported" not in state:
        return "report"
    return None


def build(world: World) -> tuple[PlainAdapter, object]:
    """The graph and the registry its tools are declared in."""
    tools = build_tools(world)
    adapter = PlainAdapter.of([search, identify, decide, report], route)
    return adapter, registry_of(tools)  # type: ignore[arg-type]
