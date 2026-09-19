"""An on-call agent: read the pipeline and the runbook, decide, reserve, restart, report.

The second of the three sample apps, and it is here to be a *different shape* from
``support_agent`` rather than a second instance of it. Three differences carry the weight:

**It runs its turn through ``session.call_turn``.** ``support_agent`` calls the model directly
and then issues each tool itself, which is a supported pattern and the one Demo 1 uses -- but it
routes around tier-0 early issue and the drafters entirely, because those live in the turn the
runtime drives. This agent hands the whole turn to the runtime instead, so its reads are issued
as their blocks parse and its next call is predicted while the stream is still going. It is the
only sample app on which speculation actually happens, and having exactly one of each shape is
the point: a suite where every workload took the same path would test one path three times.

**Its turn emits several calls.** The drafter is asked after each block and sees the calls
within the current turn, so a one-call turn offers nothing to predict from. Emitting three is
what gives the pattern index a context to rank against -- and the middle call's *result* is
what lets it fill the third call's argument.

**Its write is idempotent and declared so.** ``restart_job`` promises that a repeat delivery
is harmless, so the ambiguous-crash window resolves by redelivery rather than by the dead
letter ``charge_card`` earns. Both outcomes are correct; only one of them is reachable from
``support_agent``.

**It reserves capacity through a COMPENSABLE tool.** Compensation is a second effect that
reaches the world *after* retirement -- it is not what makes speculation safe, and having one
workload that uses it keeps that distinction testable rather than merely documented.

``fetch_runbook`` is the unwitnessed read. There is no version counter to re-check at
retirement, so the ledger reports it as *unwitnessed* and never as fresh. That is attack 7.3's
shape, present here in a workload that is supposed to succeed.
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
)
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.testing.world import World

TOOL_DEFS = (
    ToolDef(name="get_pipeline_status", description="Read a pipeline's state", input_schema={}),
    ToolDef(name="fetch_runbook", description="Read a runbook section", input_schema={}),
    ToolDef(name="restart_job", description="Restart a failed pipeline job", input_schema={}),
    ToolDef(name="post_summary", description="Post an incident summary", input_schema={}),
)

#: Units reserved before a restart. A constant rather than a model-chosen number: the point of
#: this workload is the effect class, and a varying amount would only add noise to the ledger.
RESERVE_UNITS = 2


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

    @tool(effect=EffectClass.READ, witness=True, forward_keys="job:{args.pipeline_id}")
    async def get_pipeline_status(pipeline_id: str) -> JsonValue:
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    # No witness. A runbook page has no version counter, so staleness here is undetectable
    # rather than absent, and the ledger says "unwitnessed" instead of guessing "fresh".
    @tool(effect=EffectClass.READ, witness=False)
    async def fetch_runbook(section: str) -> JsonValue:
        return await world.fetch_runbook(section=section)

    @tool(
        effect=EffectClass.COMPENSABLE,
        compensator="release_capacity",
        idempotent=False,
        forward_keys="job:{args.job_id}",
    )
    async def reserve_capacity(job_id: str, units: int) -> JsonValue:
        return await world.reserve_capacity(job_id=job_id, units=units)

    @tool(effect=EffectClass.WRITE, idempotent=False, forward_keys="job:{args.job_id}")
    async def release_capacity(job_id: str, units: int) -> JsonValue:
        return await world.release_capacity(job_id=job_id, units=units)

    # Idempotent, and that is a promise about the upstream rather than a hint. Restarting a
    # job that is already running is a no-op there, so a redelivery after an ambiguous crash
    # is safe in a way a second card charge never is.
    @tool(effect=EffectClass.WRITE, idempotent=True, forward_keys="job:{args.job_id}")
    async def restart_job(job_id: str) -> JsonValue:
        return await world.restart_job(job_id=job_id)

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)

    return [
        get_pipeline_status,
        fetch_runbook,
        reserve_capacity,
        release_capacity,
        restart_job,
        post_summary,
    ]


@node(name="triage", emits="tool_call")
async def triage(session: RunSession) -> Decision:
    """One turn, three calls, driven by the runtime.

    The two reads are issued as their blocks parse rather than after the turn ends, and the
    restart is staged into the store buffer and dispatched only when this branch retires. The
    node gets the restart's real result back here -- it parks until the drain supplies it, and
    never sees the placeholder.
    """
    if session.call_turn is None:  # pragma: no cover - the scheduler always binds one
        raise ModelError("no turn runner was bound for this run")
    pipeline_id = str(session.state.get("pipeline_id", "etl-2"))
    results = await session.call_turn(
        RequestEnvelope(
            model=target_model(),
            system=(TextBlock(text="You are the on-call engineer for a data platform."),),
            messages=(
                Message(
                    role="user",
                    content=(TextBlock(text=f"Pipeline {pipeline_id} needs attention."),),
                ),
            ),
            tools=TOOL_DEFS,
            max_tokens=256,
            stream=True,
        )
    )
    session.state["pipeline_id"] = pipeline_id
    restart = results[-1] if results else None
    session.state["restarts"] = restart.get("applied") if isinstance(restart, Mapping) else None
    session.state["triaged"] = True
    return ToolCall("restart_job", {"job_id": pipeline_id})


@node(name="report", emits="tool_call")
async def report(session: RunSession) -> Decision:
    """Reserve the capacity the restarted job will need, then say what happened.

    Two mutating calls in one node, which is what makes the step cursor load-bearing: they
    differ in tool and arguments, but a runtime deriving one key per *node* rather than per
    *call* would collide them and silently drop one.
    """
    job_id = str(session.state.get("pipeline_id", "etl-2"))
    await session.call_tool("reserve_capacity", {"job_id": job_id, "units": RESERVE_UNITS})
    text = f"restarted {job_id} (applied={session.state.get('restarts')})"
    await session.call_tool("post_summary", {"channel": "#incidents", "text": text})
    session.state["reported"] = True
    return ToolCall("post_summary", {"channel": "#incidents", "text": text})


ORDER = ("triage", "report")


def route(state: Mapping[str, JsonValue]) -> str | None:
    if "triaged" not in state:
        return "triage"
    if "reported" not in state:
        return "report"
    return None


def build(world: World) -> tuple[PlainAdapter, object]:
    """The graph and the registry its tools are declared in."""
    tools = build_tools(world)
    adapter = PlainAdapter.of([triage, report], route)
    return adapter, registry_of(tools)  # type: ignore[arg-type]
