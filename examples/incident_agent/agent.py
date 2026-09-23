"""An on-call agent that works an alert end to end, in as few model turns as the task allows.

The task: four pipelines alerted. Check each one, read the restart runbook, restart the ones
that failed, post one summary. Nine model turns if the model asks for one call at a time --
five reads, two restarts, a summary and a closing reply -- and four if it asks for every
independent call at once. When the model is the slow part of a run, that difference is the
whole of what infrastructure can buy, and this app exists to measure it.

It runs one node, which drives the conversation with :func:`specunode.core.loop.agent_loop`.
Every turn is journaled; the reads a turn asks for are issued as their blocks parse and run
concurrently; its writes are staged and leave only once the turn that asked for them is
durable. None of that changes with the prompting style -- only how many turns there are.

``STYLES`` are the three prompts the benchmark compares. ``one_call`` is how the 300 real
agent trajectories in ``bench/corpus`` behave: every turn in them makes exactly one call.
``default`` says nothing either way, so it measures the model's own habit. ``parallel`` gives
the guidance Anthropic's tool-use documentation gives for independent calls.

``one_call`` is enforced, not only asked for. Told in its system prompt to make exactly one
call per reply, Claude Sonnet 5 asked for several at once anyway -- ten calls in five replies,
the same as with no guidance at all -- so a benchmark that relied on the prompt compared the
treatment with itself. The style also sets the API's own switch, ``disable_parallel_tool_use``,
which is what an app that runs one call per reply effectively has.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText
from specunode.core.effects import EffectClass
from specunode.core.graph import RunSession
from specunode.core.loop import LoopResult, agent_loop
from specunode.core.model import (
    Message,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    TextBlock,
    ToolDef,
    ToolUseBlock,
    Usage,
)
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.testing.world import World

#: Room for a model that thinks before it answers; 256 was not enough to reach a tool call.
MAX_TOKENS = 4096
MAX_TURNS = 16

PIPELINES = ("etl-1", "etl-2", "etl-3", "etl-4")

#: The operating guide an on-call agent actually carries. It is long because production
#: system prompts are: rules for what may be restarted, what to read first, how to escalate,
#: what a summary must contain. It is also *stable* -- no date, no run id, nothing that changes
#: between requests -- because the prompt cache keys on the exact prefix, and a timestamp at
#: the top of a system prompt silently makes every request a cache miss.
OPERATING_GUIDE = """You are the on-call engineer for a data platform that runs scheduled ETL \
pipelines. You respond to alerts by investigating, taking the minimum safe action, and \
reporting what you did. You act only through the tools you are given.

## What the tools do

- get_pipeline_status(pipeline_id) returns the pipeline's current state: running, queued, or \
failed, with its restart count. It is read-only and safe to call at any time.
- fetch_runbook(section) returns a section of the operations runbook. The sections are \
"restart", "escalate" and "billing". Read the relevant section before you act on a pipeline; \
the runbook is the source of truth for procedure, and it changes more often than this guide.
- restart_job(job_id) restarts a pipeline's job. It changes production. Restarting a job that \
is already running is harmless, but restarting one that is healthy wastes a slot and pages \
the owning team, so only restart what is actually failed.
- post_summary(channel, text) posts a message to a team channel. It is visible to the whole \
team and cannot be edited or deleted.

## Operating rules

1. Establish the facts before acting. Check the status of every pipeline named in the alert \
before restarting anything. An alert names what fired, not what is broken; queued and \
running pipelines are normal states and need no action.
2. Follow the runbook. Read the restart section before your first restart in an incident, \
and do what it says. If it tells you to escalate instead of restarting, escalate.
3. Restart only pipelines whose status is failed. Never restart a running or queued \
pipeline. Never restart the same pipeline twice in one incident.
4. Do not guess identifiers. Use exactly the pipeline ids the alert names. The job id for a \
pipeline is the same as its pipeline id.
5. Report once, at the end. Post a single summary to #ops after you have finished acting, \
listing each pipeline you checked, its status, and whether you restarted it. Do not post \
progress updates; the channel is shared and noisy summaries get ignored.
6. If a tool returns an error, do not retry blindly. Read the error, decide whether the \
action is still correct, and say in your summary what failed and what you did about it.
7. You are done when the summary is posted. Reply with one sentence confirming what you did, \
and do not call any further tools.

## Escalation

Escalate instead of restarting when a pipeline has failed twice or more in the same day, \
when the runbook says so, or when a restart itself fails. Escalation means saying so clearly \
in the summary, naming the pipeline and the reason, so the owning team sees it; you do not \
page anyone directly.

## The summary

Keep it short and factual: one line per pipeline, in the order the alert named them, then \
one line saying what was restarted. No speculation about root cause unless a tool result \
states one. Name pipelines by id. The team reads these on a phone during an incident, so \
lead with what changed.

## Judgement

Prefer the least invasive action that resolves the alert. When the facts are unclear, gather \
more of them before acting rather than after. Everything you do is recorded, and a reviewer \
will read the record against these rules."""

#: How each style asks the model to structure its calls. Appended after the guide, so the
#: guide itself -- the long, cacheable part -- is byte-identical across styles.
STYLES: Mapping[str, str] = {
    "one_call": (
        "Work one step at a time: make exactly one tool call per reply, and wait for its "
        "result before deciding on the next."
    ),
    "default": "",
    "parallel": (
        "When several tool calls do not depend on each other's results, make all of them in "
        "the same reply rather than one per reply. Only wait for a result when a later call "
        "actually needs it."
    ),
}

TASK = (
    "Alerts fired for pipelines etl-1, etl-2, etl-3 and etl-4. Check each pipeline's status "
    "and read the restart runbook. Restart every pipeline whose status is failed, then post "
    "one summary to #ops."
)

TOOL_DEFS = (
    ToolDef(
        name="get_pipeline_status",
        description="Read a pipeline's current state. Read-only.",
        input_schema={
            "type": "object",
            "properties": {"pipeline_id": {"type": "string"}},
            "required": ["pipeline_id"],
        },
    ),
    ToolDef(
        name="fetch_runbook",
        description="Read a section of the operations runbook. Read-only.",
        input_schema={
            "type": "object",
            "properties": {"section": {"type": "string"}},
            "required": ["section"],
        },
    ),
    ToolDef(
        name="restart_job",
        description="Restart a failed pipeline's job. Changes production.",
        input_schema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    ),
    ToolDef(
        name="post_summary",
        description="Post a message to a team channel. Visible to the team; cannot be edited.",
        input_schema={
            "type": "object",
            "properties": {"channel": {"type": "string"}, "text": {"type": "string"}},
            "required": ["channel", "text"],
        },
    ),
)


def target_model() -> str:
    """``"scripted"`` unless the benchmark names a real model; read at call time."""
    return os.environ.get("SPECUNODE_MODEL", "scripted")


def system_prompt(style: str) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLES)}")
    guidance = STYLES[style]
    return OPERATING_GUIDE if not guidance else f"{OPERATING_GUIDE}\n\n## Tool calls\n\n{guidance}"


#: The API's switch for at most one tool call per reply. Part of the request hash, like
#: everything else the model is asked, so a replay notices if it changes.
ONE_CALL_PER_REPLY: Mapping[str, JsonValue] = {"type": "auto", "disable_parallel_tool_use": True}


def envelope(style: str) -> RequestEnvelope:
    return RequestEnvelope(
        model=target_model(),
        system=(TextBlock(text=system_prompt(style)),),
        messages=(Message(role="user", content=(TextBlock(text=TASK),)),),
        tools=TOOL_DEFS,
        tool_choice=ONE_CALL_PER_REPLY if style == "one_call" else None,
        max_tokens=MAX_TOKENS,
        stream=True,
    )


def build_tools(world: World) -> list[object]:
    """The world's tools, with the effect classes this agent asserts for them."""

    @tool(effect=EffectClass.READ, witness=True, forward_keys="job:{args.pipeline_id}")
    async def get_pipeline_status(pipeline_id: str) -> JsonValue:
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    @tool(effect=EffectClass.READ, witness=False)
    async def fetch_runbook(section: str) -> JsonValue:
        return await world.fetch_runbook(section=section)

    @tool(effect=EffectClass.WRITE, idempotent=True, forward_keys="job:{args.job_id}")
    async def restart_job(job_id: str) -> JsonValue:
        return await world.restart_job(job_id=job_id)

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)

    return [get_pipeline_status, fetch_runbook, restart_job, post_summary]


def build(world: World, style: str = "parallel") -> tuple[PlainAdapter, object]:
    """The graph for one style: a single node that works the alert to completion."""
    system_prompt(style)  # fail at build time on an unknown style, not mid-run

    @node(name="respond", emits="tool_call")
    async def respond(session: RunSession) -> Decision:
        if session.call_turn is None:  # pragma: no cover - the scheduler always binds one
            raise ModelError("no turn runner was bound for this run")
        result: LoopResult = await agent_loop(session, envelope(style), max_turns=MAX_TURNS)
        session.state["turns"] = result.turns
        session.state["calls"] = result.calls
        session.state["stopped"] = result.stopped
        session.state["done"] = True
        return FreeText.of("responded")

    def route(state: Mapping[str, JsonValue]) -> str | None:
        return None if state.get("done") else "respond"

    adapter = PlainAdapter.of([respond], route)
    return adapter, registry_of(build_tools(world))  # type: ignore[arg-type]


def expected_world_changes() -> dict[str, object]:
    """What a correct run does, whatever its style: two restarts and one summary."""
    return {"restarted": ["etl-2", "etl-4"], "summaries": 1}


# -- the scripted replies ----------------------------------------------------------------------


#: One tool call a scripted reply asks for: the tool's name and its arguments.
Call = tuple[str, Mapping[str, JsonValue]]


def _reply(calls: Sequence[Call], turn: int) -> ModelResponse:
    content: tuple[object, ...] = tuple(
        ToolUseBlock(id=f"toolu_{turn:02d}_{index}", name=name, args=dict(args))
        for index, (name, args) in enumerate(calls)
    )
    if not calls:
        content = (TextBlock(text="Restarted etl-2 and etl-4; summary posted to #ops."),)
    return ModelResponse(
        model="scripted",
        content=content,  # type: ignore[arg-type]
        stop_reason="tool_use" if calls else "end_turn",
        usage=Usage(input_tokens=1800, output_tokens=60),
    )


def scripted_replies(style: str) -> list[ModelResponse]:
    """What a model following each style would reply, turn by turn, for the deterministic tests.

    The scripted model plays these whatever it is sent, so they show what the *runtime* does
    with each shape of reply. What a real model actually does with each style is what the
    benchmark measures; nothing here is evidence about that.
    """
    reads: list[Call] = [("get_pipeline_status", {"pipeline_id": p}) for p in PIPELINES]
    runbook: Call = ("fetch_runbook", {"section": "restart"})
    restarts: list[Call] = [
        ("restart_job", {"job_id": "etl-2"}),
        ("restart_job", {"job_id": "etl-4"}),
    ]
    summary: Call = (
        "post_summary",
        {
            "channel": "#ops",
            "text": "etl-1 running; etl-2 failed, restarted; etl-3 queued; "
            "etl-4 failed, restarted.",
        },
    )
    if style == "parallel":
        turns: list[list[Call]] = [
            [*reads, runbook],
            restarts,
            [summary],
            [],
        ]
    else:
        turns = [[call] for call in (*reads, runbook, *restarts, summary)] + [[]]
    return [_reply(calls, index) for index, calls in enumerate(turns)]
