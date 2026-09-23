"""Three independent investigations, then one report: the shape parallel nodes are for.

Each ``check_*`` node asks the model about one pipeline and reads its status. None of them
needs another's answer, so the router names all three at once and the runtime runs them side
by side (:class:`specunode.core.graph.Parallel`). The ``report`` node then reads all three
findings and posts one summary.

When the model is the slow part of a run, three model turns one after another cost three turns
of wall clock; side by side they cost one. Nothing else changes: each node still forks from the
same state, draws the same program positions and idempotency keys, and the three retire in the
order the router named them -- so their effects reach the world in that order however the
scheduler happened to interleave them.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field

from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText, ToolCall
from specunode.core.effects import EffectClass
from specunode.core.graph import RunSession
from specunode.core.model import (
    Message,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolDef,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
    Usage,
)
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.testing.world import World

PIPELINES = ("etl-1", "etl-2", "etl-3")
MAX_TOKENS = 4096

SYSTEM = (
    "You are one member of an on-call team. You have been given a single pipeline to check. "
    "Read its status with the tool, and say in one sentence whether it needs a restart."
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
)


def target_model() -> str:
    return os.environ.get("SPECUNODE_MODEL", "scripted")


def _check_node(pipeline: str) -> object:
    @node(name=f"check_{pipeline.replace('-', '_')}", emits="tool_call")
    async def check(session: RunSession) -> Decision:
        if session.call_turn is None:  # pragma: no cover - the scheduler always binds one
            raise ModelError("no turn runner was bound for this run")
        results = await session.call_turn(
            RequestEnvelope(
                model=target_model(),
                system=(TextBlock(text=SYSTEM),),
                messages=(Message(role="user", content=(TextBlock(text=f"Check {pipeline}."),)),),
                tools=TOOL_DEFS,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
        )
        # One key per node: nodes declared independent may not write the same state key.
        session.state[f"status:{pipeline}"] = results[-1] if results else None
        return ToolCall("get_pipeline_status", {"pipeline_id": pipeline})

    return check


def build_tools(world: World) -> list[object]:
    @tool(effect=EffectClass.READ, witness=True, forward_keys="job:{args.pipeline_id}")
    async def get_pipeline_status(pipeline_id: str) -> JsonValue:
        return await world.get_pipeline_status(pipeline_id=pipeline_id)

    @tool(effect=EffectClass.WRITE, idempotent=False)
    async def post_summary(channel: str, text: str) -> JsonValue:
        return await world.post_summary(channel=channel, text=text)

    return [get_pipeline_status, post_summary]


@node(name="report", emits="tool_call")
async def report(session: RunSession) -> Decision:
    lines = []
    for pipeline in PIPELINES:
        found = session.state.get(f"status:{pipeline}")
        value = found.get("value") if isinstance(found, Mapping) else None
        status = value.get("status") if isinstance(value, Mapping) else "unknown"
        lines.append(f"{pipeline}: {status}")
    text = "; ".join(lines)
    await session.call_tool("post_summary", {"channel": "#ops", "text": text})
    session.state["reported"] = True
    return FreeText.of(text)


def route(state: Mapping[str, JsonValue]) -> str | list[str] | None:
    pending = [f"check_{p.replace('-', '_')}" for p in PIPELINES if f"status:{p}" not in state]
    if pending:
        return pending
    return None if state.get("reported") else "report"


def build(world: World) -> tuple[PlainAdapter, object]:
    checks = [_check_node(p) for p in PIPELINES]
    adapter = PlainAdapter.of([*checks, report], route)  # type: ignore[list-item]
    return adapter, registry_of(build_tools(world))  # type: ignore[arg-type]


@dataclass
class KeyedScriptedModel:
    """A stand-in model that answers by *what it was asked*, not by when.

    Nodes running side by side call the model in whatever order the scheduler interleaves
    them, so a model that replays a fixed list of turns in call order would hand one node
    another's reply. This one reads the pipeline out of the request. It also counts how many
    requests are in flight at once, which is how a test proves the nodes really overlapped
    without timing anything.
    """

    think_ms: float = 0.0
    in_flight: int = 0
    high_water: int = 0
    calls: list[str] = field(default_factory=list)

    def _reply_for(self, envelope: RequestEnvelope) -> ModelResponse:
        text = "".join(
            block.text
            for message in envelope.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        found = re.search(r"etl-\d+", text)
        if found is None:
            raise ModelError("the keyed stand-in could not tell which pipeline it was asked about")
        pipeline = found.group(0)
        self.calls.append(pipeline)
        return ModelResponse(
            model="scripted",
            content=(
                ToolUseBlock(
                    id=f"toolu_{pipeline}",
                    name="get_pipeline_status",
                    args={"pipeline_id": pipeline},
                ),
            ),
            stop_reason="tool_use",
            usage=Usage(input_tokens=120, output_tokens=20),
        )

    async def _think(self) -> None:
        self.in_flight += 1
        self.high_water = max(self.high_water, self.in_flight)
        try:
            await asyncio.sleep(self.think_ms / 1000.0)
        finally:
            self.in_flight -= 1

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        await self._think()
        return self._reply_for(envelope)

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        await self._think()
        response = self._reply_for(envelope)
        for index, block in enumerate(response.content):
            if isinstance(block, ToolUseBlock):
                yield ToolUseComplete(index=index, block=block)
        yield TurnComplete(response=response)
