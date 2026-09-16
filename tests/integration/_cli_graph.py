"""A graph the CLI can reach by ``module:attribute``, for the resume/replay tests.

Deliberately **stateless**. The CLI resolves its graph by importing a module by name, and pytest
imports a test module under a different name than the CLI does -- ``integration.test_cli_replay``
against ``tests.integration.test_cli_replay`` -- so the two end up holding separate copies of the
same module and anything stored in a module global is invisible across the boundary. That cost an
hour once; everything this module needs comes from the environment and from disk instead, which
are the two things both copies genuinely share.
"""

from __future__ import annotations

import os
from pathlib import Path

from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import Message, RequestEnvelope, TextBlock
from specunode.testing.world import World

#: The system prompt the graph sends. The replay test changes it and expects to be refused.
SYSTEM_ENV = "SPECUNODE_TEST_SYSTEM"
#: Where the world writes its durable log, so a separate process -- or a separate copy of this
#: module -- can see what actually reached it.
WORLD_ENV = "SPECUNODE_TEST_WORLD"

DEFAULT_SYSTEM = "You are the on-call engineer."
TURN = (("restart_job", {"job_id": "etl-1"}),)


def system_prompt() -> str:
    return os.environ.get(SYSTEM_ENV, DEFAULT_SYSTEM)


def world_for(path: Path | None = None) -> World:
    log = path or Path(os.environ[WORLD_ENV])
    world = World(log_path=log)
    if not world.tables["jobs"]:
        world.seed("jobs", "etl-1", job_id="etl-1", status="failed", restarts=0, reserved=0)
    return world


def registry_for(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    return registry


class SystemGraph:
    """One node, one turn, one write -- with a system prompt that can be changed."""

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(drives_itself=False, framework="plain")

    def nodes(self) -> list[NodeRef]:
        return [NodeRef(name="agent")]

    def decision_kind(self, node: NodeRef) -> str:
        return "tool_call"

    def next(self, state: object) -> NextNode:
        return END if isinstance(state, dict) and state.get("done") else NodeRef(name="agent")

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        assert session.call_turn is not None
        await session.call_turn(
            RequestEnvelope(
                model="scripted",
                system=(TextBlock(text=system_prompt()),),
                messages=(Message(role="user", content=(TextBlock(text="go"),)),),
                max_tokens=128,
                stream=True,
            )
        )
        session.state["done"] = True
        return ToolCall("restart_job", {"job_id": "etl-1"})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


def build() -> tuple[SystemGraph, ToolRegistry]:
    """The ``module:attribute`` target a test config points at."""
    world = world_for()
    return SystemGraph(), registry_for(world)
