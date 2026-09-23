"""A deterministic stand-in for the target model, so the latency harness runs without a key.

It is not a latency measurement and the runner labels every report it produces accordingly.
What it *is* for: exercising the arms, the timing, the spend accounting, the budget gate and
the bootstrap, so that the code path is known to work before anyone spends money on it. The
MCP proxy in this project shipped for weeks unable to start because only its rules were tested
and never its transport; a benchmark that cannot run without a credential is the same trap.

It answers whatever the workload asks for by reading the turn out of the envelope's declared
tools, so it stays correct as the workloads change rather than pinning a script per workload.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from bench.workloads import WORKLOADS
from specunode.core.model import (
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    ToolUseComplete,
    TurnComplete,
)

#: A small, fixed think-time so the arms are distinguishable at all. Real model latency is
#: three orders of magnitude larger and far noisier; nothing here is a claim about it.
THINK_MS = 4.0


@dataclass
class ScriptedTarget:
    """Plays each workload's declared turn, matched by the tools the envelope offers."""

    calls: int = 0
    _served: dict[str, int] = field(default_factory=dict)
    #: Milliseconds before each block. The module default keeps the arms distinguishable at
    #: all; the break-even benchmark sets it to a real model's per-block streaming time, which
    #: is the window a guess can run ahead in.
    think_ms: float = THINK_MS

    def _turn_for(self, envelope: RequestEnvelope) -> ModelResponse:
        offered = {tool.name for tool in envelope.tools}
        for workload in WORKLOADS:
            names = {name for name, _args in workload.decision}
            if names and names <= offered:
                index = self._served.get(workload.name, 0)
                self._served[workload.name] = index + 1
                return workload.turns()[0]
        # A workload whose envelope declares no tools is driven through call_turn, where the
        # turn is the workload's own. Fall back to the first one whose name matches nothing
        # rather than inventing a call the run never asked for.
        for workload in WORKLOADS:
            if not envelope.tools:
                return workload.turns()[0]
        raise ModelError(
            f"no workload declares the tools this envelope offers ({sorted(offered)}); the "
            "scripted target refuses to invent a decision"
        )

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(self.think_ms / 1000.0)
        return self._turn_for(envelope)

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        response = self._turn_for(envelope)
        index = 0
        for block in response.content:
            await asyncio.sleep(self.think_ms / 1000.0)
            if getattr(block, "name", None) is not None:
                yield ToolUseComplete(index=index, block=block)  # type: ignore[arg-type]
                index += 1
        yield TurnComplete(response=response)
