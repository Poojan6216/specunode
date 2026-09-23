"""A model/tool loop inside one node, shaped so the model can ask for several calls at once.

When the model, not the tools, is what a run waits on, the lever is the number of model turns.
A reply that asks for four independent reads at once replaces four round trips with one, and
this runtime is built for exactly that reply: it issues each read the moment its block parses,
runs them concurrently, and holds the reply's writes until the turn that asked for them is
durable. What it needs from the loop around it is not to undo that.

Two details decide whether a loop does, and both are easy to get wrong:

**Every result goes back in one message.** The Messages API accepts tool results split across
several user messages, and a model shown its results that way learns to stop asking for more
than one call at a time. So the whole turn's results travel together, each paired with the
``tool_use`` id that asked for it, in the order the model asked.

**The reply goes back unchanged.** The assistant message carries the model's thinking blocks
and their signatures; the next request has to include them exactly as they came, or a model
that reasons across tool calls loses the thread.

The loop runs inside a node, so every turn is journaled, every write is staged and released
only by the turn that asked for it, and a resume or replay sees the same conversation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from specunode.core.graph import RunSession
from specunode.core.model import (
    Message,
    ModelResponse,
    RequestEnvelope,
    ToolResultBlock,
    TurnResults,
)

__all__ = ["LoopResult", "agent_loop"]

#: Why the loop stopped. ``end_turn`` is the model's own decision that it was finished.
StopReason = Literal["end_turn", "max_turns"]


@dataclass(frozen=True)
class LoopResult:
    """How a loop went: the number that matters here is ``turns``."""

    turns: int
    calls: int
    final: ModelResponse | None
    messages: tuple[Message, ...]
    stopped: StopReason

    @property
    def calls_per_turn(self) -> float:
        """Tool calls per turn that asked for any. One means nothing ran side by side."""
        tool_turns = self.turns - (1 if self.stopped == "end_turn" else 0)
        return self.calls / tool_turns if tool_turns else 0.0


async def agent_loop(
    session: RunSession, envelope: RequestEnvelope, *, max_turns: int = 20
) -> LoopResult:
    """Drive the model until it stops asking for tools, or ``max_turns`` is reached.

    ``envelope`` is the first request; each later one is the same request with the
    conversation so far as its messages, so the system prompt and tool definitions -- the
    stable prefix a prompt cache keys on -- never move.
    """
    if session.call_turn is None:
        raise RuntimeError("agent_loop runs inside a node the runtime drives (call_turn)")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    messages: list[Message] = list(envelope.messages)
    calls = 0
    final: ModelResponse | None = None
    for turn in range(1, max_turns + 1):
        results = await session.call_turn(replace(envelope, messages=tuple(messages)))
        response = results.response if isinstance(results, TurnResults) else None
        if response is None:
            raise RuntimeError("call_turn returned no reply to continue the conversation from")
        final = response
        uses = response.tool_uses
        if not uses:
            return LoopResult(turn, calls, final, tuple(messages), "end_turn")
        if len(uses) != len(results):
            raise RuntimeError(
                f"the reply asked for {len(uses)} call(s) and {len(results)} result(s) came back"
            )
        calls += len(uses)
        messages.append(Message(role="assistant", content=response.content))
        messages.append(Message(role="user", content=_results_for(uses, results)))
    return LoopResult(max_turns, calls, final, tuple(messages), "max_turns")


def _results_for(uses: Sequence[object], results: Sequence[object]) -> tuple[ToolResultBlock, ...]:
    """One block per call, in the order the model asked, each naming the call it answers."""
    return tuple(
        ToolResultBlock(tool_use_id=str(getattr(use, "id", "")), content=result)  # type: ignore[arg-type]
        for use, result in zip(uses, results, strict=True)
    )
