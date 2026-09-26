"""SpecuNode -- agents that take real actions, and never take one twice.

A runtime for AI agents whose tools change the world. Every write is held until the model's
decision behind it is durable, claimed in a journal under a deterministic key before it is sent,
and never sent twice -- a crash is resumed onto the same keys, and a reply lost in a crash is
either asked about (``reconcile``) or handed to a human, never guessed at. Every run replays.

The public API is exposed lazily, so ``import specunode`` stays cheap and does not pull in the
optional integrations (LangGraph, MCP, Anthropic) until they are used::

    import specunode

    @specunode.tool(effect="write")
    async def charge_card(customer_id: str, amount: float) -> dict: ...

    runtime = specunode.Runtime(specunode.graph(nodes, route), tools=[charge_card])
    result = await runtime.run(inputs)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

#: Public name -> (module, attribute). Imported on first use.
_EXPORTS: dict[str, tuple[str, str]] = {
    "tool": ("specunode.integrations.plain", "tool"),
    "node": ("specunode.integrations.plain", "node"),
    "graph": ("specunode.api", "graph"),
    "Runtime": ("specunode.api", "Runtime"),
    "current_idempotency_key": ("specunode.api", "current_idempotency_key"),
    "agent_loop": ("specunode.core.loop", "agent_loop"),
    "RunSession": ("specunode.core.graph", "RunSession"),
    "Decision": ("specunode.core.decision", "Decision"),
    "FreeText": ("specunode.core.decision", "FreeText"),
    "ToolCall": ("specunode.core.decision", "ToolCall"),
    "EffectClass": ("specunode.core.effects", "EffectClass"),
    "Policy": ("specunode.core.policy", "Policy"),
    "RunResult": ("specunode.core.scheduler", "RunResult"),
    "ToolDispatchError": ("specunode.buffer.dispatcher", "ToolDispatchError"),
    "ModelError": ("specunode.core.model", "ModelError"),
    "AnthropicModel": ("specunode.integrations.anthropic", "AnthropicModel"),
}

__all__ = [
    "AnthropicModel",
    "Decision",
    "EffectClass",
    "FreeText",
    "ModelError",
    "Policy",
    "RunResult",
    "RunSession",
    "Runtime",
    "ToolCall",
    "ToolDispatchError",
    "__version__",
    "agent_loop",
    "current_idempotency_key",
    "graph",
    "node",
    "tool",
]

if TYPE_CHECKING:  # the same names, for type checkers and editors
    from specunode.api import Runtime, current_idempotency_key, graph
    from specunode.buffer.dispatcher import ToolDispatchError
    from specunode.core.decision import Decision, FreeText, ToolCall
    from specunode.core.effects import EffectClass
    from specunode.core.graph import RunSession
    from specunode.core.loop import agent_loop
    from specunode.core.model import ModelError
    from specunode.core.policy import Policy
    from specunode.core.scheduler import RunResult
    from specunode.integrations.anthropic import AnthropicModel
    from specunode.integrations.plain import node, tool


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'specunode' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
