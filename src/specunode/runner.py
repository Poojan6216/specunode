"""Turning a config into the objects a run needs.

``specunode resume`` and ``specunode replay`` both have to re-drive a graph, and a journal does
not contain one: it records what a graph *did*, not what it is. This module is the small amount
of wiring that turns ``specunode.yaml`` into a graph, a registry and a target model, so the two
commands do not each invent their own way of doing it.

It refuses rather than guesses. A config with no ``graph`` cannot be resumed or replayed, and
saying so is better than importing something plausible and re-driving the wrong program.
"""

from __future__ import annotations

import importlib
from typing import cast

from specunode.config import Config
from specunode.core.effects import ToolRegistry
from specunode.core.graph import GraphAdapter
from specunode.core.model import ModelClient

__all__ = ["RunnerError", "build_graph", "build_target", "load_reference"]


class RunnerError(RuntimeError):
    """The config does not describe something this command can run."""


def load_reference(reference: str) -> object:
    """Import ``"module:attribute"`` and return the attribute."""
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise RunnerError(f"{reference!r} is not a 'module:attribute' reference")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RunnerError(f"cannot import {module_name!r} from {reference!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise RunnerError(f"{module_name!r} has no attribute {attribute!r}") from exc


def build_graph(config: Config) -> tuple[GraphAdapter, ToolRegistry]:
    """Call the configured builder and check that it returned what it promised.

    The checks are here rather than left to the first confusing ``AttributeError`` several
    frames into the scheduler, because the person who sees this message is the one who wrote
    the reference and can fix it.
    """
    if config.graph is None:
        raise RunnerError(
            "this config has no 'graph:' entry, so there is nothing to re-drive. Add "
            "graph: \"your_module:build\" naming a callable that returns "
            "(graph_adapter, tool_registry)."
        )
    builder = load_reference(config.graph)
    if not callable(builder):
        raise RunnerError(f"{config.graph!r} is not callable")
    built = builder()
    if not isinstance(built, tuple) or len(built) != 2:
        raise RunnerError(
            f"{config.graph!r} returned {type(built).__name__}; it must return a "
            "(graph_adapter, tool_registry) pair"
        )
    adapter, registry = built
    if not isinstance(registry, ToolRegistry):
        raise RunnerError(
            f"{config.graph!r} returned {type(registry).__name__} where a ToolRegistry was "
            "expected. Effect classes are declared out of band and never inferred, so the "
            "registry is not optional."
        )
    for method in ("nodes", "run_node", "next", "capabilities"):
        if not hasattr(adapter, method):
            raise RunnerError(
                f"{config.graph!r} returned something without a {method!r} method; it does "
                "not satisfy the GraphAdapter protocol"
            )
    # Overrides last, so a config can reclassify a tool the graph declared -- which is the
    # supported way to say "this one actually writes" without editing someone else's code.
    for name, spec in config.tool_overrides().items():
        registry.register(spec)
        del name
    return cast(GraphAdapter, adapter), registry


def build_target(config: Config) -> ModelClient:
    """The model a resumed run asks for the turns its journal does not already hold.

    The model *name* is not set here: it travels on each ``RequestEnvelope``, because Hard
    Rule 13 makes the request the unit of identity and a client that silently substituted a
    different model would make the journal's record of what was asked untrue.
    """
    provider = config.target.provider
    if provider != "anthropic":
        raise RunnerError(
            f"target provider {provider!r} has no adapter in this release; only 'anthropic' "
            "is wired, and a resume needs a real model for the turns it has not journaled"
        )
    from specunode.integrations.anthropic import AnthropicModel

    return AnthropicModel()
