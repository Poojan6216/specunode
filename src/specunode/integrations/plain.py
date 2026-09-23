"""The plain-Python integration: decorators and a loop, over the same core.

A developer with a hand-written agent loop rather than a framework gets the store buffer by
declaring their tools and writing their nodes as functions. There is no second runtime behind
this: :class:`PlainAdapter` is a :class:`~specunode.core.graph.GraphAdapter` like any other,
and the same scheduler, journal, buffer, keys and ledger serve it. That is what makes spec task
2.4's check meaningful -- the same agent written both ways must produce the same ledger, and it
can only do that if there is one implementation to produce it.

The effect class is declared here, in code, by the developer who knows what the tool does
(Hard Rule 2). A tool decorated with no class is a ``WRITE``, and so is one nobody decorated at
all, because the cost of being wrong in that direction is a lost speculation while the cost of
being wrong in the other is an effect that escaped.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from specunode.canonical import JsonValue
from specunode.core.decision import Decision
from specunode.core.effects import (
    EffectClass,
    ToolRegistry,
    ToolSpec,
    forward_keys_from_template,
)
from specunode.core.graph import (
    END,
    AdapterCapabilities,
    NextNode,
    NodeRef,
    Parallel,
    RoutingIsInternal,
    RunSession,
)

__all__ = ["PlainAdapter", "node", "registry_of", "tool"]

NodeFn = Callable[[RunSession], Awaitable[Decision]]
F = TypeVar("F", bound=Callable[..., Awaitable[JsonValue]])

DecisionKind = Literal["tool_call", "route", "structured", "free_text", "unknown"]


def tool(
    *,
    effect: EffectClass | str = EffectClass.WRITE,
    idempotent: bool = False,
    compensator: str | None = None,
    forward_keys: str | None = None,
    witness: bool = False,
    name: str | None = None,
    registry: ToolRegistry | None = None,
) -> Callable[[F], F]:
    """Declare a function's effect class and register it.

    The default is ``WRITE``, not ``READ``, and not "infer it from the name". A tool whose
    upstream enqueues, schedules or triggers anything is a write whatever its response looks
    like: ``{"status": "queued"}`` is not a read result.
    """
    resolved = EffectClass(effect) if isinstance(effect, str) else effect

    def decorate(fn: F) -> F:
        spec = ToolSpec(
            name=name or fn.__name__,
            effect=resolved,
            fn=fn,
            idempotent=idempotent,
            compensator=compensator,
            forward_keys=forward_keys_from_template(forward_keys) if forward_keys else None,
            witness=witness,
        )
        if registry is not None:
            registry.register(spec)
        fn.__specunode_tool__ = spec  # type: ignore[attr-defined]
        return fn

    return decorate


def node(
    *,
    name: str | None = None,
    speculable: bool = False,
    emits: DecisionKind = "tool_call",
) -> Callable[[NodeFn], NodeFn]:
    """Mark a function as a graph node.

    ``speculable`` is off by default and is the developer saying "this body is safe to run and
    throw away". A node body is unbounded code -- it can touch the filesystem, a socket or a
    global, none of which the runtime can see -- so speculation does not run one until it has
    been opted in, and a predicted route into a node that has not been is named as a hazard
    rather than silently skipped.
    """

    def decorate(fn: NodeFn) -> NodeFn:
        fn.__specunode_node__ = {  # type: ignore[attr-defined]
            "name": name or fn.__name__,
            "speculable": speculable,
            "emits": emits,
        }
        return fn

    return decorate


@dataclass
class PlainAdapter:
    """A graph of decorated functions plus a router. The scheduler drives it."""

    node_fns: Mapping[str, NodeFn]
    #: The next node's name; several names to run those nodes side by side; ``None`` to stop.
    route: Callable[[Mapping[str, JsonValue]], str | Sequence[str] | None]
    _order: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._order = tuple(self.node_fns)

    @classmethod
    def of(
        cls,
        fns: Iterable[NodeFn],
        route: Callable[[Mapping[str, JsonValue]], str | Sequence[str] | None],
    ) -> PlainAdapter:
        return cls({_name_of(fn): fn for fn in fns}, route)

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            drives_itself=False,
            speculable_nodes=frozenset(
                name
                for name, fn in self.node_fns.items()
                if getattr(fn, "__specunode_node__", {}).get("speculable", False)
            ),
            supports_interrupt=False,
            supports_streaming=False,
            framework="plain",
        )

    def nodes(self) -> Sequence[NodeRef]:
        return [NodeRef(name=name) for name in self._order]

    def decision_kind(self, node: NodeRef) -> DecisionKind:
        fn = self.node_fns.get(node.name)
        if fn is None:
            return "unknown"
        kind: DecisionKind = getattr(fn, "__specunode_node__", {}).get("emits", "unknown")
        return kind

    def next(self, state: Mapping[str, JsonValue]) -> NextNode:
        chosen = self.route(state)
        if chosen is None:
            return END
        # A router may name several nodes at once: they are independent and run side by side.
        names = _named(chosen)
        for name in names:
            if name not in self.node_fns:
                raise KeyError(f"router chose {name!r}, which is not a node in this graph")
        if len(names) == 1:
            return NodeRef(name=names[0])
        return Parallel(nodes=tuple(NodeRef(name=name) for name in names))

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        fn = self.node_fns.get(node.name)
        if fn is None:
            raise KeyError(f"no node named {node.name!r}")
        return await fn(session)

    async def drive(self, session: RunSession, inputs: JsonValue) -> JsonValue:
        raise RoutingIsInternal(
            "PlainAdapter is driven by the scheduler; drive() is for adapters that own their "
            "own run loop"
        )


def _named(chosen: object) -> list[str]:
    """The node names a router returned, in the order the group retires them.

    A list or a tuple, and nothing else. The order is load-bearing: it fixes each lane's node
    id, so its idempotency keys, and the order its effects reach the world. A set iterates in
    an order string hashing decides afresh in every process, so a resume or a replay would mint
    the lanes under different ids -- and send again what had already been sent.
    """
    if isinstance(chosen, str):
        return [chosen]
    if not isinstance(chosen, list | tuple):
        raise TypeError(
            f"a router returns a node name, a list or tuple of names in the order they should "
            f"retire, or None to end the run; got {type(chosen).__name__}"
        )
    names = [str(name) for name in chosen]
    if not names:
        raise ValueError("the router named no nodes; return None to end the run")
    if len(set(names)) != len(names):
        raise ValueError(f"the router named a node twice in one group: {names}")
    return names


def _name_of(fn: NodeFn) -> str:
    meta = getattr(fn, "__specunode_node__", None)
    if meta is None:
        raise TypeError(f"{fn!r} is not decorated with @specunode.node")
    return str(meta["name"])


def registry_of(fns: Iterable[Callable[..., Awaitable[JsonValue]]]) -> ToolRegistry:
    """Build a registry from decorated tool functions."""
    registry = ToolRegistry()
    for fn in fns:
        spec = getattr(fn, "__specunode_tool__", None)
        if spec is None:
            raise TypeError(f"{fn!r} is not decorated with @specunode.tool")
        registry.register(spec)
    return registry
