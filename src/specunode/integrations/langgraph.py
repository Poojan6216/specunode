"""LangGraph integration: the developer's graph file does not change.

LangGraph owns its own run loop. Routing, reducers, ``Send``/map-reduce and the whole Pregel
scaffolding live inside ``ainvoke``, and driving a compiled graph node by node was tried and
rejected -- calling a node's bound runnable directly raises ``RuntimeError: Called get_config
outside of a runnable context`` the moment the body calls ``get_config()``, ``interrupt()`` or
``get_stream_writer()``, and re-deriving the rest outside Pregel would break the property spec
task 2.3 actually checks: that a wrapped graph reaches the same final state as an unwrapped one.

So this integration does not drive. It **substitutes**: each compiled node's bound runnable is
replaced by a shim that runs the original body inside a branch of its own. LangGraph still
decides what runs next and still reduces state; the runtime sees each node's tool calls and
model calls and retires a branch per node.

Two things a developer must do, and neither is a change to the graph's shape:

* declare each tool's effect class, and call it through :func:`~specunode.core.graph.routed`.
  A tool the runtime never sees cannot be staged, and Hard Rule 2 means an undeclared one is a
  WRITE rather than a guess.
* let the runtime supply the model, so requests and responses are journaled before use.
  ``wrap()`` substitutes it; a node that constructs its own client bypasses the journal, and
  ``docs/limitations.md`` says so.

Version pin: probed against langgraph 1.2.11. ``node.bound`` is a documented-enough internal
that Decision Gate D4 permits using it, and :func:`probe` fails loudly rather than silently
degrading if a future version moves it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import (
    AdapterCapabilities,
    NextNode,
    NodeRef,
    RoutingIsInternal,
    RunSession,
    current_session,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["LangGraphAdapter", "LangGraphUnsupported", "probe", "wrap"]

#: The version this integration was probed against. A newer one is allowed and warned about;
#: a layout change fails loudly, because the alternative is a runtime that silently sees none
#: of the graph's effects while still producing a confident-looking ledger.
PROBED_VERSION = "1.2.11"


class LangGraphUnsupported(RuntimeError):
    """The installed LangGraph does not expose what this integration substitutes."""


@dataclass(frozen=True)
class Probe:
    version: str
    nodes_attr: bool
    bound_attr: bool

    @property
    def usable(self) -> bool:
        return self.nodes_attr and self.bound_attr


def probe(compiled: object) -> Probe:
    """Check that this compiled graph exposes the seam the integration needs."""
    try:
        import importlib.metadata as metadata

        version = metadata.version("langgraph")
    except Exception:  # pragma: no cover - langgraph not installed
        version = "unknown"

    nodes = getattr(compiled, "nodes", None)
    nodes_attr = isinstance(nodes, Mapping)
    bound_attr = False
    if nodes_attr and nodes:
        bound_attr = all(hasattr(node, "bound") for node in nodes.values())
    return Probe(version=version, nodes_attr=nodes_attr, bound_attr=bound_attr)


def _node_names(compiled: object) -> list[str]:
    nodes = getattr(compiled, "nodes", {})
    if not isinstance(nodes, Mapping):
        return []
    # LangGraph's own entry marker is not a node the developer wrote and has no decision.
    return [str(name) for name in nodes if not str(name).startswith("__")]


@dataclass
class LangGraphAdapter:
    """A compiled LangGraph, with each node's body substituted."""

    compiled: Any
    node_names: tuple[str, ...] = ()
    speculable: frozenset[str] = frozenset()
    #: Every node the runtime substituted, by its qualified path. A sub-graph's nodes appear
    #: as ``sub/inner``; the container itself does not appear, because it does no work.
    installed_nodes: list[str] = field(default_factory=list)
    subgraphs: int = 0
    _installed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        result = probe(self.compiled)
        if not result.usable:
            raise LangGraphUnsupported(
                f"langgraph {result.version} does not expose compiled.nodes[*].bound, which is "
                f"the seam this integration substitutes (probed against {PROBED_VERSION}). "
                "Pin langgraph or open an issue; silently running unwrapped would produce a "
                "ledger for a run the runtime never saw."
            )
        if not self.node_names:
            self.node_names = tuple(_node_names(self.compiled))
        self._install()

    def _install(self) -> None:
        """Replace each node's bound runnable with one that runs it under a branch.

        A node whose bound runnable is itself a compiled graph is a *sub-graph*, and its
        container is not substituted: the inner nodes are, under the parent's path. That gives
        a nested branch tree whose ids read ``sub/inner_a`` rather than one opaque branch for
        the whole sub-graph, which is what makes the leak invariant meaningful inside one.
        """
        if self._installed:
            return
        self._substitute(self.compiled, path=())
        self._installed = True

    def _substitute(self, compiled: Any, path: tuple[str, ...]) -> None:
        from langchain_core.runnables import RunnableLambda

        for name in _node_names(compiled):
            node = compiled.nodes[name]
            original = node.bound
            if hasattr(original, "nodes") and _node_names(original):
                # A sub-graph. Recurse rather than wrapping the container, so each inner node
                # gets its own branch and its own place in the ledger.
                self._substitute(original, path=(*path, name))
                self.subgraphs += 1
                continue
            qualified = "/".join((*path, name))
            node.bound = RunnableLambda(_make_shim(qualified, original))
            self.installed_nodes.append(qualified)

    # -- the GraphAdapter surface ---------------------------------------------------------

    def capabilities(self) -> AdapterCapabilities:
        version = probe(self.compiled).version
        return AdapterCapabilities(
            drives_itself=True,
            speculable_nodes=self.speculable,
            supports_interrupt=True,
            supports_streaming=True,
            framework="langgraph",
            version=version,
        )

    def nodes(self) -> Sequence[NodeRef]:
        refs: list[NodeRef] = []
        for qualified in self.installed_nodes or list(self.node_names):
            *path, name = qualified.split("/")
            refs.append(NodeRef(name=name, path=tuple(path)))
        return refs

    def decision_kind(
        self, node: NodeRef
    ) -> Literal["tool_call", "route", "structured", "free_text", "unknown"]:
        # The runtime cannot see inside an arbitrary node body, and guessing here would put a
        # speculation barrier in the wrong place in both directions.
        return "unknown"

    def next(self, state: Mapping[str, JsonValue]) -> NextNode:
        raise RoutingIsInternal(
            "LangGraph decides its own next node inside Pregel; this adapter is driven through "
            "drive() rather than by the scheduler's own loop"
        )

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        raise RoutingIsInternal("a self-driving adapter runs its own nodes")

    async def drive(self, session: RunSession, inputs: JsonValue) -> JsonValue:
        result = await self.compiled.ainvoke(inputs)
        return result if isinstance(result, (Mapping, list, str, int, float, bool)) else None


def _make_shim(name: str, original: Any) -> Callable[..., Awaitable[JsonValue]]:
    """A node body that runs inside a branch when there is a runtime, and plainly when not."""

    async def shim(state: Any, config: Any = None) -> Any:
        session = current_session()
        if session is None or session.run_in_node is None:
            # Unwrapped: this is the developer's own graph, behaving exactly as before.
            return await original.ainvoke(state, config)

        async def body() -> Any:
            return await original.ainvoke(state, config)

        return await session.run_in_node(name, body)

    shim.__name__ = f"specunode_shim_{name}"
    return shim


@dataclass
class SpecuNodeGraph:
    """A wrapped graph. ``ainvoke`` runs it under the runtime and produces a ledger."""

    adapter: LangGraphAdapter
    scheduler: Any
    default_run_id: str | None = None

    def _run_id(self, run_id: str | None) -> str:
        from specunode.ids import new_ulid

        return run_id or self.default_run_id or new_ulid()

    async def ainvoke(self, inputs: JsonValue, run_id: str | None = None) -> Any:
        result = await self.scheduler.run(self._run_id(run_id), inputs)
        if not result.ok and result.error:
            raise RuntimeError(result.error)
        return result.state

    async def astream(self, inputs: JsonValue, run_id: str | None = None) -> Any:
        """Yield the canonical path's output, and nothing a speculation produced.

        A speculative branch's model output may be squashed, so streaming it to a user would
        show them text that the run then decided against. There is no way to take it back once
        it is on their screen, which is why this yields after each node retires rather than as
        each branch produces something.
        """
        resolved = self._run_id(run_id)
        result = await self.scheduler.run(resolved, inputs)
        if not result.ok and result.error:
            raise RuntimeError(result.error)
        for row in result.ledger.rows:
            yield {"effect": row.call.name, "args": dict(row.call.args), "status": row.status}
        yield {"state": result.state}

    async def run(self, inputs: JsonValue, run_id: str | None = None) -> Any:
        """Like :meth:`ainvoke` but returns the whole result, ledger included."""
        return await self.scheduler.run(self._run_id(run_id), inputs)


def wrap(
    compiled: Any,
    *,
    registry: Any,
    journal: Any,
    target: Any,
    run_id: str | None = None,
    policy: Any = None,
    dispatcher: Any = None,
    speculable: frozenset[str] = frozenset(),
) -> SpecuNodeGraph:
    """Wrap a compiled ``StateGraph`` so it runs under SpecuNode.

    The graph definition is not rewritten: its nodes keep their bodies, its edges keep their
    routing, and LangGraph keeps its reducers. What changes is that each node runs inside a
    branch, its routed tool calls are classified and staged, and its model calls are journaled.
    """
    from specunode.buffer.dispatcher import Dispatcher
    from specunode.buffer.store_buffer import StoreBuffer
    from specunode.core.policy import Policy
    from specunode.core.scheduler import Scheduler

    adapter = LangGraphAdapter(compiled=compiled, speculable=speculable)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        # The run id is set by Scheduler.run, which is the only place that knows it. Passing
        # one here as well is how the buffer and the ledger end up reading different runs.
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=dispatcher or Dispatcher(registry=registry),
        target=target,
        policy=policy or Policy(speculation=False),
    )
    return SpecuNodeGraph(adapter=adapter, scheduler=scheduler, default_run_id=run_id)


def as_tool_call(name: str, args: Mapping[str, JsonValue]) -> ToolCall:
    return ToolCall(name=name, args=dict(args))
