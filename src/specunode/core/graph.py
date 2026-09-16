"""The framework boundary: what the scheduler needs from a graph, and nothing more.

Two adapter shapes share one core, which is what makes the LangGraph integration and the
plain-Python API produce the same ledger from the same decisions.

``drives_itself = False``
    The scheduler drives: it asks ``next()`` for the node and ``run_node()`` for the decision.
    This is the plain-Python loop, and it is section 7's protocol verbatim.

``drives_itself = True``
    The framework drives, and the scheduler waits inside ``drive()``. This is LangGraph.
    Driving a compiled graph node by node was tried and rejected: calling a node's bound
    runnable directly raises ``RuntimeError: Called get_config outside of a runnable context``
    the moment the body calls ``get_config()``, ``interrupt()`` or ``get_stream_writer()``, and
    re-deriving routing, reducers and map-reduce outside Pregel would break the promise that a
    wrapped graph produces the same final state as an unwrapped one.

Both shapes reach the runtime through the same ports, so journal, buffer, keys and ledger have
exactly one implementation.

**A speculative branch does not run user node bodies by default.** A node body is unbounded
code: it can touch the filesystem, a socket or a global, none of which the fake world can see,
so the leak test could not catch a side effect from a squashed branch that did not go through a
registered tool. Speculation therefore executes *predicted tool calls* and, in read-only
stretches, the next model turn. A node that opts in with ``speculable=True`` is the developer
saying its body is safe to run and discard; a predicted route into one that has not opted in
stalls with :data:`~specunode.core.hazards.Hazard.NODE_NOT_SPECULABLE`, which is named rather
than silent so the benchmark's hazard histogram stays complete.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from specunode.canonical import JsonValue
from specunode.core.decision import Decision
from specunode.core.effects import ToolSpec
from specunode.core.model import ModelClient

__all__ = [
    "END",
    "AdapterCapabilities",
    "GraphAdapter",
    "NodeRef",
    "RoutingIsInternal",
    "RunSession",
    "active_session",
    "current_session",
    "routed",
    "session_scope",
]


class RoutingIsInternal(RuntimeError):
    """Raised by a self-driving adapter asked to expose its routing.

    LangGraph decides its own next node inside Pregel. Reimplementing that outside would mean
    re-deriving conditional edges, reducers and map-reduce, and the first thing to break would
    be the property task 2.3 checks: that a wrapped graph reaches the same final state.
    """


@dataclass(frozen=True, slots=True)
class NodeRef:
    """A node, identified structurally so that a replay reproduces its id exactly.

    Never by object identity, memory address, line number or a framework's own task id --
    LangGraph's task ids are stable within a run and random across runs, so a key derived from
    one would not survive a replay.
    """

    name: str
    path: tuple[str, ...] = ()

    @property
    def structural_id(self) -> str:
        return "/".join((*self.path, self.name))

    def under(self, parent: NodeRef) -> NodeRef:
        """This node as a child of a sub-graph, for nested branch trees."""
        return NodeRef(name=self.name, path=(*parent.path, parent.name))


class _End:
    """The terminal marker returned by ``next()``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "END"


END = _End()

NextNode: TypeAlias = NodeRef | _End


@dataclass(frozen=True)
class AdapterCapabilities:
    """What this adapter can and cannot do, so the scheduler never has to guess."""

    #: True when the framework owns the run loop and the scheduler waits inside ``drive()``.
    drives_itself: bool = False
    #: Nodes whose bodies may run on a speculative branch. Empty is the safe default.
    speculable_nodes: frozenset[str] = frozenset()
    supports_interrupt: bool = False
    supports_streaming: bool = False
    #: Reported so ``docs/adapters.md`` and the ledger can name the framework and its version.
    framework: str = "plain"
    version: str | None = None


@dataclass
class RunSession:
    """Everything a node body reaches the runtime through.

    Deliberately a small, explicit surface: the ports are the only way a node can touch the
    world or the model, so anything a node does *outside* them is invisible to the leak test --
    which is why a speculative branch does not run node bodies unless the developer opted in.
    """

    run_id: str
    #: Called by an adapter to run one tool call, either executing or staging it.
    call_tool: Callable[[str, Mapping[str, JsonValue]], Awaitable[JsonValue]]
    #: Called by an adapter when a node reaches a decision point.
    decide: Callable[[Decision], Awaitable[Decision]]
    #: How a self-driving framework runs one node under the runtime. The shim hands over the
    #: node's name and a thunk for its body; the runtime mints a branch, runs the body as a
    #: task, retires it, and returns whatever the body returned. It is a thunk rather than a
    #: context manager because a node parked on a staged write's result must retire *while its
    #: body is still suspended* -- wrapping the body in `async with` would only reach the exit
    #: after the body finished, which is the deadlock the store buffer design exists to avoid.
    run_in_node: (
        Callable[[str, Callable[[], Awaitable[JsonValue]]], Awaitable[JsonValue]] | None
    ) = None
    #: The target model, already wrapped so every request and response is journaled before
    #: the runtime acts on it. A node calls this rather than constructing its own client --
    #: that substitution is what lets a developer's graph file stay unchanged.
    model: ModelClient | None = None
    #: The branch's working state. A MutableMapping rather than a dict so the runtime can hand
    #: over a copy-on-write fork that measures its own delta, while a node body still just
    #: reads and writes keys.
    state: MutableMapping[str, JsonValue] = field(default_factory=dict)


#: The session a node body is running inside. A ContextVar because a framework's node
#: signature belongs to the framework, not to us: a LangGraph node is handed state and a
#: config, with nowhere to pass a runtime handle. Each asyncio task gets its own copy, so one
#: branch cannot reach another's session (Hard Rule 6).
active_session: ContextVar[RunSession | None] = ContextVar("specunode_active_session", default=None)


def current_session() -> RunSession | None:
    """The session this node body is running inside, or ``None`` outside a run."""
    return active_session.get()


@contextmanager
def session_scope(session: RunSession | None) -> Iterator[None]:
    token = active_session.set(session)
    try:
        yield
    finally:
        active_session.reset(token)


def routed(spec: ToolSpec) -> Callable[..., Awaitable[JsonValue]]:
    """Wrap a tool so it goes through the runtime when there is one, and runs plainly when not.

    This is what lets one graph file run both wrapped and unwrapped, which spec task 2.3 asks
    for and task 2.4 compares. Outside a run the call is the developer's own function; inside
    one it is classified, journaled, and either executed (a READ) or staged (anything else).

    A tool the developer did not route is invisible to the runtime -- which is also why an
    undeclared tool is a WRITE rather than an error: the runtime cannot make a call it never
    sees safe, and it says so rather than pretending.
    """

    async def call(**kwargs: JsonValue) -> JsonValue:
        session = active_session.get()
        if session is None:
            return await spec.fn(**kwargs)
        return await session.call_tool(spec.name, kwargs)

    call.__name__ = spec.name
    call.__doc__ = spec.fn.__doc__
    return call


@runtime_checkable
class GraphAdapter(Protocol):
    """What the scheduler needs from a graph."""

    def capabilities(self) -> AdapterCapabilities: ...

    def nodes(self) -> Sequence[NodeRef]: ...

    def decision_kind(
        self, node: NodeRef
    ) -> Literal["tool_call", "route", "structured", "free_text", "unknown"]:
        """What this node emits, so a free-text barrier is known before anything is spent."""
        ...

    def next(self, state: Mapping[str, JsonValue]) -> NextNode:
        """The next node. Raises :class:`RoutingIsInternal` on a self-driving adapter."""
        ...

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        """Run one node to its decision point. Scheduler-driven adapters only."""
        ...

    async def drive(self, session: RunSession, inputs: JsonValue) -> JsonValue:
        """Run the whole graph, calling back into the session. Self-driving adapters only."""
        ...
