"""An MCP proxy, so a developer who cannot change their agent still gets the store buffer.

The proxy sits between an MCP client and an upstream MCP server. It forwards everything it does
not care about, classifies each `tools/call` by effect class, runs the reads, and holds the
writes.

**The hard part is that the proxy cannot see the model.** It sees tool calls, not the decision
that produced them, so it has no way to know whether a call it is holding was confirmed. The
branch-resolution signal therefore has to arrive out of band: the client sends
`notifications/specunode/decision`, or a second client call invokes the `specunode.retire` tool.

That leads to the one genuinely awkward choice in this integration, and it is resolved by
asking the client rather than by picking a default:

* A client that **advertises the decision capability** gets the real thing. Writes are staged,
  the call returns a placeholder handle, and the buffer drains when a decision arrives.
* A client that **does not** would be handed a placeholder it does not understand, and would
  put it straight into the next prompt -- which is exactly what Hard Rule 13 forbids, and the
  proxy cannot see the prompt to stop it. So for that client a write is **blocked** until a
  decision arrives, the call blocks rather than returning a handle, and the run's ledger is
  stamped ``context_identity: unenforced`` because the proxy genuinely cannot enforce it.

Neither mode is pretended to be the full runtime. The stamp says which one ran.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from specunode.canonical import JsonValue, chash
from specunode.core.decision import Decision, ToolCall, decisions_equal
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec

__all__ = [
    "DECISION_NOTIFICATION",
    "PROXY_TOOLS",
    "ClientMode",
    "ProxyState",
    "ProxyUnsupported",
    "StagedCall",
    "probe_sdk",
    "serve",
]

#: The notification a client sends to tell the proxy what the model actually decided.
DECISION_NOTIFICATION = "notifications/specunode/decision"

#: The capability a client advertises to say it understands a staged-write handle.
DECISION_CAPABILITY = "specunode/decisions"


class ClientMode(Enum):
    """What this client can be trusted with."""

    #: Advertises the decision capability: understands a handle and will report decisions.
    HANDLES = "handles"
    #: Does not. A write blocks until a decision arrives, because handing this client a
    #: placeholder would put one in the next prompt where nothing can see it.
    BLOCKING = "blocking"


class _Absent:
    """A private sentinel for "the client did not send this argument".

    Not ``None``, because ``None`` is a value a client can legitimately send and the two must
    stay distinguishable: forwarding an invented null broke every upstream tool with a
    non-nullable optional parameter, and silently changed the argument hash of every staged
    write that had one.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT = _Absent()


#: How long a blocking client's write waits for a decision before giving up, in seconds.
#:
#: Generous, because the decision legitimately comes from outside the proxy and a model turn
#: takes as long as it takes. Finite, because the alternative is what shipped: an unbounded wait
#: in the default mode, which reads to an operator as a hung server rather than as a runtime
#: doing exactly what it promised.
DEFAULT_DECISION_DEADLINE_S = 300.0


@dataclass(frozen=True, slots=True)
class StagedCall:
    """A write the proxy is holding."""

    effect_id: str
    tool: str
    args: Mapping[str, JsonValue]
    effect: EffectClass
    stage_index: int
    handle: str

    @property
    def decision(self) -> ToolCall:
        return ToolCall(name=self.tool, args=self.args)


@dataclass
class ProxyState:
    """What the proxy is holding, and for whom.

    Deliberately a plain object with no I/O: the protocol layer calls into it, so the staging
    rules can be tested without a transport, and a protocol change cannot quietly alter them.
    """

    registry: ToolRegistry
    mode: ClientMode = ClientMode.BLOCKING
    #: How long a blocking write waits for a decision. Per-state so an operator can tune it;
    #: see :data:`DEFAULT_DECISION_DEADLINE_S` for why it is finite at all.
    decision_deadline_s: float = DEFAULT_DECISION_DEADLINE_S
    staged: list[StagedCall] = field(default_factory=list)
    dispatched: list[StagedCall] = field(default_factory=list)
    discarded: list[StagedCall] = field(default_factory=list)
    reads_forwarded: int = 0
    #: Upstream results for calls that retired, by effect id. A blocking caller reads its own
    #: result from here rather than sending the call itself, so exactly one party sends.
    results: dict[str, JsonValue] = field(default_factory=dict)
    #: Monotonic across the whole session, and never reset by :meth:`retire`.
    #:
    #: Effect ids used to be numbered from ``len(self.staged)``, which ``retire`` sets back to
    #: an empty list -- so the first write of turn 2 got the same id as the first write of
    #: turn 1. ``StagedCall`` is a frozen dataclass, so two structurally identical calls from
    #: different turns then compared *equal*, and a membership test against the dispatched list
    #: said yes for a call this turn's decision had just discarded. The proxy forwarded a write
    #: it had explicitly refused, which is the one thing this whole project exists to prevent.
    _seq: int = 0
    #: One event per staged call, created when it is staged and set only when *that* call has
    #: been resolved and its result recorded.
    #:
    #: A single shared event was wrong twice over. ``publish()`` is reachable from every
    #: ``specunode.retire`` and from ``discard_all``, so while one retire was awaiting its
    #: upstream round trip any other call to either could release a blocked caller whose result
    #: had not been recorded yet -- handing the client an empty answer for a write that really
    #: did reach the world. And a write staged *during* that window saw the event already set,
    #: found itself not in ``dispatched``, and was told it had been discarded -- while it was
    #: still held, still listed by ``specunode.status``, and still due to be forwarded by the
    #: next matching decision.
    _events: dict[str, asyncio.Event] = field(default_factory=dict)
    #: What happened to each resolved call, by effect id.
    _outcome: dict[str, str] = field(default_factory=dict)

    @property
    def context_identity(self) -> str:
        """The proxy never sees a prompt, so it can never claim to have checked one."""
        return "unenforced"

    def classify(self, tool: str) -> ToolSpec:
        return self.registry.get(tool)

    def stage(self, tool: str, args: Mapping[str, JsonValue]) -> StagedCall:
        """Hold a write. Nothing is sent upstream."""
        spec = self.classify(tool)
        self._seq += 1
        effect_id = f"mcp-{self._seq:04d}-{chash(dict(args))[:8]}"
        call = StagedCall(
            effect_id=effect_id,
            tool=tool,
            args=dict(args),
            effect=spec.effect,
            stage_index=len(self.staged),
            handle=f"$specunode.handle:{effect_id}",
        )
        self.staged.append(call)
        self._events[call.effect_id] = asyncio.Event()
        return call

    def result_for(self, call: StagedCall) -> Mapping[str, JsonValue]:
        """What a handle-capable client is handed instead of a value."""
        return {
            "_specunode": {
                "staged": True,
                "effect_id": call.effect_id,
                "handle": call.handle,
                "note": (
                    "This write is held in a store buffer and has not happened. It will be "
                    "sent when you report the model's decision, and discarded if you report a "
                    "different one. Do not put this handle in a prompt."
                ),
            }
        }

    def retire(self, actual: Decision) -> tuple[list[StagedCall], list[StagedCall]]:
        """Resolve everything held against the decision the model actually made.

        Exact canonical equality, the same relation the in-process gate uses. A call that does
        not match is discarded unsent -- the proxy is a different transport for the same rule,
        not a weaker version of it.
        """
        confirmed: list[StagedCall] = []
        dropped: list[StagedCall] = []
        for call in self.staged:
            # One decision authorises **one** call. Every structurally identical held write
            # used to match, so a single model decision forwarded all of them -- and the proxy
            # computes no idempotency key and keeps no dedupe table, so nothing downstream
            # could absorb the repeat. A later decision can still confirm the next one.
            if not confirmed and decisions_equal(call.decision, actual):
                confirmed.append(call)
            else:
                dropped.append(call)
        self.staged = []
        self.dispatched.extend(confirmed)
        self.discarded.extend(dropped)
        for call in confirmed:
            self._outcome[call.effect_id] = "dispatched"
        for call in dropped:
            self._outcome[call.effect_id] = "discarded"
        # Deliberately no event is set here. The caller forwards each confirmed call and
        # records its result *after* this returns, and releasing the blocked client first meant
        # it read ``results`` before anything was in it. Publishing is a separate step.
        return confirmed, dropped

    def publish(self, calls: Sequence[StagedCall]) -> None:
        """Release the callers waiting on *these* calls, now their results are recorded.

        Scoped to the calls a decision actually resolved. A blanket release reached callers
        whose own write was untouched by this decision, and callers whose result had not been
        written yet.
        """
        for call in calls:
            event = self._events.get(call.effect_id)
            if event is not None:
                event.set()

    def outcome_of(self, call: StagedCall) -> str | None:
        """``"dispatched"``, ``"discarded"``, or ``None`` while the call is still held."""
        return self._outcome.get(call.effect_id)

    def was_confirmed(self, call: StagedCall) -> bool:
        """Did *this* call retire? Compared by effect id, never by value.

        A value comparison answers for any structurally identical call from any earlier turn,
        which is how a discarded write got forwarded.
        """
        return any(sent.effect_id == call.effect_id for sent in self.dispatched)

    def record_result(self, call: StagedCall, result: JsonValue) -> None:
        """Keep what the upstream returned, so the caller that is waiting can have it."""
        self.results[call.effect_id] = result

    def result_of(self, call: StagedCall) -> JsonValue:
        return self.results.get(call.effect_id)

    def discard_all(self, reason: str = "squashed") -> int:
        dropped = list(self.staged)
        count = len(dropped)
        self.discarded.extend(dropped)
        self.staged = []
        for call in dropped:
            self._outcome[call.effect_id] = "discarded"
        # Safe to release immediately, and only these: nothing was confirmed, so there is no
        # result to wait for.
        self.publish(dropped)
        return count

    async def wait_for_call(self, call: StagedCall, deadline_s: float | None = None) -> bool:
        """Wait until *this* call has been resolved and its result recorded."""
        event = self._events.get(call.effect_id)
        if event is None:  # pragma: no cover - a call that was never staged here
            return False
        try:
            await asyncio.wait_for(
                event.wait(),
                self.decision_deadline_s if deadline_s is None else deadline_s,
            )
        except TimeoutError:
            return False
        return True

    async def wait_for_decision(self, deadline_s: float | None = None) -> bool:
        """Wait until *any* call is resolved. Kept for callers that hold no particular one.

        Prefer :meth:`wait_for_call`. This answers a question with no owner -- "has something
        been decided?" -- and a blocking client needs "has *mine*?". Using it for that is how a
        caller came to be released by a decision about somebody else's write.
        """
        if not self._events:
            return False
        waiters = [
            asyncio.ensure_future(event.wait())
            for effect_id, event in list(self._events.items())
            if not self._outcome.get(effect_id)
        ]
        if not waiters:
            return True
        try:
            done, _pending = await asyncio.wait(
                waiters,
                timeout=self.decision_deadline_s if deadline_s is None else deadline_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for waiter in waiters:
                waiter.cancel()
        return bool(done)

    def status(self) -> Mapping[str, JsonValue]:
        return {
            "mode": self.mode.value,
            "context_identity": self.context_identity,
            "staged": [
                {"effect_id": c.effect_id, "tool": c.tool, "args": dict(c.args)}
                for c in self.staged
            ],
            "dispatched": len(self.dispatched),
            "discarded": len(self.discarded),
            "reads_forwarded": self.reads_forwarded,
            "note": (
                "Staged writes have not happened. They are sent when a decision confirming "
                "them arrives, and discarded if one contradicting them does."
            ),
        }


def mode_for(client_capabilities: Mapping[str, Any] | None) -> ClientMode:
    """Decide what this client can be trusted with, from what it advertised.

    Asked rather than assumed. Handing a placeholder to a client that does not understand it
    puts that placeholder in the next prompt, and the proxy cannot see the prompt to stop it.
    """
    capabilities = client_capabilities or {}
    experimental = capabilities.get("experimental") or {}
    if isinstance(experimental, Mapping) and DECISION_CAPABILITY in experimental:
        return ClientMode.HANDLES
    return ClientMode.BLOCKING


#: The proxy's own tools, so a client can see and steer what is being held.
PROXY_TOOLS: Sequence[Mapping[str, JsonValue]] = (
    {
        "name": "specunode.status",
        "description": "What the proxy is holding, and whether context identity is enforced.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "specunode.ledger",
        "description": "Effects dispatched and discarded so far in this session.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "specunode.stall",
        "description": "Stop speculating and run sequentially for the rest of the session.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "specunode.discard",
        "description": "Discard every held write unsent. Requires confirmation.",
        "inputSchema": {"type": "object", "properties": {"confirm": {"type": "boolean"}}},
    },
    {
        "name": "specunode.retire",
        "description": "Report the model's actual decision, releasing or discarding held writes.",
        "inputSchema": {
            "type": "object",
            "properties": {"tool": {"type": "string"}, "args": {"type": "object"}},
            "required": ["tool"],
        },
    },
    {
        "name": "specunode.replay_check",
        "description": "Whether this session's held writes match a journaled run.",
        "inputSchema": {"type": "object", "properties": {}},
    },
)


# -- the transport --------------------------------------------------------------------------
#
# Kept below the rules and deliberately thin. Everything above this line is testable without a
# socket, which matters because the rules carry the correctness claims and the SDK does not:
# `mcp` went from 1.x to 2.x with a breaking API change, and a design that put the staging
# rules inside protocol handlers would have to be re-verified every time that happens.


class ProxyUnsupported(RuntimeError):
    """The installed MCP SDK does not expose what the proxy needs."""


#: Probed against this. A newer SDK is allowed and warned about; a missing surface fails loudly.
PROBED_MCP = "2.2.0"


def probe_sdk() -> str:
    """Check the SDK is one this proxy knows how to drive."""
    try:
        import importlib.metadata as metadata

        version = metadata.version("mcp")
    except Exception as exc:  # pragma: no cover - mcp not installed
        raise ProxyUnsupported(
            "the MCP proxy needs the optional extra: pip install 'specunode[mcp]'"
        ) from exc
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise ProxyUnsupported(
            f"mcp {version} does not expose mcp.server.mcpserver.MCPServer (probed against "
            f"{PROBED_MCP}). Pin the SDK rather than running a proxy that silently forwards "
            "writes it was meant to hold."
        ) from exc
    # The proxy has to replace each tool's inferred schema with the upstream's, and the public
    # add_tool neither returns the Tool nor takes a schema. Checked here, at startup, because
    # the alternative is discovering it on the first tool call -- by which time a client is
    # connected and believes it is talking to something that holds writes.
    probe = MCPServer("specunode-probe")
    if not hasattr(probe, "run_stdio_async"):
        raise ProxyUnsupported(
            f"mcp {version} does not expose MCPServer.run_stdio_async (probed against "
            f"{PROBED_MCP}). The synchronous run() opens its own event loop, which would put "
            "the served tools and the upstream session on different loops and hang the first "
            "forwarded call."
        )
    manager = getattr(probe, "_tool_manager", None)
    if manager is None or not hasattr(manager, "add_tool"):
        raise ProxyUnsupported(
            f"mcp {version} does not expose MCPServer._tool_manager.add_tool (probed against "
            f"{PROBED_MCP}), so tool schemas cannot be forwarded from the upstream server. "
            "Pin the SDK rather than running a proxy that advertises the wrong schema."
        )
    return version


async def serve(
    upstream_command: Sequence[str],
    state: ProxyState,
    *,
    server_name: str = "specunode-proxy",
) -> None:
    """Run the stdio proxy against an upstream MCP server.

    Forwards ``tools/list`` with the upstream's annotations merged with the config's override
    table, runs reads immediately, and holds everything else in ``state`` until a decision
    arrives. The rules live in :class:`ProxyState`; this function only moves bytes.
    """
    probe_sdk()
    from mcp import types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.server.mcpserver import MCPServer

    params = StdioServerParameters(command=upstream_command[0], args=list(upstream_command[1:]))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as upstream:
        await upstream.initialize()
        listing = await upstream.list_tools()
        server = MCPServer(server_name)

        for tool in listing.tools:
            _register_proxied(server, upstream, state, tool, types)
        _register_control_tools(server, upstream, state)
        # ``run_stdio_async`` rather than ``run("stdio")`` in a thread. ``run`` opens its own
        # event loop, which put the served tools on one loop and the upstream ClientSession on
        # another -- so the first forwarded read awaited a session belonging to a loop that was
        # not running it, and the proxy hung instead of answering. Everything here has to share
        # one loop, because every proxied call is a call back out to the upstream.
        await server.run_stdio_async()


#: The attribute an SDK's tool model uses for its JSON Schema, newest spelling first.
_SCHEMA_ATTRS = ("input_schema", "inputSchema")


def _upstream_schema(tool: Any) -> Mapping[str, Any] | None:
    """The upstream tool's declared argument schema, whatever the SDK calls the field.

    Raises rather than returning ``None`` when the field is absent entirely. ``mcp`` 2.x renamed
    ``inputSchema`` to ``input_schema``, and a ``getattr(tool, "inputSchema", None)`` went on
    quietly returning ``None`` -- so the proxy advertised a schema it had inferred rather than
    the upstream's, and rejected every call made against the schema it advertised. A silent
    ``None`` on a field this load-bearing is the failure this module's probe exists to prevent,
    so a missing field is SDK drift and says so.
    """
    for attribute in _SCHEMA_ATTRS:
        if hasattr(tool, attribute):
            value = getattr(tool, attribute)
            return value if isinstance(value, Mapping) else None
    raise ProxyUnsupported(
        f"the tool {getattr(tool, 'name', '?')!r} has none of {_SCHEMA_ATTRS} (probed against "
        f"mcp {PROBED_MCP}), so its arguments cannot be forwarded. Pin the SDK rather than "
        "running a proxy that advertises a schema the upstream did not declare."
    )


def _register_proxied(server: Any, upstream: Any, state: ProxyState, tool: Any, types: Any) -> None:
    """Expose one upstream tool, classified and either forwarded or held.

    Two things here are about the SDK rather than about this proxy.

    ``proxied`` is annotated ``Any`` and not ``JsonValue``. The SDK derives a tool's schema
    from its function signature by building a pydantic model, and ``JsonValue`` is a recursive
    alias it cannot resolve in its own namespace -- registration raised ``PydanticUserError``
    and the proxy never came up at all against a real server. The rules tests could not see
    that: they exercise :class:`ProxyState` without a transport.

    The schema the client is then shown is the *upstream's*, copied over the one the SDK
    inferred. A proxy that advertised a free-form schema where the upstream declares typed
    arguments would push every argument error from the client's validation out to the server's,
    which is a worse place to find it.
    """

    async def proxied(**kwargs: Any) -> Any:
        # Drop the parameters the client did not send. See ``_with_upstream_signature``.
        kwargs = {name: value for name, value in kwargs.items() if value is not _ABSENT}
        spec = state.classify(tool.name)
        if spec.effect is EffectClass.READ:
            state.reads_forwarded += 1
            result = await upstream.call_tool(tool.name, kwargs)
            return [block.model_dump() for block in result.content]

        held = state.stage(tool.name, kwargs)
        if state.mode is ClientMode.HANDLES:
            return dict(state.result_for(held))

        # A client that cannot be told "this has not happened yet" waits instead. Returning a
        # handle here would put it in that client's next prompt, unseen by anything.
        #
        # It does not send the call itself. ``specunode.retire`` is the single party that
        # forwards a confirmed write -- which is what makes the HANDLES mode work at all (its
        # caller has already returned by the time a decision arrives, so a write confirmed
        # there used to be reported as dispatched and never sent), and what stops the two paths
        # from both sending in BLOCKING mode.
        decided = await state.wait_for_call(held)
        if state.outcome_of(held) == "dispatched":
            return state.result_of(held)
        if not decided:
            # A timeout is NOT a discard, and saying it was is the more dangerous of the two
            # lies: the call is still in ``state.staged``, ``specunode.status`` still lists it,
            # and the next matching decision forwards it upstream for real. A client told its
            # write was discarded reissues it, and then both go out.
            return {
                "_specunode": {
                    "timed_out": True,
                    "still_held": True,
                    "effect_id": held.effect_id,
                    "detail": (
                        "no decision arrived within the proxy's deadline. This write has NOT "
                        "been sent and has NOT been discarded -- it is still held, and a "
                        "later specunode.retire matching it will send it. Call "
                        "specunode.discard to drop it unsent."
                    ),
                }
            }
        return {"_specunode": {"discarded": True, "effect_id": held.effect_id}}

    schema = _upstream_schema(tool)
    registered = server._tool_manager.add_tool(
        _with_upstream_signature(proxied, schema),
        name=tool.name,
        description=tool.description,
        annotations=getattr(tool, "annotations", None),
    )
    if schema is not None:
        registered.parameters = dict(schema)


def _with_upstream_signature(fn: Any, schema: Any) -> Any:
    """Give a ``**kwargs`` forwarder the upstream tool's parameter names.

    The SDK parses a call's arguments against a model built from the function's *signature*,
    not from the schema the tool advertises. A bare ``**kwargs`` forwarder therefore advertises
    the upstream's schema and then rejects every call against it, asking for a literal
    ``kwargs`` field -- which is what happened, and what a rules-only test suite cannot see.

    Every parameter is typed ``Any`` on purpose. The upstream is the authority on its own
    argument types, it validates them itself, and a proxy that re-derived Python types from
    JSON Schema would invent disagreements. The schema the client is shown is still the
    upstream's, copied over the inferred one by the caller.
    """
    if not isinstance(schema, Mapping):
        return fn
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return fn
    required = schema.get("required")
    required_names = set(required) if isinstance(required, Sequence) else set()
    # Optional parameters default to a private sentinel, not to ``None``. The SDK builds its
    # argument model from this signature and materialises the default into ``**kwargs``, so a
    # ``None`` default meant an argument the client never sent arrived at the forwarder as an
    # explicit null -- which was then forwarded upstream, where any optional parameter that is
    # not nullable rejects it outright, and which also changed the canonical argument hash a
    # staged write is keyed and compared on. The forwarder strips the sentinel before it
    # forwards, so "absent" stays absent.
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if name in required_names else _ABSENT,
            annotation=Any,
        )
        for name in properties
        if name.isidentifier()
    ]
    fn.__signature__ = inspect.Signature(parameters)
    fn.__annotations__ = {name: Any for name in properties if name.isidentifier()}
    fn.__annotations__["return"] = Any
    return fn


def _register_control_tools(server: Any, upstream: Any, state: ProxyState) -> None:
    """The proxy's own six tools, so a client can see and steer what is held.

    Annotated with concrete types rather than ``JsonValue`` for the same reason the proxied
    tools are: the SDK builds each tool's schema by making a pydantic model from the signature,
    and a recursive alias it cannot resolve in its own namespace raises ``PydanticUserError``
    at registration time. That took the whole proxy down before it served a single request.
    """

    async def status() -> dict[str, Any]:
        return dict(state.status())

    async def ledger() -> dict[str, Any]:
        return {
            "dispatched": [c.effect_id for c in state.dispatched],
            "discarded": [c.effect_id for c in state.discarded],
            "still_held": [c.effect_id for c in state.staged],
            "context_identity": state.context_identity,
        }

    async def stall() -> dict[str, Any]:
        state.mode = ClientMode.BLOCKING
        return {"mode": state.mode.value}

    async def discard(confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            return {
                "error": "discard drops held writes unsent; call again with confirm=true",
            }
        return {"discarded": state.discard_all()}

    async def retire(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Report the model's decision, and forward whatever it confirms.

        The forwarding happens *here* and nowhere else. It used to happen in the blocked
        caller, which meant a write confirmed while the client was in HANDLES mode was recorded
        as dispatched and never actually sent -- the caller had returned a handle and ended
        long before any decision arrived. Three separate surfaces reported that effect as
        dispatched while the upstream had never heard of it.
        """
        confirmed, dropped = state.retire(ToolCall(name=tool, args=dict(args or {})))
        sent: list[str] = []
        failed: list[dict[str, str]] = []
        try:
            for call in confirmed:
                try:
                    result = await upstream.call_tool(call.tool, dict(call.args))
                except Exception as exc:  # an upstream that refused, timed out or died
                    # Recorded rather than raised: the other confirmed calls still have to be
                    # attempted, and a blocked caller is waiting for an answer of some kind.
                    reason = f"{type(exc).__name__}: {exc}"
                    failed.append({"effect_id": call.effect_id, "error": reason})
                    state.record_result(call, {"_specunode": {"error": reason}})
                    continue
                state.record_result(call, [block.model_dump() for block in result.content])
                sent.append(call.effect_id)
        finally:
            # After the results are recorded, and in a ``finally`` so a blocked client is
            # released even if this tool raises. Scoped to the calls this decision resolved:
            # a blanket release reached callers this decision never touched.
            state.publish([*confirmed, *dropped])
        return {
            "dispatched": sent,
            "discarded": [c.effect_id for c in dropped],
            "failed": failed,
        }

    async def replay_check() -> dict[str, Any]:
        return {
            "held": len(state.staged),
            "note": "The proxy has no journal of its own; a replay check needs the run's journal.",
        }

    for fn, name, description in (
        (status, "specunode.status", "What the proxy is holding."),
        (ledger, "specunode.ledger", "Effects dispatched and discarded this session."),
        (stall, "specunode.stall", "Stop returning handles; block on writes instead."),
        (discard, "specunode.discard", "Discard every held write unsent."),
        (retire, "specunode.retire", "Report the model's decision."),
        (replay_check, "specunode.replay_check", "Whether held writes match a journaled run."),
    ):
        server.add_tool(fn, name=name, description=description)
