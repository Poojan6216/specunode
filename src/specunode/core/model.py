"""The model boundary: request envelopes, streaming, and the journaled wrapper.

Two models exist in this runtime and no others (Hard Rule 1). The **target** model's output
is the ground truth that retires or squashes a branch. An optional **draft** model produces
guesses that the gate must then confirm by exact canonical equality. Nothing here asks a
model anything about control flow, and nothing here writes a prompt: this module transports
the developer's request and records it.

The centre of the module is :func:`request_hash`, which is Hard Rule 13's instrument. A
speculative branch may send a model request only if the prompt it would send is one the
sequential run could send; at retirement the runtime rebuilds that prompt from the canonical
context and compares hashes. Two decisions make that check real rather than decorative:

**What is hashed is the whole envelope, not a suffix.** The fault classes Rule 13 names --
a message from a squashed sibling, results assembled in completion order rather than program
order -- all live upstream of any delta, and a suffix hash cannot see them. Sampling
parameters are hashed too: a branch that quietly sets a smaller ``max_tokens`` to make its
speculative turn cheap is asking a different question, and Rule 13 exists to make that a
fault rather than an optimisation.

**The hash is taken at the wire boundary**, inside :class:`JournaledModel`, immediately
before the envelope goes to the provider adapter. If the caller hashed instead, anything
between the caller and the socket -- an adapter injecting a cached system block, a default
preamble -- would leave a clean hash recorded for a dirty request, and Rule 13 would become a
no-op that still reports zero divergences.

What is excluded is exhaustive and published (``docs/replay.md``): runtime-only bookkeeping
on a message, ``cache_control`` blocks, transport fields, and correlation ids -- which are
rewritten to positional tokens rather than dropped, because a *mispaired* tool result must
still change the hash even though the id itself changes no token the model conditions on.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from specunode.canonical import JsonValue, canonical, chash
from specunode.core.decision import (
    Decision,
    FreeText,
    ToolCall,
    decision_key,
    decision_payload,
)
from specunode.ids import new_ulid
from specunode.journal.journal import Journal

__all__ = [
    "BuiltRequest",
    "CallScope",
    "ContentBlock",
    "ContextDivergence",
    "JournaledModel",
    "Message",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "OpaqueBlock",
    "PromptBuilder",
    "RecordedTurn",
    "RecordedTurnSource",
    "RequestEnvelope",
    "StreamEvent",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ToolDef",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUseComplete",
    "TurnComplete",
    "Usage",
    "check_structural",
    "current_scope",
    "decisions_of",
    "project",
    "request_hash",
    "scoped",
]

Role: TypeAlias = Literal["system", "user", "assistant"]


class ModelError(RuntimeError):
    """The provider could not answer."""


# -- content ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Reasoning content. Part of the request the model sees, so part of the hash."""

    text: str
    signature: str | None = None
    kind: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    args: Mapping[str, JsonValue]
    kind: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: JsonValue
    is_error: bool = False
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class OpaqueBlock:
    """A block this runtime does not interpret, carried exactly as the provider sent it.

    In a tool-use loop the API wants the model's reply back unchanged -- ``redacted_thinking``
    included, and whatever block types a later API version adds. A block with no class here
    used to be dropped on the way in, so the next request echoed a reply with a block missing,
    which the API refuses. ``payload`` is the block as the provider sent it, ``type`` included.
    It is part of what the model is asked, so it is part of the request hash.
    """

    type: str
    payload: Mapping[str, JsonValue]
    kind: Literal["opaque"] = "opaque"


ContentBlock: TypeAlias = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | OpaqueBlock


@dataclass(frozen=True, slots=True)
class Message:
    """One message, plus bookkeeping that never reaches the model.

    ``origin_branch`` is what makes Hard Rule 6 checkable: before serialising, the builder
    asserts every message's origin is in this branch's lineage, so a sibling's message cannot
    reach a prompt even by accident.
    """

    role: Role
    content: tuple[ContentBlock, ...]
    #: Runtime-only, excluded from the hash.
    id: str = ""
    origin_branch: str = ""
    journal_offset: int | None = None


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    input_schema: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
        )


# -- the request ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Everything the model is asked, and nothing about how it is transported."""

    model: str
    messages: tuple[Message, ...] = ()
    system: tuple[ContentBlock, ...] = ()
    tools: tuple[ToolDef, ...] = ()
    tool_choice: Mapping[str, JsonValue] | None = None
    max_tokens: int = 4096
    #: ``None`` means "not specified", like ``top_p`` and ``top_k`` beside it, and nothing is
    #: sent to the provider. It defaulted to ``0.0``, so every request carried a temperature
    #: nobody had asked for -- and the current models reject sampling parameters outright
    #: (``temperature`` is deprecated on Claude Sonnet 5 and Opus 5, a 400), so the first real
    #: call the online benchmark ever made failed on a field the developer never set.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] = ()
    thinking: Mapping[str, JsonValue] | None = None
    #: Transport only: excluded from the hash, because streaming or not changes no token.
    stream: bool = False

    def with_messages(self, messages: Sequence[Message]) -> RequestEnvelope:
        return replace(self, messages=tuple(messages))


def _block_payload(block: ContentBlock, ids: Mapping[str, str]) -> JsonValue:
    """One content block, with correlation ids rewritten to positional tokens."""
    match block:
        case TextBlock(text=text):
            return {"kind": "text", "text": text}
        case ThinkingBlock(text=text, signature=signature):
            return {"kind": "thinking", "text": text, "signature": signature}
        case ToolUseBlock(id=block_id, name=name, args=args):
            return {
                "kind": "tool_use",
                "ref": ids.get(block_id, block_id),
                "name": name,
                "args": args,
            }
        case ToolResultBlock(tool_use_id=use_id, content=content, is_error=is_error):
            return {
                "kind": "tool_result",
                "ref": ids.get(use_id, use_id),
                "content": content,
                "is_error": is_error,
            }
        case OpaqueBlock(type=block_type, payload=payload):
            return {"kind": "opaque", "type": block_type, "payload": dict(payload)}


def _positional_ids(messages: Sequence[Message]) -> dict[str, str]:
    """Map each provider tool_use id to ``tu:<turn>:<ordinal>``.

    Rewritten rather than dropped. The id itself changes no token the model conditions on, so
    hashing it would make every replay diverge for no reason; but the *pairing* it expresses
    does matter, and a result attached to the wrong call lands at a different ordinal and so
    still moves the hash.
    """
    mapping: dict[str, str] = {}
    turn = 0
    for message in messages:
        if message.role != "assistant":
            continue
        ordinal = 0
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                mapping[block.id] = f"tu:{turn}:{ordinal}"
                ordinal += 1
        turn += 1
    return mapping


def project(envelope: RequestEnvelope) -> JsonValue:
    """The canonical projection of a request: exactly what the model is being asked.

    Excluded, exhaustively: a message's ``id``, ``origin_branch`` and ``journal_offset``;
    ``cache_control`` (a billing hint, not content); ``stream`` and every other transport
    field; and raw correlation ids, which are rewritten positionally above.
    """
    ids = _positional_ids(envelope.messages)
    return {
        "model": envelope.model,
        "system": [_block_payload(block, ids) for block in envelope.system],
        "messages": [
            {"role": message.role, "content": [_block_payload(b, ids) for b in message.content]}
            for message in envelope.messages
        ],
        "tools": [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in envelope.tools
        ],
        "tool_choice": envelope.tool_choice,
        "params": {
            "max_tokens": envelope.max_tokens,
            "temperature": envelope.temperature,
            "top_p": envelope.top_p,
            "top_k": envelope.top_k,
            "stop_sequences": list(envelope.stop_sequences),
            "thinking": envelope.thinking,
        },
    }


def request_hash(envelope: RequestEnvelope) -> str:
    """Hard Rule 13's prompt identity."""
    return chash(project(envelope))


def canonical_request(envelope: RequestEnvelope) -> bytes:
    """The exact bytes the handle scan runs over, so the scan and the hash agree."""
    return canonical(project(envelope))


# -- the response ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelResponse:
    model: str
    content: tuple[ContentBlock, ...] = ()
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))


def decisions_of(response: ModelResponse) -> tuple[Decision, ...]:
    """The ordered decision sequence a turn yields.

    One turn is the resolution *event*; it produces one decision per ``tool_use`` block, in
    stream order. A turn with no tool calls yields a single :class:`FreeText`, which is a
    barrier: nothing speculates on it and it never resolves equal to anything.
    """
    uses = response.tool_uses
    if uses:
        return tuple(ToolCall(name=use.name, args=use.args) for use in uses)
    return (FreeText.of(response.text),)


# -- streaming -------------------------------------------------------------------------------


class TurnResults(list):  # type: ignore[type-arg]
    """What one model turn's tool calls returned, in the order the model asked for them.

    A list, so every caller that indexed or iterated ``call_turn``'s result still works, with
    the reply that produced it attached. A multi-turn loop needs that reply: the next request
    has to carry the assistant's content back *unchanged* -- thinking blocks and their
    signatures included -- and pair each result with the ``tool_use`` id that asked for it.
    """

    def __init__(self, results: Sequence[JsonValue], response: ModelResponse | None) -> None:
        super().__init__(results)
        self.response = response


@dataclass(frozen=True, slots=True)
class TextDelta:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class ToolUseComplete:
    """A ``tool_use`` block has finished parsing out of the stream.

    This is the tier-0 drafter's entire input: the call can be issued now, before the turn
    ends. The pattern is Claude Code's streaming tool executor, credited as theirs.
    """

    index: int
    block: ToolUseBlock


@dataclass(frozen=True, slots=True)
class TurnComplete:
    response: ModelResponse


StreamEvent: TypeAlias = TextDelta | ToolUseComplete | TurnComplete


@runtime_checkable
class ModelClient(Protocol):
    """What the runtime needs from a model. Implemented by providers, replay and test doubles."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse: ...

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]: ...


# -- faithful serialisation, for the journal --------------------------------------------------
#
# Distinct from project(): the projection rewrites correlation ids positionally because the
# model does not condition on them, but the journal must be able to rebuild the exact response
# the provider returned, ids included, or replay could not reproduce a turn.


def block_to_json(block: ContentBlock) -> JsonValue:
    match block:
        case TextBlock(text=text):
            return {"kind": "text", "text": text}
        case ThinkingBlock(text=text, signature=signature):
            return {"kind": "thinking", "text": text, "signature": signature}
        case ToolUseBlock(id=block_id, name=name, args=args):
            return {"kind": "tool_use", "id": block_id, "name": name, "args": dict(args)}
        case ToolResultBlock(tool_use_id=use_id, content=content, is_error=is_error):
            return {
                "kind": "tool_result",
                "tool_use_id": use_id,
                "content": content,
                "is_error": is_error,
            }
        case OpaqueBlock(type=block_type, payload=payload):
            return {"kind": "opaque", "type": block_type, "payload": dict(payload)}


def block_from_json(payload: Mapping[str, JsonValue]) -> ContentBlock:
    kind = payload.get("kind")
    match kind:
        case "text":
            return TextBlock(text=str(payload["text"]))
        case "thinking":
            signature = payload.get("signature")
            return ThinkingBlock(
                text=str(payload["text"]),
                signature=str(signature) if isinstance(signature, str) else None,
            )
        case "tool_use":
            args = payload["args"]
            if not isinstance(args, Mapping):
                raise ValueError(f"tool_use block has non-object args: {payload!r}")
            return ToolUseBlock(id=str(payload["id"]), name=str(payload["name"]), args=args)
        case "tool_result":
            return ToolResultBlock(
                tool_use_id=str(payload["tool_use_id"]),
                content=payload["content"],
                is_error=bool(payload.get("is_error", False)),
            )
        case "opaque":
            raw = payload["payload"]
            if not isinstance(raw, Mapping):
                raise ValueError(f"opaque block has a non-object payload: {payload!r}")
            return OpaqueBlock(type=str(payload["type"]), payload=dict(raw))
        case _:
            raise ValueError(f"unknown content block kind {kind!r}")


def message_to_json(message: Message) -> JsonValue:
    return {"role": message.role, "content": [block_to_json(b) for b in message.content]}


def response_to_json(response: ModelResponse) -> JsonValue:
    return {
        "model": response.model,
        "content": [block_to_json(b) for b in response.content],
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read": response.usage.cache_read_tokens,
            "cache_creation": response.usage.cache_creation_tokens,
        },
    }


def response_from_json(payload: Mapping[str, JsonValue]) -> ModelResponse:
    blocks = payload.get("content")
    if not isinstance(blocks, Sequence) or isinstance(blocks, str):
        raise ValueError("model_response payload has no content list")
    raw_usage = payload.get("usage")
    usage_map: Mapping[str, JsonValue] = raw_usage if isinstance(raw_usage, Mapping) else {}

    def count(name: str) -> int:
        value = usage_map.get(name, 0)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

    return ModelResponse(
        model=str(payload.get("model", "")),
        content=tuple(block_from_json(b) for b in blocks if isinstance(b, Mapping)),
        stop_reason=str(payload.get("stop_reason", "end_turn")),
        usage=Usage(
            input_tokens=count("input_tokens"),
            output_tokens=count("output_tokens"),
            cache_read_tokens=count("cache_read"),
            cache_creation_tokens=count("cache_creation"),
        ),
    )


# -- who is calling ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CallScope:
    """The branch a model call belongs to.

    Carried in a ContextVar rather than as an argument, because :class:`JournaledModel` has to
    satisfy :class:`ModelClient` in order to be dropped in wherever the developer's graph
    already calls a model -- the point of the LangGraph integration is that their graph file
    does not change. Each asyncio task gets its own copy of the context, so one branch cannot
    scribble on another's scope (Hard Rule 6).
    """

    run_id: str = ""
    branch_id: str = ""
    lineage: tuple[str, ...] = ()
    step: int = 0
    node_id: str = ""
    speculative: bool = False
    tier: int | None = None
    #: The idempotency key of the effect being dispatched, when one is. The fake world
    #: records it, so a duplicate delivery is attributable to a key rather than only to a
    #: branch.
    effect_key: str = ""
    #: Hard Rule 13: the branch records (step, request_hash) for every target request it
    #: sends, and retirement rebuilds each one from the canonical context and compares.
    record_prompt: Callable[[int, str], None] | None = None
    #: Told +1 when a target turn is asked for and -1 once its response is journaled, so the
    #: branch can refuse a write made while the decision behind it is not on disk (Hard Rule 5).
    track_turn: Callable[[int], None] | None = None


call_scope: ContextVar[CallScope | None] = ContextVar("specunode_call_scope", default=None)

#: True while the runtime reads a stream on a node's behalf (``call_turn``): if it fails, the
#: node is handed none of it, so the turn is closed rather than left open to refuse its writes.
_partial_discarded: ContextVar[bool] = ContextVar("specunode_partial_discarded", default=False)


@contextmanager
def partial_turns_discarded() -> Iterator[None]:
    """Streams read inside this block hand a failed turn's blocks to nobody."""
    token = _partial_discarded.set(True)
    try:
        yield
    finally:
        _partial_discarded.reset(token)


def current_scope() -> CallScope:
    scope = call_scope.get()
    if scope is None:
        raise ModelError(
            "a model call was made outside any run scope. The runtime sets the scope around "
            "every node it drives; a node that starts a raw thread without propagating the "
            "context loses it (see docs/limitations.md)."
        )
    return scope


@contextmanager
def scoped(scope: CallScope) -> Iterator[None]:
    token = call_scope.set(scope)
    try:
        yield
    finally:
        call_scope.reset(token)


@dataclass(frozen=True, slots=True)
class RecordedTurn:
    """A model turn the journal already holds, served to a resumed run instead of asked again."""

    response: ModelResponse
    #: Offset of the ``model_response`` entry the turn was first recorded in.
    offset: int
    #: The error the turn ended in, if it failed -- raised again when it is served, so a node
    #: that caught it and asked again is matched with what it asked the second time.
    failed: str | None = None
    #: The caller stopped waiting before it had the whole turn -- a timeout, a cancel. Served
    #: as a turn that never finishes: what the caller saw of it, then nothing, until it stops
    #: waiting again -- for as long as it waited the first time, and a margin.
    cancelled: bool = False
    #: How long the recorded caller waited for this turn.
    latency_ms: int = 0


#: The outcome recorded for a turn its caller stopped waiting for.
CANCELLED = "cancelled: the caller stopped waiting before the turn was handed over"

#: How much longer a served "never answers" turn waits than twice the recorded caller did. The
#: recorded wait is the model call's alone; a node whose deadline also covers earlier work --
#: faster on a resume -- stops waiting later into the call, and too short a bound failed a
#: resume that was doing exactly what it did before.
_ABANDON_MARGIN_S = 30.0


class TurnAbandoned(BaseException):
    """A resumed or replayed node kept waiting for a turn the recorded run stopped waiting for.

    The recorded node gave up on this question -- its timeout fired, or something cancelled
    it -- and went on to decide what it decided. This one is still waiting, so something that
    shaped it changed across the crash, and it is not making the calls it made before. Waiting
    forever hung the resume with no word of why; answering from the live model could decide
    differently from what may already have been sent. So it stops here, and says so.

    A ``BaseException``, like a cancellation, and not a :class:`ModelError` or any other
    ``Exception``: it ends the node. As a ``RuntimeError`` it was caught by an ordinary
    ``except Exception`` fallback, and the node went on to ask something the recorded run never
    asked -- live, and a card was charged a second time. Do not catch it.
    """


def _abandon_after_s(recorded: RecordedTurn) -> float:
    return 2 * recorded.latency_ms / 1000.0 + _ABANDON_MARGIN_S


async def _served_never_answers(recorded: RecordedTurn) -> None:
    """Wait well past when the recorded caller stopped; then give up, loudly."""
    await asyncio.sleep(_abandon_after_s(recorded))
    raise TurnAbandoned(
        "the run being resumed or replayed stopped waiting for this turn after "
        f"{recorded.latency_ms / 1000.0:.1f} s, and this node is still waiting after "
        f"{_abandon_after_s(recorded):.0f} s: it is not asking what it asked before -- "
        "something that shaped it changed -- so it stops here rather than decide anew"
    )


def _waited_ms(recorded: RecordedTurn | None, begun: float) -> int:
    """How long a served turn's caller waited, for its outcome: never less than the recorded
    wait -- journaled as 0, the next resume gave up after the margin alone."""
    waited = int((time.monotonic() - begun) * 1000)
    return max(waited, recorded.latency_ms if recorded is not None else 0)


async def _let_finish(writing: asyncio.Future[Any]) -> None:
    """Wait for a journal write to finish, however many times the caller is cancelled meanwhile.

    The write is on the journal's own thread and finishes anyway; a caller that stopped waiting
    for it went on before the outcome it records was on disk, and a write still queued behind
    another could be cancelled before it ran at all.
    """
    while not writing.done():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait({writing})


def _failure_text(exc: BaseException) -> str:
    """What a failed turn's outcome says: a ModelError's message, anything else's type too."""
    return str(exc) if isinstance(exc, ModelError) else f"{type(exc).__name__}: {exc}"


def _partial_response(
    envelope: RequestEnvelope, blocks: Mapping[int, ContentBlock], texts: Mapping[int, list[str]]
) -> ModelResponse:
    """What of a failed stream reached the caller, in block order."""
    content: list[ContentBlock] = []
    for index in sorted(set(blocks) | set(texts)):
        block = blocks.get(index)
        content.append(block if block is not None else TextBlock(text="".join(texts[index])))
    return ModelResponse(model=envelope.model, content=tuple(content), stop_reason="error")


class RecordedTurnSource(Protocol):
    """Where a resumed run's recorded turns come from; see ``journal.replay.RecordedTurns``."""

    def take(self, digest: str, scope: CallScope) -> RecordedTurn | None:
        """The recorded answer to this request, or ``None`` if the model must be asked."""
        ...


#: Stop reasons that mean the reply did not finish: it ran out of room, or was stopped.
CUT_OFF = frozenset({"max_tokens", "model_context_window_exceeded", "refusal"})


def refuse_cut_off(response: ModelResponse) -> None:
    """Raise if ``response`` was cut off with a tool call in it.

    A call cut off mid-argument can still parse -- ``"amount": 150.0`` stopped after the ``1``
    is a charge of 1 -- and nothing downstream can tell it from a call the model finished. So a
    reply that stopped before it finished -- out of tokens or context, or stopped by a refusal
    -- and asked for anything is not acted on at all, and is not journaled as an answer; the
    node sees a ``ModelError`` and can ask again.
    """
    if response.stop_reason in CUT_OFF and any(
        isinstance(block, ToolUseBlock) for block in response.content
    ):
        raise ModelError(
            f"the reply stopped at {response.stop_reason} with a tool call in it, which may be "
            "incomplete; nothing in it is acted on -- give the model room and ask again"
        )


class JournaledModel:
    """Wraps a :class:`ModelClient` so every request and response is durable before use.

    Hard Rule 5. The request is journaled *before* the provider is called, so a crash mid-call
    leaves evidence that the call was attempted; the response is journaled *before* it is
    returned to the caller, so the runtime cannot act on an output that is not on disk.

    The hash is taken here, at the wire boundary, and not by the caller -- see the module
    docstring for why that distinction is what makes Hard Rule 13 more than decoration.
    """

    def __init__(
        self,
        inner: ModelClient,
        journal: Journal,
        *,
        role: Literal["target", "draft"] = "target",
        provider: str = "",
    ) -> None:
        self._inner = inner
        self._journal = journal
        self._role = role
        self._provider = provider
        self._recorded: dict[str, RecordedTurnSource] = {}

    def serve_recorded(self, run_id: str, source: RecordedTurnSource | None) -> None:
        """Answer run ``run_id``'s questions from ``source`` where it can; ``None`` to stop.

        The scheduler sets this while it resumes a run. A node whose branch never retired is
        run again from the same position and asks what it asked before. Where the dead
        process's answer may already have sent something, asking again risked a different
        answer -- a different call, under a different idempotency key, which the dedupe table
        cannot connect to what already went out. A served turn is journaled again under the
        resumed branch, with ``recorded_from`` naming the entry it came from.

        Kept per run: this object is often shared, and a fresh run on another scheduler must
        not switch off a resume in flight.
        """
        if source is None:
            self._recorded.pop(run_id, None)
        else:
            self._recorded[run_id] = source

    def _recorded_for(self, digest: str, scope: CallScope) -> RecordedTurn | None:
        source = self._recorded.get(scope.run_id)
        if source is None or self._role != "target" or scope.speculative:
            return None
        return source.take(digest, scope)

    async def _journal_request(
        self,
        envelope: RequestEnvelope,
        scope: CallScope,
        digest: str,
        recorded: RecordedTurn | None,
    ) -> str:
        request_id = new_ulid()
        # Hard Rule 13 tracks target requests only. A draft request legitimately differs --
        # a different model, at minimum -- and folding it in would make every tier-2 run
        # raise ContextDivergence on its first call, whose obvious fix is to loosen the
        # comparison until it stops catching real target-side divergence too.
        if self._role == "target" and scope.record_prompt is not None:
            scope.record_prompt(scope.step, digest)
        await self._journal.append_async(
            scope.run_id,
            "model_request",
            {
                "v": 1,
                "step": scope.step,
                "branch_id": scope.branch_id,
                "lineage": list(scope.lineage),
                "node_id": scope.node_id,
                "speculative": scope.speculative,
                "tier": scope.tier,
                "role": self._role,
                "provider": self._provider,
                "model": envelope.model,
                "stream": envelope.stream,
                "request_id": request_id,
                "request_hash": digest,
                "request": project(envelope),
                **({"recorded_from": recorded.offset} if recorded is not None else {}),
            },
        )
        return request_id

    async def _journal_response(
        self,
        response: ModelResponse,
        scope: CallScope,
        request_id: str,
        digest: str,
        latency_ms: int,
        recorded: RecordedTurn | None = None,
        *,
        failed: str | None = None,
        cancelled: bool = False,
    ) -> None:
        if failed is None:
            decisions = decisions_of(response)
            decided: dict[str, JsonValue] = {
                "decision": decision_payload(decisions[0]),
                "decisions": [decision_payload(d) for d in decisions],
                "decision_hash": decision_key(decisions[0]),
                "end_of_turn": True,
            }
        else:
            # A failed turn decides nothing. Its outcome is recorded all the same: a node that
            # catches the failure and asks again asks a second question, and on a resume or a
            # replay the first has to fail again for the second to be matched with its answer.
            # Unrecorded, the first had no answer, and every question the node asked after it
            # went to the live model -- whose answer could differ from one that already sent
            # something.
            decided = {"decision": None, "decisions": [], "end_of_turn": False, "failed": failed}
            if cancelled:
                decided["cancelled"] = True
        await self._journal.append_async(
            scope.run_id,
            "model_response",
            {
                "v": 1,
                "step": scope.step,
                "branch_id": scope.branch_id,
                "request_id": request_id,
                "request_hash": digest,
                "role": self._role,
                "provider": self._provider,
                "speculative": scope.speculative,
                "response": response_to_json(response),
                **decided,
                # The literal text whose hash is FreeText.content_hash. A FreeText decision
                # carries only a digest, so without this the run would not be replayable.
                "text": response.text or None,
                "latency_ms": latency_ms,
                # Served from the journal, not asked: no model call was made for this entry.
                **({"recorded_from": recorded.offset} if recorded is not None else {}),
            },
        )

    async def _outcome(
        self,
        response: ModelResponse,
        scope: CallScope,
        request_id: str,
        digest: str,
        latency_ms: int,
        recorded: RecordedTurn | None = None,
        *,
        failed: str | None = None,
    ) -> None:
        """Journal a turn's outcome -- and if the caller stops waiting while it is written, say so.

        The write runs on the journal's own thread and finishes whatever the caller does: a
        node whose timeout fired during it never saw the answer, and it was on disk as though
        it had. A resume then served that answer to the question the node had given up on, and
        the node acted on it as well as on the answer it asked for next. So the write is let
        finish, and a second outcome records that the turn was never handed over.
        """
        writing = asyncio.ensure_future(
            self._journal_response(
                response, scope, request_id, digest, latency_ms, recorded, failed=failed
            )
        )
        try:
            await asyncio.shield(writing)
        except asyncio.CancelledError:
            await _let_finish(writing)
            if not writing.cancelled():
                writing.exception()  # retrieved, so a failed write is not reported as unobserved
            await self._cancelled(response, scope, request_id, digest, latency_ms, recorded)
            raise

    async def _cancelled(
        self,
        response: ModelResponse,
        scope: CallScope,
        request_id: str,
        digest: str,
        latency_ms: int,
        recorded: RecordedTurn | None = None,
    ) -> None:
        """Record that the caller stopped waiting -- a timeout, a cancel -- before it had the turn.

        Left unrecorded, the question had no outcome: on resume it matched nothing, and every
        question the node asked after it went to the live model, which could decide
        differently and charge again. Written to the end even if the caller is cancelled again.
        """
        writing = asyncio.ensure_future(
            self._journal_response(
                response,
                scope,
                request_id,
                digest,
                latency_ms,
                recorded,
                failed=CANCELLED,
                cancelled=True,
            )
        )
        await _let_finish(writing)
        if not writing.cancelled():
            writing.exception()  # retrieved, so a failed write is not reported as unobserved

    async def _ask(
        self,
        envelope: RequestEnvelope,
        scope: CallScope,
        digest: str,
        recorded: RecordedTurn | None,
    ) -> str:
        """Journal the question -- and if the caller stops waiting while it is written, say so.

        The write finishes on the journal's thread whatever the caller does, and a question on
        disk with no outcome sent every later question at its position to the live model on
        resume. So it is let finish, and given its outcome: the caller never waited for an
        answer to it.
        """
        writing = asyncio.ensure_future(self._journal_request(envelope, scope, digest, recorded))
        try:
            return await asyncio.shield(writing)
        except asyncio.CancelledError:
            await _let_finish(writing)
            if not writing.cancelled() and writing.exception() is None:
                empty = ModelResponse(model=envelope.model, stop_reason="error")
                waited = recorded.latency_ms if recorded is not None else 0
                await self._cancelled(empty, scope, writing.result(), digest, waited, recorded)
            raise

    def _in_run(self, scope: CallScope) -> bool:
        """Whether this process still drives the scope's run -- a stream closed by the garbage
        collector after the run ended is not, and writes nothing into a run it let go of."""
        return self._journal.holds(scope.run_id)

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        scope = current_scope()
        digest = request_hash(envelope)
        recorded = self._recorded_for(digest, scope)
        request_id = await self._ask(envelope, scope, digest, recorded)
        track = scope.track_turn if self._role == "target" else None
        if track is not None:
            track(1)
        try:
            if recorded is not None:
                served = recorded.response
                if recorded.cancelled:
                    begun = time.monotonic()
                    try:
                        await _served_never_answers(recorded)
                    except asyncio.CancelledError:
                        waited = _waited_ms(recorded, begun)
                        await self._cancelled(served, scope, request_id, digest, waited, recorded)
                        raise
                    except TurnAbandoned:
                        # Given an outcome too, so the next resume is served this one again.
                        waited = recorded.latency_ms
                        await self._cancelled(served, scope, request_id, digest, waited, recorded)
                        raise
                await self._outcome(
                    served, scope, request_id, digest, 0, recorded, failed=recorded.failed
                )
                if recorded.failed is not None:
                    raise ModelError(recorded.failed)
                return served
            started = time.monotonic()
            response: ModelResponse | None = None
            try:
                response = await self._inner.complete(envelope)
                refuse_cut_off(response)
            except asyncio.CancelledError:
                empty = ModelResponse(model=envelope.model, stop_reason="error")
                latency_ms = int((time.monotonic() - started) * 1000)
                await self._cancelled(response or empty, scope, request_id, digest, latency_ms)
                raise
            except Exception as exc:
                failed = response or ModelResponse(model=envelope.model, stop_reason="error")
                latency_ms = int((time.monotonic() - started) * 1000)
                failure = _failure_text(exc)
                await self._outcome(failed, scope, request_id, digest, latency_ms, failed=failure)
                if isinstance(exc, ModelError):
                    raise
                # What a served failure raises, so a node that catches it does so the same way
                # live, on resume and in replay; the client's own error is its cause.
                raise ModelError(failure) from exc
            latency_ms = int((time.monotonic() - started) * 1000)
            await self._outcome(response, scope, request_id, digest, latency_ms)
            return response
        finally:
            # Nothing of the turn reaches the caller before its response is on disk.
            if track is not None:
                track(-1)

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        """Stream, journaling the turn before ``TurnComplete`` is handed to the caller.

        Intermediate :class:`ToolUseComplete` events are forwarded as they parse, which is what
        lets the tier-0 drafter issue a read before the turn ends. A read is not an effect, so
        that is not a Hard Rule 5 violation; a write is. ``call_turn`` stages a turn's writes
        only after its stream ends, and a guessed branch's writes wait for the confirming entry
        -- journaled here, before ``TurnComplete`` escapes. A node that reads this stream itself
        and writes as a block parses would have its write dispatched first, on a decision not
        yet on disk: nothing stops the drain once the node parks. So the turn is reported to
        the branch as open until its response is journaled (``CallScope.track_turn``), and a
        write made while it is open -- or after the node stopped reading before its end -- is
        refused.
        """
        scope = current_scope()
        digest = request_hash(envelope)
        recorded = self._recorded_for(digest, scope)
        request_id = await self._ask(envelope, scope, digest, recorded)
        track = scope.track_turn if self._role == "target" else None
        if track is not None:
            track(1)
        handed_over = False
        try:
            if recorded is not None:
                # The recorded turn's blocks, in order and with no delay, as a live stream
                # would emit them -- so early issue sees a served turn as it saw the original.
                response = recorded.response
                begun = time.monotonic()
                try:
                    for index, block in enumerate(response.content):
                        if isinstance(block, ToolUseBlock):
                            handed_over = True
                            yield ToolUseComplete(index=index, block=block)
                        elif isinstance(block, TextBlock):
                            handed_over = True
                            yield TextDelta(index=index, text=block.text)
                    if recorded.cancelled:
                        await _served_never_answers(recorded)
                except (asyncio.CancelledError, GeneratorExit, TurnAbandoned) as stopped:
                    finalised = isinstance(stopped, GeneratorExit) and not self._in_run(scope)
                    if recorded.cancelled and not finalised:
                        waited = (
                            recorded.latency_ms
                            if isinstance(stopped, TurnAbandoned)
                            else _waited_ms(recorded, begun)
                        )
                        await self._cancelled(response, scope, request_id, digest, waited, recorded)
                    raise
                await self._outcome(
                    response, scope, request_id, digest, 0, recorded, failed=recorded.failed
                )
                if recorded.failed is not None:
                    raise ModelError(recorded.failed)
                if track is not None:
                    track(-1)
                    track = None
                yield TurnComplete(response=response)
                return
            started = time.monotonic()
            completed = False
            #: An outcome is on disk, or will be: a turn has exactly one, the last word.
            recorded_outcome = False
            blocks: dict[int, ContentBlock] = {}
            texts: dict[int, list[str]] = {}
            final: ModelResponse | None = None
            try:
                async for event in self._inner.stream(envelope):
                    if isinstance(event, TurnComplete):
                        final = event.response
                        refuse_cut_off(event.response)
                        latency_ms = int((time.monotonic() - started) * 1000)
                        recorded_outcome = True
                        await self._outcome(event.response, scope, request_id, digest, latency_ms)
                        completed = True
                        if track is not None:
                            track(-1)
                            track = None
                    elif isinstance(event, ToolUseComplete):
                        blocks[event.index] = event.block
                    elif isinstance(event, TextDelta):
                        texts.setdefault(event.index, []).append(event.text)
                    handed_over = True
                    yield event
                if not completed:
                    # A stream that simply stops -- a dropped connection the client did not
                    # report -- ended a turn that was never journaled. Treated as what it is,
                    # a failure: read as a complete turn, it let a confirmed guess's write go
                    # out with no decision on disk.
                    raise ModelError("the model's stream ended without completing its turn")
            except (asyncio.CancelledError, GeneratorExit) as stopped:
                # The caller stopped waiting -- its timeout, or it stopped reading -- before the
                # turn was handed over. What it saw of the reply is recorded with that. Not a
                # stream the garbage collector closes after the run ended: the process no
                # longer drives that run, and its "run_finished" is already written.
                finalised = isinstance(stopped, GeneratorExit) and not self._in_run(scope)
                if not recorded_outcome and not finalised:
                    latency_ms = int((time.monotonic() - started) * 1000)
                    seen = final or _partial_response(envelope, blocks, texts)
                    await self._cancelled(seen, scope, request_id, digest, latency_ms)
                raise
            except Exception as exc:
                if recorded_outcome:
                    raise
                # What of the reply reached the caller is recorded with the failure, and
                # served again before it, so a resumed node sees what this one saw.
                latency_ms = int((time.monotonic() - started) * 1000)
                failure = _failure_text(exc)
                await self._outcome(
                    final or _partial_response(envelope, blocks, texts),
                    scope,
                    request_id,
                    digest,
                    latency_ms,
                    failed=failure,
                )
                if isinstance(exc, ModelError):
                    raise
                raise ModelError(failure) from exc
        except BaseException:
            # A turn that failed before the caller saw any of it left nothing to act on, and
            # neither did one read by call_turn, which hands a failed turn to nobody. One a
            # node read itself and stopped reading part-way stays open: it may act on what it
            # saw.
            if track is not None and (not handed_over or _partial_discarded.get()):
                track(-1)
            raise


# -- building a request, and Hard Rule 13's structural check -------------------------------------


#: Kept in step with specunode.core.hazards.HANDLE_PREFIX; a test asserts the two agree.
_HANDLE_SCAN = re.compile(rb"(?i)\$specunode\.handle:")


class ContextDivergence(RuntimeError):
    """A request was, or would have been, different from the one the sequential run would send.

    Carries the step and what differed, because "the context diverged" without saying where is
    not actionable, and this fault squashes a branch whose model output is then thrown away.
    """

    def __init__(self, step: int, problems: Sequence[str], *, kind: str = "structural") -> None:
        self.step = step
        self.problems = tuple(problems)
        self.kind = kind
        super().__init__(f"context divergence at step {step} ({kind}):\n  " + "\n  ".join(problems))


@dataclass(frozen=True, slots=True)
class BuiltRequest:
    """A request, its identity, and the record of anything the runtime did not derive."""

    envelope: RequestEnvelope
    request_hash: str
    #: Indices in the message list of blocks the node supplied rather than the runtime deriving.
    #: Recorded at build time so the retirement check can compare the derivable part exactly
    #: and count the rest, instead of failing on material it could never have rebuilt.
    injected_index: tuple[int, ...] = ()
    injected_hash: str | None = None


def check_structural(envelope: RequestEnvelope, attested: frozenset[str]) -> list[str]:
    """Hard Rule 13's total check, over the bytes actually being sent.

    Three named fault classes, and this catches all three without needing to rebuild anything:

    1. a placeholder anywhere in the request -- the model would be conditioned on a value that
       does not exist
    2. a message from a branch whose output may not be read back -- a squashed sibling, or one
       that has not resolved yet. The tempting predicate "origin is not a squashed branch"
       admits the *unresolved* sibling, which is exactly the Hard Rule 6 channel, so the test
       is membership in the attested set rather than absence from a blacklist
    3. tool results out of program order, or gapped, or split across several user messages --
       the model must see results in the order it asked for them, not the order they finished

    None of these can false-positive, so none is ever waived.
    """
    problems: list[str] = []

    # The loose, case-insensitive scan over the exact bytes canonical() produced. Defined
    # here rather than imported from hazards, which imports branch, which imports this module.
    if _HANDLE_SCAN.search(canonical_request(envelope)):
        problems.append("a staged write's placeholder appears in the request")

    for index, message in enumerate(envelope.messages):
        origin = message.origin_branch
        if origin and origin not in attested:
            problems.append(
                f"messages[{index}] came from branch {origin!r}, which is neither retired nor "
                "in this branch's lineage"
            )

    for index, message in enumerate(envelope.messages):
        results = [b for b in message.content if isinstance(b, ToolResultBlock)]
        if not results:
            continue
        if len(results) != len(message.content):
            problems.append(f"messages[{index}] mixes tool results with other content")
        others = [
            other
            for other in envelope.messages[index + 1 :]
            if any(isinstance(b, ToolResultBlock) for b in other.content)
        ]
        preceding_uses = _tool_use_ids_before(envelope.messages, index)
        ordered = [b.tool_use_id for b in results]
        if ordered != preceding_uses[: len(ordered)]:
            problems.append(
                f"messages[{index}] returns tool results in completion order "
                f"{ordered} rather than program order {preceding_uses[: len(ordered)]}"
            )
        if others and preceding_uses and len(ordered) < len(preceding_uses):
            problems.append(
                f"messages[{index}] carries {len(ordered)} of {len(preceding_uses)} results; "
                "one turn's results belong in one message"
            )
    return problems


def _tool_use_ids_before(messages: Sequence[Message], index: int) -> list[str]:
    """The tool_use ids of the most recent assistant turn before ``index``, in program order."""
    for message in reversed(messages[:index]):
        uses = [b.id for b in message.content if isinstance(b, ToolUseBlock)]
        if uses:
            return uses
    return []


@dataclass
class PromptBuilder:
    """The only code in the runtime that constructs a target-model request.

    One builder for sequential mode, speculative mode, replay and the retirement rebuild, so
    the four cannot drift apart. Anything a node supplies that the runtime did not derive --
    a system message assembled from graph state, a retrieved document, a templated turn --
    must pass through :meth:`inject`, which records it so the retirement check can compare the
    derivable part exactly and count the rest.

    Without that split, Rule 13 would report a divergence on every step of every ordinary
    LangGraph app, and the repair an implementer reaches for is to rebuild from the branch's
    own message list -- which compares the list to itself and can never fail.
    """

    base: RequestEnvelope
    _injected: list[tuple[int, ContentBlock]] = field(default_factory=list)

    def inject(self, index: int, block: ContentBlock) -> None:
        """Declare a block the runtime did not derive, at its position in the message list."""
        self._injected.append((index, block))

    def build(self, messages: Sequence[Message], *, attested: frozenset[str]) -> BuiltRequest:
        """Assemble, check, and hash -- in that order.

        The check runs before the hash and before anything leaves, so a request that would
        violate Rule 13 is never sent rather than being detected after the fact.
        """
        envelope = self.base.with_messages(messages)
        problems = check_structural(envelope, attested)
        if problems:
            raise ContextDivergence(-1, problems, kind="structural")
        indices = tuple(sorted(index for index, _ in self._injected))
        return BuiltRequest(
            envelope=envelope,
            request_hash=request_hash(envelope),
            injected_index=indices,
            injected_hash=(
                chash([block_to_json(block) for _, block in sorted(self._injected)])
                if self._injected
                else None
            ),
        )
