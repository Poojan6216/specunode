"""Model doubles: a scripted target, and a recorder that keeps what it was actually sent.

:class:`ScriptedModel` plays a fixed sequence of turns. Its ids are derived from position
rather than minted, so the same script produces the same journal every time and a replay of
that journal reproduces it.

:class:`RecordingModel` exists for one specific reason. Hard Rule 13's live check and the
retirement-time rebuild share a prompt builder, so they can be wrong in the same way and
still agree with each other -- a Rule 13 implementation that hashes a clean envelope while a
dirty one goes to the wire reports zero divergences forever. The context-equivalence test
(spec task 5.5) therefore asserts against what a model *received*, which is what this class
records, and never against what the runtime says it sent.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field

from specunode.canonical import JsonValue
from specunode.core.model import (
    ModelClient,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
    Usage,
    project,
    request_hash,
)

__all__ = ["RecordedCall", "RecordingModel", "ScriptedModel", "free_text_turn", "tool_turn"]


def tool_turn(
    *calls: tuple[str, Mapping[str, JsonValue]],
    turn: int = 0,
    model: str = "scripted",
    output_tokens: int = 64,
) -> ModelResponse:
    """A turn that emits one or more tool calls.

    Tool-use ids are positional (``toolu_<turn>_<ordinal>``) rather than random, so a script
    produces the same journal on every run.
    """
    return ModelResponse(
        model=model,
        content=tuple(
            ToolUseBlock(id=f"toolu_{turn}_{ordinal}", name=name, args=dict(args))
            for ordinal, (name, args) in enumerate(calls)
        ),
        stop_reason="tool_use",
        usage=Usage(input_tokens=128, output_tokens=output_tokens),
    )


def free_text_turn(text: str, *, model: str = "scripted", output_tokens: int = 64) -> ModelResponse:
    """A turn that emits prose. A speculation barrier: nothing predicts it."""
    return ModelResponse(
        model=model,
        content=(TextBlock(text=text),),
        stop_reason="end_turn",
        usage=Usage(input_tokens=128, output_tokens=output_tokens),
    )


@dataclass
class ScriptedModel:
    """Plays a fixed list of turns, in call order."""

    turns: Sequence[ModelResponse]
    #: Latency before each block is emitted by :meth:`stream`. Task 3.1 needs a stream long
    #: enough that an early-issued read can demonstrably finish before the turn ends.
    block_delay_ms: float = 0.0
    complete_delay_ms: float = 0.0
    #: Latency before :meth:`stream` emits anything: reading the prompt and thinking, the part
    #: of a reply that does not grow with what it says. It is what a run pays once per reply,
    #: and so what asking for several calls in one reply saves.
    reply_delay_ms: float = 0.0
    #: Turns already played, so the next call serves ``turns[consumed]``.
    #:
    #: A resumed run is a *new process*: its script starts over while the run does not, so a
    #: node that is the run's second model call becomes the script's first and is handed the
    #: wrong turn. A real model would simply be asked the new question and would answer it;
    #: the off-by-one is an artefact of scripting, not of resuming. Set this to the number of
    #: turns the journal already records so the stand-in behaves like the thing it stands in
    #: for.
    consumed: int = 0
    #: Every envelope this model was handed, in order.
    received: list[RequestEnvelope] = field(default_factory=list)
    _calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._calls = self.consumed

    def _next(self, envelope: RequestEnvelope) -> ModelResponse:
        self.received.append(envelope)
        if self._calls >= len(self.turns):
            raise ModelError(
                f"the script has {len(self.turns)} turns and the run asked for turn "
                f"{self._calls + 1}"
            )
        response = self.turns[self._calls]
        self._calls += 1
        return response

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        response = self._next(envelope)
        if self.complete_delay_ms:
            await asyncio.sleep(self.complete_delay_ms / 1000.0)
        return response

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        response = self._next(envelope)
        if self.reply_delay_ms:
            await asyncio.sleep(self.reply_delay_ms / 1000.0)
        for index, block in enumerate(response.content):
            if self.block_delay_ms:
                await asyncio.sleep(self.block_delay_ms / 1000.0)
            if isinstance(block, ToolUseBlock):
                yield ToolUseComplete(index=index, block=block)
            elif isinstance(block, TextBlock):
                yield TextDelta(index=index, text=block.text)
        if self.block_delay_ms:
            await asyncio.sleep(self.block_delay_ms / 1000.0)
        yield TurnComplete(response=response)

    @property
    def calls(self) -> int:
        return self._calls


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One request as the model actually received it."""

    envelope: RequestEnvelope
    projection: JsonValue
    digest: str
    streamed: bool


@dataclass
class RecordingModel:
    """Wraps a model and keeps the projection and hash of every request it was handed."""

    inner: ModelClient
    calls: list[RecordedCall] = field(default_factory=list)

    def _record(self, envelope: RequestEnvelope, *, streamed: bool) -> None:
        self.calls.append(
            RecordedCall(
                envelope=envelope,
                projection=project(envelope),
                digest=request_hash(envelope),
                streamed=streamed,
            )
        )

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self._record(envelope, streamed=False)
        return await self.inner.complete(envelope)

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        self._record(envelope, streamed=True)
        async for event in self.inner.stream(envelope):
            yield event

    @property
    def digests(self) -> tuple[str, ...]:
        return tuple(call.digest for call in self.calls)
