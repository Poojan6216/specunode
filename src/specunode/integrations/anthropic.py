"""Anthropic provider adapter (optional extra: ``pip install 'specunode[anthropic]'``).

Lives in ``integrations/`` rather than ``core/`` deliberately. Hard Rule 1 says there is no
model SDK in the control path, and ``tests/test_no_llm_in_control_path.py`` enforces that by
scanning ``core/``, ``buffer/``, ``verify/`` and ``journal/`` for exactly this kind of import.
Putting the adapter here keeps that check meaningful instead of forcing an exemption into it.

The adapter translates and nothing else. It does not add a system block, a cached prefix, a
default preamble or any other content: :func:`~specunode.core.model.request_hash` is taken
immediately before the envelope arrives here, so anything this module added would be invisible
to Hard Rule 13 and would make the recorded hash describe a request that was never sent.

Hard Rule 11: requests go to the endpoint the developer configured, and the client is
constructed from that configuration alone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from specunode.canonical import JsonValue
from specunode.core.model import (
    ContentBlock,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic

__all__ = ["AnthropicModel", "block_to_api", "envelope_to_params"]


class _Messages(Protocol):  # pragma: no cover - structural, for typing only
    async def create(self, **kwargs: object) -> object: ...
    def stream(self, **kwargs: object) -> object: ...


def block_to_api(block: ContentBlock) -> JsonValue:
    """One content block in Anthropic Messages API shape."""
    match block:
        case TextBlock(text=text):
            return {"type": "text", "text": text}
        case ThinkingBlock(text=text, signature=signature):
            payload: dict[str, JsonValue] = {"type": "thinking", "thinking": text}
            if signature is not None:
                payload["signature"] = signature
            return payload
        case ToolUseBlock(id=block_id, name=name, args=args):
            return {"type": "tool_use", "id": block_id, "name": name, "input": dict(args)}
        case ToolResultBlock(tool_use_id=use_id, content=content, is_error=is_error):
            return {
                "type": "tool_result",
                "tool_use_id": use_id,
                "content": content
                if isinstance(content, str)
                else [{"type": "text", "text": str(content)}],
                "is_error": is_error,
            }


def envelope_to_params(envelope: RequestEnvelope) -> dict[str, JsonValue]:
    """Translate an envelope into Messages API parameters. Adds nothing of its own."""
    params: dict[str, JsonValue] = {
        "model": envelope.model,
        "max_tokens": envelope.max_tokens,
        "messages": [
            {"role": message.role, "content": [block_to_api(b) for b in message.content]}
            for message in envelope.messages
            if message.role != "system"
        ],
    }
    if envelope.system:
        params["system"] = [block_to_api(b) for b in envelope.system]
    if envelope.tools:
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": dict(t.input_schema)}
            for t in envelope.tools
        ]
    if envelope.tool_choice is not None:
        params["tool_choice"] = dict(envelope.tool_choice)
    if envelope.temperature is not None:
        params["temperature"] = envelope.temperature
    if envelope.top_p is not None:
        params["top_p"] = envelope.top_p
    if envelope.top_k is not None:
        params["top_k"] = envelope.top_k
    if envelope.stop_sequences:
        params["stop_sequences"] = list(envelope.stop_sequences)
    if envelope.thinking is not None:
        params["thinking"] = dict(envelope.thinking)
    return params


def _field(raw: object, name: str, default: object = None) -> object:
    """Read a field from an SDK object or from the plain dict a test supplies."""
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _block_from_api(raw: object) -> ContentBlock | None:
    kind = _field(raw, "type")
    if kind == "text":
        return TextBlock(text=str(_field(raw, "text", "")))
    if kind == "thinking":
        signature = _field(raw, "signature")
        return ThinkingBlock(
            text=str(_field(raw, "thinking", "")),
            signature=str(signature) if isinstance(signature, str) else None,
        )
    if kind == "tool_use":
        raw_input = _field(raw, "input", {})
        return ToolUseBlock(
            id=str(_field(raw, "id", "")),
            name=str(_field(raw, "name", "")),
            args=dict(raw_input) if isinstance(raw_input, Mapping) else {},
        )
    return None


def _blocks_from_api(content: object) -> tuple[ContentBlock, ...]:
    if not isinstance(content, list):
        return ()
    return tuple(block for block in (_block_from_api(raw) for raw in content) if block is not None)


def _usage_from_api(usage: object) -> Usage:
    def count(name: str) -> int:
        value = getattr(usage, name, 0) or 0
        return int(value) if isinstance(value, int) else 0

    return Usage(
        input_tokens=count("input_tokens"),
        output_tokens=count("output_tokens"),
        cache_read_tokens=count("cache_read_input_tokens"),
        cache_creation_tokens=count("cache_creation_input_tokens"),
    )


class AnthropicModel:
    """A :class:`~specunode.core.model.ModelClient` backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncAnthropic | None = None,
        max_retries: int = 2,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ModelError(
                "the Anthropic provider needs the optional extra: "
                "pip install 'specunode[anthropic]'"
            ) from exc
        # Same boundary, same reason as ``_params`` below: the SDK types each constructor
        # keyword individually and a ``dict[str, object]`` cannot be matched against them.
        kwargs: dict[str, object] = {"max_retries": max_retries}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**cast(Any, kwargs))

    # ``envelope_to_params`` returns a JSON-shaped mapping this module builds and validates
    # itself. The SDK types each request field as its own TypedDict, which cannot be expressed
    # through ``**kwargs`` of a ``dict[str, JsonValue]`` -- so mypy reports one error per field
    # it cannot match, about thirty of them, none of which is a real defect.
    #
    # Worth recording why this only appeared now: ``anthropic`` is in mypy's
    # ``ignore_missing_imports`` list and was not installed in the dev environment, so every
    # one of these calls type-checked against ``Any`` and "mypy --strict clean" meant nothing
    # here. Installing the extra is what made the checker look. The cast is narrow and
    # deliberate; the alternative of leaving the package uninstalled is how a whole file
    # escapes the type checker while appearing to pass it.
    def _params(self, envelope: RequestEnvelope) -> Any:
        return cast(Any, envelope_to_params(envelope))

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raw = await self._client.messages.create(**self._params(envelope))
        return ModelResponse(
            model=str(getattr(raw, "model", envelope.model)),
            content=_blocks_from_api(getattr(raw, "content", [])),
            stop_reason=str(getattr(raw, "stop_reason", "end_turn") or "end_turn"),
            usage=_usage_from_api(getattr(raw, "usage", None)),
        )

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        """Surface each ``tool_use`` block the moment it finishes parsing.

        That is the whole of the tier-0 drafter's input, and the reason it can issue a read
        before the turn ends. The pattern is Claude Code's streaming tool executor.
        """
        async with self._client.messages.stream(**self._params(envelope)) as stream:
            index = 0
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "text":
                    yield TextDelta(index=index, text=str(getattr(event, "text", "")))
                elif event_type == "content_block_stop":
                    block = getattr(event, "content_block", None)
                    parsed = _blocks_from_api([block] if block is not None else [])
                    if parsed and isinstance(parsed[0], ToolUseBlock):
                        yield ToolUseComplete(index=index, block=parsed[0])
                    index += 1
            final = await stream.get_final_message()
        yield TurnComplete(
            response=ModelResponse(
                model=str(getattr(final, "model", envelope.model)),
                content=_blocks_from_api(getattr(final, "content", [])),
                stop_reason=str(getattr(final, "stop_reason", "end_turn") or "end_turn"),
                usage=_usage_from_api(getattr(final, "usage", None)),
            )
        )
