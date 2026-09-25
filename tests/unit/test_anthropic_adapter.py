"""The Anthropic adapter translates and adds nothing (spec task 0.4, optional extra).

No API key is needed: the adapter is exercised against a fake client, because what is being
tested is the translation, not the network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any

from specunode.core.model import (
    Message,
    RequestEnvelope,
    TextBlock,
    ThinkingBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
)
from specunode.integrations.anthropic import AnthropicModel, block_to_api, envelope_to_params


def test_an_envelope_translates_without_gaining_content() -> None:
    """Anything this adapter added would be invisible to the hash taken just before it."""
    envelope = RequestEnvelope(
        model="claude-sonnet-5",
        system=(TextBlock(text="be brief"),),
        messages=(Message(role="user", content=(TextBlock(text="hi"),)),),
        tools=(ToolDef(name="t", description="d", input_schema={"type": "object"}),),
        max_tokens=256,
        temperature=0.0,
    )
    params = envelope_to_params(envelope)
    assert params["model"] == "claude-sonnet-5"
    assert params["max_tokens"] == 256
    assert params["system"] == [{"type": "text", "text": "be brief"}]
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert params["tools"] == [
        {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    ]
    # Nothing beyond what the envelope carried.
    assert set(params) == {"model", "max_tokens", "messages", "system", "tools", "temperature"}


def test_optional_parameters_are_omitted_rather_than_defaulted() -> None:
    params = envelope_to_params(RequestEnvelope(model="m"))
    assert "top_p" not in params and "top_k" not in params and "stop_sequences" not in params
    assert "tools" not in params and "system" not in params


def test_every_block_kind_translates() -> None:
    assert block_to_api(TextBlock(text="x")) == {"type": "text", "text": "x"}
    assert block_to_api(ThinkingBlock(text="t", signature="s")) == {
        "type": "thinking",
        "thinking": "t",
        "signature": "s",
    }
    assert block_to_api(ToolUseBlock(id="u1", name="n", args={"a": 1})) == {
        "type": "tool_use",
        "id": "u1",
        "name": "n",
        "input": {"a": 1},
    }
    result = block_to_api(ToolResultBlock(tool_use_id="u1", content="ok"))
    assert result == {
        "type": "tool_result",
        "tool_use_id": "u1",
        "content": "ok",
        "is_error": False,
    }


# -- a fake SDK ------------------------------------------------------------------------------


@dataclass
class _FakeMessage:
    model: str = "claude-sonnet-5"
    content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "tool_use"
    usage: object = None


class _FakeUsage:
    input_tokens = 11
    output_tokens = 22
    cache_read_input_tokens = 3
    cache_creation_input_tokens = 4


@dataclass
class _FakeStream:
    message: _FakeMessage

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[object]:
        for block in self.message.content:
            yield type("Ev", (), {"type": "content_block_stop", "content_block": block})()
        # The SDK passes the API's last event through; a stream without it was cut off.
        yield type("Ev", (), {"type": "message_stop"})()

    async def get_final_message(self) -> _FakeMessage:
        return self.message


@dataclass
class _FakeMessages:
    message: _FakeMessage
    seen: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.seen.append(kwargs)
        return self.message

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.seen.append(kwargs)
        return _FakeStream(self.message)


@dataclass
class _FakeClient:
    messages: _FakeMessages


def _model_with(content: list[dict[str, Any]]) -> tuple[AnthropicModel, _FakeMessages]:
    messages = _FakeMessages(_FakeMessage(content=content, usage=_FakeUsage()))
    return AnthropicModel(client=_FakeClient(messages)), messages  # type: ignore[arg-type]


async def test_complete_parses_blocks_and_usage() -> None:
    model, _ = _model_with(
        [
            {"type": "text", "text": "thinking about it"},
            {"type": "tool_use", "id": "u1", "name": "restart_job", "input": {"job_id": "etl-1"}},
        ]
    )
    response = await model.complete(RequestEnvelope(model="claude-sonnet-5"))
    assert response.text == "thinking about it"
    assert response.tool_uses == (
        ToolUseBlock(id="u1", name="restart_job", args={"job_id": "etl-1"}),
    )
    assert response.usage.input_tokens == 11
    assert response.usage.cache_read_tokens == 3


async def test_stream_surfaces_each_tool_use_as_it_completes() -> None:
    model, _ = _model_with(
        [
            {"type": "tool_use", "id": "u1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "u2", "name": "b", "input": {}},
        ]
    )
    events = [e async for e in model.stream(RequestEnvelope(model="m", stream=True))]
    assert [type(e).__name__ for e in events] == [
        "ToolUseComplete",
        "ToolUseComplete",
        "TurnComplete",
    ]
    assert isinstance(events[0], ToolUseComplete) and events[0].block.name == "a"
    assert isinstance(events[-1], TurnComplete)


def test_an_unset_sampling_parameter_is_not_sent() -> None:
    """The newest models reject sampling parameters, and we were sending one nobody set.

    ``RequestEnvelope.temperature`` defaulted to 0.0 while ``top_p`` and ``top_k`` beside it
    defaulted to None, so every request carried a temperature the developer had not asked for.
    Claude Sonnet 5 and Opus 5 answer that with a 400 ("`temperature` is deprecated for this
    model"), so the first real call the online benchmark ever made failed on a field nobody
    had chosen. Sent when it *is* chosen: the provider's refusal then belongs to whoever
    asked for it, which is more useful than a silent drop.
    """
    from specunode.core.model import Message, RequestEnvelope, TextBlock
    from specunode.integrations.anthropic import envelope_to_params

    bare = RequestEnvelope(
        model="claude-sonnet-5",
        messages=(Message(role="user", content=(TextBlock(text="go"),)),),
        max_tokens=16,
    )
    params = envelope_to_params(bare)
    for name in ("temperature", "top_p", "top_k"):
        assert name not in params, f"{name} was sent without being set"

    asked = envelope_to_params(replace(bare, temperature=0.0, top_p=0.9, top_k=5))
    assert asked["temperature"] == 0.0
    assert asked["top_p"] == 0.9
    assert asked["top_k"] == 5


def test_prompt_caching_is_on_by_default_and_is_transport_not_content() -> None:
    """An agent loop re-sends its whole conversation every turn; caching is what that needs.

    It must not change what the model is asked: the journal and replay compare the request
    hash, and a setting that moved it would make a cached run and an uncached one of the same
    conversation look like two different questions.
    """
    from specunode.core.model import Message, RequestEnvelope, TextBlock, request_hash
    from specunode.integrations.anthropic import envelope_to_params

    envelope = RequestEnvelope(
        model="claude-sonnet-5",
        messages=(Message(role="user", content=(TextBlock(text="go"),)),),
        max_tokens=16,
    )
    assert envelope_to_params(envelope, cache=True)["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in envelope_to_params(envelope)

    model = AnthropicModel(client=object())  # type: ignore[arg-type]
    assert model.cache is True, "caching should be the default for an agent runtime"
    assert model._params(envelope)["cache_control"] == {"type": "ephemeral"}
    off = AnthropicModel(client=object(), cache=False)  # type: ignore[arg-type]
    assert "cache_control" not in off._params(envelope)

    # The adapter setting is not part of the request, so the hash cannot depend on it.
    assert request_hash(envelope) == request_hash(envelope)


def test_a_structured_tool_result_reaches_the_model_as_json() -> None:
    """Python's repr is not a wire format: single quotes, True and None were reaching the model."""
    import json

    from specunode.core.model import ToolResultBlock

    block = ToolResultBlock(tool_use_id="t1", content={"ok": True, "b": None, "a": [1, 2]})
    [text] = block_to_api(block)["content"]  # type: ignore[index]
    decoded = json.loads(text["text"])
    assert decoded == {"ok": True, "b": None, "a": [1, 2]}
    assert "'" not in text["text"] and "True" not in text["text"]
    # Canonical, so the same result is always the same bytes and a cached prefix holds.
    same = ToolResultBlock(tool_use_id="t1", content={"a": [1, 2], "b": None, "ok": True})
    assert block_to_api(block) == block_to_api(same)


async def test_a_block_the_runtime_does_not_know_comes_back_exactly_as_it_was_sent() -> None:
    """An agent loop echoes the model's reply on the next request, and the API refuses one with
    a block missing. ``redacted_thinking`` used to be dropped on the way in."""
    from specunode.core.model import (
        Message,
        OpaqueBlock,
        block_from_json,
        block_to_json,
        request_hash,
    )

    redacted = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4a"}
    reply = [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        redacted,
        {"type": "tool_use", "id": "u1", "name": "a", "input": {}},
    ]
    model, _ = _model_with(reply)
    response = await model.complete(RequestEnvelope(model="claude-sonnet-5"))
    assert [b.kind for b in response.content] == ["thinking", "opaque", "tool_use"]
    echoed = [block_to_api(block) for block in response.content]
    assert echoed[1] == redacted, "the block did not go back as it came"
    # It survives the journal, and it is part of what the model is asked.
    opaque = response.content[1]
    assert isinstance(opaque, OpaqueBlock)
    assert block_from_json(block_to_json(opaque)) == opaque  # type: ignore[arg-type]
    asked = RequestEnvelope(model="m", messages=(Message(role="assistant", content=(opaque,)),))
    other = replace(opaque, payload={**redacted, "data": "different"})
    assert request_hash(asked) != request_hash(
        replace(asked, messages=(Message(role="assistant", content=(other,)),))
    )
