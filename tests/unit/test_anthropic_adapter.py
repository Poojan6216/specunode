"""The Anthropic adapter translates and adds nothing (spec task 0.4, optional extra).

No API key is needed: the adapter is exercised against a fake client, because what is being
tested is the translation, not the network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
