"""A reply cut off with a tool call in it is never acted on.

The real Anthropic SDK assembles a message from whatever arrived, and a tool call cut off
mid-argument still parses: ``"amount": 150.0`` stopped after the ``1`` is a charge of 1. The
adapter built a finished turn from that when the connection dropped, and rewrote a missing stop
reason to "end_turn"; and nothing checked ``stop_reason``, so a reply that ran out of
``max_tokens`` in the middle of a call was sent as it stood. Found by the eleventh review.

These run the real SDK offline, against an HTTP transport that replays server-sent events.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import RunSession
from specunode.core.loop import agent_loop
from specunode.core.model import (
    JournaledModel,
    Message,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    TextBlock,
    ToolUseBlock,
    Usage,
    scoped,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")

ASK = RequestEnvelope(
    model="claude-sonnet-5",
    max_tokens=256,
    stream=True,
    messages=(Message(role="user", content=(TextBlock(text="Charge cus-1 25 and cus-2 150."),)),),
)


def sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


START = sse(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
)


def call(index: int, arguments: str, *, stopped: bool) -> str:
    out = sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": f"toolu_{index}",
                "name": "charge_card",
                "input": {},
            },
        },
    )
    out += sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        },
    )
    if stopped:
        out += sse("content_block_stop", {"type": "content_block_stop", "index": index})
    return out


def finish(stop_reason: str) -> str:
    return sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 256},
        },
    ) + sse("message_stop", {"type": "message_stop"})


WHOLE = call(0, '{"customer_id": "cus-1", "amount": 25.0}', stopped=True)
#: The connection drops after "amount": 1 of a charge of 150.0 -- nothing after it arrives.
DROPPED = START + WHOLE + call(1, '{"customer_id": "cus-2", "amount": 1', stopped=False)
#: The reply runs out of tokens in the same place, and the API finishes the message there.
MAX_TOKENS = (
    START
    + WHOLE
    + call(1, '{"customer_id": "cus-2", "amount": 1', stopped=True)
    + finish("max_tokens")
)


def model_serving(body: str) -> object:
    from specunode.integrations.anthropic import AnthropicModel

    def handler(request: object) -> object:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )

    sdk = anthropic.AsyncAnthropic(
        api_key="offline",
        base_url="http://offline.invalid",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return AnthropicModel(client=sdk, cache=False)


async def charge_through(tmp_path: Path, body: str) -> tuple[list[str], Journal, str]:
    taken: list[str] = []

    @tool(effect="write", idempotent=False)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        taken.append(f"{customer_id} {amount}")
        return {"charge_id": f"ch_{len(taken)}"}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        await agent_loop(session, ASK, max_turns=2)
        session.state["billed"] = True
        return ToolCall("charge_card", {})

    def route(state: Mapping[str, JsonValue]) -> str | None:
        return None if state.get("billed") else "bill"

    registry = registry_of([charge_card])
    journal = Journal(tmp_path / "journal.db")
    scheduler = Scheduler(
        graph=PlainAdapter.of([bill], route),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(model_serving(body), journal),  # type: ignore[arg-type]
        policy=Policy(speculation=False),
    )
    result = await scheduler.run("01CUTOFFAAAAAAAAAAAAAAAAAA", {})
    assert not result.ok
    return taken, journal, str(result.error)


async def test_a_reply_whose_connection_dropped_is_refused(tmp_path: Path) -> None:
    taken, journal, error = await charge_through(tmp_path, DROPPED)
    assert taken == [], "a call from a reply that never finished was sent"
    assert "before the model finished" in error, error
    assert not list(journal.read("01CUTOFFAAAAAAAAAAAAAAAAAA", kinds=["model_response"]))


async def test_a_reply_cut_off_at_max_tokens_is_not_acted_on(tmp_path: Path) -> None:
    """Not even the call before the cut, which is whole: the reply is one decision."""
    taken, journal, error = await charge_through(tmp_path, MAX_TOKENS)
    assert taken == [], "a charge of 1 -- a call cut off mid-argument -- was sent"
    assert "max_tokens" in error, error
    assert not list(journal.read("01CUTOFFAAAAAAAAAAAAAAAAAA", kinds=["model_response"]))


@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded", "refusal"])
async def test_complete_refuses_a_cut_off_reply_too(tmp_path: Path, stop: str) -> None:
    class CutOff:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            return ModelResponse(
                model="m",
                content=(ToolUseBlock(id="t", name="charge_card", args={"amount": 1}),),
                stop_reason=stop,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

        def stream(self, envelope: RequestEnvelope) -> object:
            raise NotImplementedError

    from specunode.core.model import CallScope

    journal = Journal(tmp_path / "journal.db")
    target = JournaledModel(CutOff(), journal)  # type: ignore[arg-type]
    with (
        scoped(CallScope(run_id="01CUTOFFBAAAAAAAAAAAAAAAAA", branch_id="b", node_id="n#0")),
        pytest.raises(ModelError, match=stop),
    ):
        await target.complete(ASK)
    assert not list(journal.read("01CUTOFFBAAAAAAAAAAAAAAAAA", kinds=["model_response"]))
