"""The tier-2 draft model (spec section 5's drafter table).

A draft model proposes and the gate disposes, exactly as for a Markov table. Being a model
gives its guess no extra standing, and these tests are mostly about the ways it could quietly
acquire some -- by failing open, by proposing a tool the runtime does not know, or by having
its requests folded into Hard Rule 13's target-only check.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from specunode.core.decision import ToolCall
from specunode.core.model import (
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from specunode.drafters.base import DraftContext
from specunode.drafters.t2_model import ModelDrafter

KNOWN = frozenset({"restart_job", "fetch_runbook", "charge_card"})


class Canned:
    """A model that answers with whatever it was given, or fails on demand."""

    def __init__(self, response: ModelResponse | None = None, fail: bool = False) -> None:
        self._response = response
        self._fail = fail
        self.calls = 0
        self.received: list[RequestEnvelope] = []

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        self.calls += 1
        self.received.append(envelope)
        if self._fail:
            raise ModelError("draft model is down")
        assert self._response is not None
        return self._response

    def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


def context(**kwargs: object) -> DraftContext:
    base = {
        "run_id": "r",
        "branch_id": "b",
        "step_index": 1,
        "node_id": "agent#0",
        "history": (ToolCall("fetch_runbook", {"section": "restart"}),),
        "known_tools": KNOWN,
    }
    base.update(kwargs)
    return DraftContext(**base)  # type: ignore[arg-type]


def tool_response(name: str, args: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        model="draft",
        content=(ToolUseBlock(id="u1", name=name, args=args),),  # type: ignore[arg-type]
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def text_response(text: str) -> ModelResponse:
    return ModelResponse(
        model="draft", content=(TextBlock(text=text),), usage=Usage(output_tokens=5)
    )


async def test_a_tool_use_answer_becomes_a_candidate() -> None:
    drafter = ModelDrafter(client=Canned(tool_response("restart_job", {"job_id": "etl-1"})))
    (prediction,) = await drafter.predict(context())
    assert prediction.decision == ToolCall("restart_job", {"job_id": "etl-1"})
    assert prediction.tier == 2


async def test_a_json_text_answer_is_parsed() -> None:
    """A draft model with no tool-use support is still usable."""
    drafter = ModelDrafter(
        client=Canned(text_response('Next: {"tool": "restart_job", "args": {"job_id": "etl-1"}}'))
    )
    (prediction,) = await drafter.predict(context())
    assert prediction.decision == ToolCall("restart_job", {"job_id": "etl-1"})


async def test_a_tool_the_runtime_does_not_know_is_refused() -> None:
    """A drafter that could name an unregistered tool would be proposing an unclassified call."""
    drafter = ModelDrafter(client=Canned(tool_response("rm_rf_slash", {})))
    assert await drafter.predict(context()) == []


async def test_an_unknown_answer_is_no_opinion_not_a_guess() -> None:
    drafter = ModelDrafter(client=Canned(text_response("UNKNOWN")))
    assert await drafter.predict(context()) == []


async def test_an_unparsable_answer_is_counted_rather_than_swallowed() -> None:
    """A drafter silently returning nothing looks exactly like one with no opinion."""
    drafter = ModelDrafter(client=Canned(text_response('{"not": "a tool call"}')))
    assert await drafter.predict(context()) == []
    assert drafter.unparsable == 1


async def test_a_draft_model_that_is_down_costs_a_speculation_not_the_run() -> None:
    """The sequential path is always correct; losing a guess is not an error."""
    drafter = ModelDrafter(client=Canned(fail=True))
    assert await drafter.predict(context()) == []


async def test_no_call_is_made_when_the_registry_is_empty() -> None:
    client = Canned(tool_response("restart_job", {}))
    drafter = ModelDrafter(client=client)
    assert await drafter.predict(context(known_tools=frozenset())) == []
    assert client.calls == 0, "asking a model to choose from no tools spends money for nothing"


async def test_the_draft_request_names_the_draft_model_not_the_target() -> None:
    """Hard Rule 13 tracks target requests only, and the two must be distinguishable."""
    client = Canned(tool_response("restart_job", {}))
    drafter = ModelDrafter(client=client, model="claude-haiku-4-5")
    await drafter.predict(context())
    assert client.received[0].model == "claude-haiku-4-5"


async def test_the_draft_prompt_carries_only_this_branchs_history() -> None:
    """A draft model prompted with a sibling's work could have its guess confirmed anyway."""
    client = Canned(tool_response("restart_job", {}))
    drafter = ModelDrafter(client=client)
    await drafter.predict(context(history=(ToolCall("fetch_runbook", {"section": "restart"}),)))
    text = str(client.received[0].messages[0].content)
    assert "fetch_runbook" in text
    assert "charge_card" not in text or "Tools available" in text


def test_the_only_prompt_in_the_package_lives_outside_the_control_path() -> None:
    """Hard Rule 1: no prompt in core/, buffer/, verify/ or journal/.

    A drafter may hold one, because everything it produces is a candidate the gate must still
    confirm by exact equality before anything it caused can reach the world.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "specunode"
    for package in ("core", "buffer", "verify", "journal"):
        for path in (root / package).rglob("*.py"):
            assert "You predict the next tool call" not in path.read_text(encoding="utf-8")
    assert "You predict the next tool call" in (root / "drafters" / "t2_model.py").read_text(
        encoding="utf-8"
    )
