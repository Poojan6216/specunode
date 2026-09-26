"""The model boundary, the journaled wrapper and replay (spec task 0.4).

The load-bearing property is Hard Rule 13's prompt identity: two requests hash equal exactly
when the model is being asked the same question. Too strict and every replay diverges on
noise; too loose and a branch that asked something different retires anyway. Most of this
file is about where that line sits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from specunode.core.decision import FreeText, ToolCall, decisions_equal
from specunode.core.model import (
    CallScope,
    JournaledModel,
    Message,
    ModelResponse,
    RequestEnvelope,
    TextBlock,
    ThinkingBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
    decisions_of,
    request_hash,
    response_from_json,
    response_to_json,
    scoped,
)
from specunode.journal.journal import Journal
from specunode.journal.replay import ReplayDivergence, ReplayExhausted, ReplayModel
from specunode.testing.models import RecordingModel, ScriptedModel, free_text_turn, tool_turn

RUN = "01MODELRUNAAAAAAAAAAAAAAAA"


def envelope(**overrides: object) -> RequestEnvelope:
    base = RequestEnvelope(
        model="scripted",
        system=(TextBlock(text="You help with support tickets."),),
        messages=(Message(role="user", content=(TextBlock(text="refund cus-1"),)),),
        tools=(
            ToolDef(name="lookup_customer", description="look up", input_schema={"type": "object"}),
            ToolDef(name="charge_card", description="charge", input_schema={"type": "object"}),
        ),
        max_tokens=1024,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


# -- what the hash sees ---------------------------------------------------------------------


def test_the_same_request_hashes_the_same() -> None:
    assert request_hash(envelope()) == request_hash(envelope())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("max_tokens", 512),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 40),
        ("stop_sequences", ("STOP",)),
        ("tool_choice", {"type": "tool", "name": "charge_card"}),
        ("thinking", {"type": "enabled", "budget_tokens": 1024}),
    ],
)
def test_every_sampling_parameter_is_part_of_the_question(field: str, value: object) -> None:
    """A branch that quietly makes its speculative turn cheaper is asking something else."""
    assert request_hash(envelope(**{field: value})) != request_hash(envelope())


def test_changing_one_token_of_the_system_prompt_changes_the_hash() -> None:
    """Attack 7.8, first half."""
    changed = envelope(system=(TextBlock(text="You help with support ticket."),))
    assert request_hash(changed) != request_hash(envelope())


def test_adding_or_reordering_a_tool_changes_the_hash() -> None:
    """Attack 7.8, second half. Order matters: it is part of what the model is shown."""
    base = envelope()
    extra = ToolDef(name="send_email", description="email", input_schema={"type": "object"})
    assert request_hash(envelope(tools=(*base.tools, extra))) != request_hash(base)
    assert request_hash(envelope(tools=tuple(reversed(base.tools)))) != request_hash(base)


def test_runtime_bookkeeping_on_a_message_is_not_part_of_the_question() -> None:
    """ids, origin branch and journal offsets change no token the model conditions on."""
    tagged = envelope(
        messages=(
            Message(
                role="user",
                content=(TextBlock(text="refund cus-1"),),
                id="msg-77",
                origin_branch="br-3",
                journal_offset=91,
            ),
        )
    )
    assert request_hash(tagged) == request_hash(envelope())


def test_provider_correlation_ids_do_not_change_the_hash_but_mispairing_does() -> None:
    """The id itself carries no meaning; which result answers which call carries all of it."""

    def conversation(first: str, second: str, pair_first_with: str) -> RequestEnvelope:
        return envelope(
            messages=(
                Message(role="user", content=(TextBlock(text="go"),)),
                Message(
                    role="assistant",
                    content=(
                        ToolUseBlock(id=first, name="lookup_customer", args={"customer_id": "c1"}),
                        ToolUseBlock(id=second, name="lookup_customer", args={"customer_id": "c2"}),
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        ToolResultBlock(tool_use_id=pair_first_with, content={"balance": 10}),
                    ),
                ),
            )
        )

    renamed_ids = request_hash(conversation("toolu_a", "toolu_b", "toolu_a"))
    different_ids = request_hash(conversation("toolu_x", "toolu_y", "toolu_x"))
    assert renamed_ids == different_ids, "renaming ids consistently must not move the hash"

    mispaired = request_hash(conversation("toolu_a", "toolu_b", "toolu_b"))
    assert mispaired != renamed_ids, "a result attached to the wrong call must move the hash"


def test_message_order_is_part_of_the_question() -> None:
    """Results assembled in completion order rather than program order must be visible."""
    ordered = envelope(
        messages=(
            Message(role="user", content=(TextBlock(text="a"),)),
            Message(role="user", content=(TextBlock(text="b"),)),
        )
    )
    swapped = envelope(
        messages=(
            Message(role="user", content=(TextBlock(text="b"),)),
            Message(role="user", content=(TextBlock(text="a"),)),
        )
    )
    assert request_hash(ordered) != request_hash(swapped)


def test_streaming_is_transport_and_not_part_of_the_question() -> None:
    assert request_hash(envelope(stream=True)) == request_hash(envelope(stream=False))


def test_thinking_content_is_part_of_the_question() -> None:
    with_thinking = envelope(
        messages=(Message(role="assistant", content=(ThinkingBlock(text="hmm"),)),)
    )
    without = envelope(messages=(Message(role="assistant", content=(TextBlock(text="hmm"),)),))
    assert request_hash(with_thinking) != request_hash(without)


# -- decisions out of a turn ------------------------------------------------------------------


def test_a_turn_yields_one_decision_per_tool_use_block_in_stream_order() -> None:
    turn = tool_turn(("a", {"x": 1}), ("b", {"y": 2}))
    assert decisions_of(turn) == (ToolCall("a", {"x": 1}), ToolCall("b", {"y": 2}))


def test_a_prose_turn_yields_a_barrier() -> None:
    (decision,) = decisions_of(free_text_turn("I think we should escalate."))
    assert isinstance(decision, FreeText)
    assert not decisions_equal(decision, decision)


def test_a_response_round_trips_through_the_journal_form() -> None:
    turn = tool_turn(("a", {"x": 1}), ("b", {"y": [1, 2]}), turn=3)
    assert response_to_json(response_from_json(response_to_json(turn))) == response_to_json(turn)


# -- the journaled wrapper -----------------------------------------------------------------------


async def _drive(
    journal: Journal, model: ScriptedModel | RecordingModel, envelopes: Sequence[RequestEnvelope]
) -> list[ModelResponse]:
    journaled = JournaledModel(model, journal, provider="scripted")
    out: list[ModelResponse] = []
    for step, env in enumerate(envelopes):
        with scoped(CallScope(run_id=RUN, branch_id="br-canon", step=step, node_id="agent")):
            out.append(await journaled.complete(env))
    return out


async def test_the_request_is_journaled_before_the_response(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    model = ScriptedModel(turns=[tool_turn(("lookup_customer", {"customer_id": "c1"}))])
    await _drive(journal, model, [envelope()])

    kinds = [e.kind for e in journal.read(RUN)]
    assert kinds == ["model_request", "model_response"]


async def test_the_journaled_hash_is_the_hash_of_what_the_model_received(tmp_path: Path) -> None:
    """The hash is taken at the wire boundary, so it cannot describe a request never sent."""
    journal = Journal(tmp_path / "j.db")
    recorder = RecordingModel(ScriptedModel(turns=[tool_turn(("a", {}))]))
    await _drive(journal, recorder, [envelope()])

    journaled = [e for e in journal.read(RUN) if e.kind == "model_request"]
    assert [e.payload["request_hash"] for e in journaled] == list(recorder.digests)


async def test_the_prompt_recorder_sees_target_requests(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    seen: list[tuple[int, str]] = []
    journaled = JournaledModel(ScriptedModel(turns=[tool_turn(("a", {}))]), journal)
    scope = CallScope(run_id=RUN, step=4, record_prompt=lambda s, h: seen.append((s, h)))
    with scoped(scope):
        await journaled.complete(envelope())
    assert seen == [(4, request_hash(envelope()))]


async def test_a_draft_request_is_journaled_but_never_recorded_for_rule_13(tmp_path: Path) -> None:
    """A draft model legitimately asks a different question; Rule 13 tracks the target only."""
    journal = Journal(tmp_path / "j.db")
    seen: list[tuple[int, str]] = []
    draft = JournaledModel(ScriptedModel(turns=[tool_turn(("a", {}))]), journal, role="draft")
    with scoped(CallScope(run_id=RUN, step=0, record_prompt=lambda s, h: seen.append((s, h)))):
        await draft.complete(envelope(model="draft-model"))
    assert seen == []
    assert [e.payload["role"] for e in journal.read(RUN)] == ["draft", "draft"]


async def test_streaming_journals_the_turn_before_turn_complete_escapes(tmp_path: Path) -> None:
    """Hard Rule 5: nothing downstream may see a turn that is not yet durable."""
    journal = Journal(tmp_path / "j.db")
    model = ScriptedModel(turns=[tool_turn(("a", {}), ("b", {}))])
    journaled = JournaledModel(model, journal)
    kinds_when_seen: list[tuple[str, list[str]]] = []
    with scoped(CallScope(run_id=RUN, step=0)):
        async for event in journaled.stream(envelope(stream=True)):
            label = type(event).__name__
            kinds_when_seen.append((label, [e.kind for e in journal.read(RUN)]))

    tool_events = [k for k, _ in kinds_when_seen if k == "ToolUseComplete"]
    assert len(tool_events) == 2, "each tool_use block must surface as it finishes parsing"
    at_turn_complete = next(k for label, k in kinds_when_seen if label == "TurnComplete")
    assert "model_response" in at_turn_complete, "the turn reached the caller before it was durable"


# -- replay -------------------------------------------------------------------------------------


async def test_replay_reproduces_every_response(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    turns = [
        tool_turn(("lookup_customer", {"customer_id": "c1"}), turn=0),
        tool_turn(("charge_card", {"customer_id": "c1", "amount": 10.0}), turn=1),
        free_text_turn("Done."),
    ]
    envelopes = [envelope(max_tokens=1024 + step) for step in range(3)]
    original = await _drive(journal, ScriptedModel(turns=turns), envelopes)

    replay = ReplayModel(journal=journal, run_id=RUN)
    assert replay.steps == (0, 1, 2)
    replayed = []
    for step, env in enumerate(envelopes):
        with scoped(CallScope(run_id=RUN, node_id="agent", step=step)):
            replayed.append(await replay.complete(env))
    assert [response_to_json(r) for r in replayed] == [response_to_json(r) for r in original]


async def test_replay_never_calls_a_model(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    model = ScriptedModel(turns=[tool_turn(("a", {}))])
    await _drive(journal, model, [envelope()])
    calls_before = model.calls

    replay = ReplayModel(journal=journal, run_id=RUN)
    with scoped(CallScope(run_id=RUN, node_id="agent", step=0)):
        await replay.complete(envelope())
    assert model.calls == calls_before


async def test_a_changed_request_diverges_at_that_step_and_writes_nothing_further(
    tmp_path: Path,
) -> None:
    """Spec task 0.4's Verify, and attack 7.8."""
    journal = Journal(tmp_path / "j.db")
    envelopes = [envelope(max_tokens=1024 + step) for step in range(3)]
    await _drive(
        journal, ScriptedModel(turns=[tool_turn(("a", {}), turn=t) for t in range(3)]), envelopes
    )
    entries_before = len(list(journal.read(RUN)))

    replay = ReplayModel(journal=journal, run_id=RUN)
    for step in (0, 1):
        with scoped(CallScope(run_id=RUN, node_id="agent", step=step)):
            await replay.complete(envelopes[step])

    tampered = replace(envelopes[2], system=(TextBlock(text="You help with support tickets!"),))
    with (
        scoped(CallScope(run_id=RUN, node_id="agent", step=2)),
        pytest.raises(ReplayDivergence) as caught,
    ):
        await replay.complete(tampered)

    assert caught.value.step == 2
    assert any("system" in note for note in caught.value.diff)
    assert len(list(journal.read(RUN))) == entries_before, "replay must not write to the journal"


async def test_a_changed_tool_list_diverges_at_the_first_step(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    await _drive(journal, ScriptedModel(turns=[tool_turn(("a", {}))]), [envelope()])

    extra = ToolDef(name="send_email", description="email", input_schema={"type": "object"})
    replay = ReplayModel(journal=journal, run_id=RUN)
    with (
        scoped(CallScope(run_id=RUN, node_id="agent", step=0)),
        pytest.raises(ReplayDivergence) as caught,
    ):
        await replay.complete(envelope(tools=(*envelope().tools, extra)))
    assert caught.value.step == 0
    assert any("send_email" in note for note in caught.value.diff)


async def test_replay_refuses_a_step_the_journal_does_not_have(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    await _drive(journal, ScriptedModel(turns=[tool_turn(("a", {}))]), [envelope()])
    replay = ReplayModel(journal=journal, run_id=RUN)
    with scoped(CallScope(run_id=RUN, node_id="agent", step=9)), pytest.raises(ReplayExhausted):
        await replay.complete(envelope())


async def test_replay_serves_a_node_only_the_turns_that_node_made(tmp_path: Path) -> None:
    """Keyed by node as well as position, because two nodes can make a call from one position.

    Running side by side, two nodes each make their first call from the same program position.
    An index keyed by position alone would hand one node's recorded reply to the other -- a
    replay that "succeeds" by answering the wrong question.
    """
    journal = Journal(tmp_path / "j.db")
    await _drive(journal, ScriptedModel(turns=[tool_turn(("a", {}))]), [envelope()])
    replay = ReplayModel(journal=journal, run_id=RUN)
    with scoped(CallScope(run_id=RUN, node_id="another", step=0)), pytest.raises(ReplayExhausted):
        await replay.complete(envelope())


async def test_replay_serves_several_turns_from_one_position_in_the_order_they_were_made(
    tmp_path: Path,
) -> None:
    """A node running a conversation makes several calls from one position; replay kept one.

    The index held a single turn per step, so every turn but the last was overwritten, and a
    replay of a multi-turn node diverged on its very first request.
    """
    journal = Journal(tmp_path / "j.db")
    first, second = envelope(), replace(envelope(), max_tokens=17)
    model = JournaledModel(
        ScriptedModel(turns=[tool_turn(("a", {})), tool_turn(("b", {}))]), journal
    )
    with scoped(CallScope(run_id=RUN, branch_id="br-canon", step=0, node_id="agent")):
        await model.complete(first)
        await model.complete(second)
    replay = ReplayModel(journal=journal, run_id=RUN)
    with scoped(CallScope(run_id=RUN, node_id="agent", step=0)):
        assert (await replay.complete(first)).tool_uses[0].name == "a"
        assert (await replay.complete(second)).tool_uses[0].name == "b"
        with pytest.raises(ReplayExhausted):
            await replay.complete(second)


async def test_replay_streams_the_journaled_blocks(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    journaled = JournaledModel(
        ScriptedModel(turns=[tool_turn(("a", {}), ("b", {}))]), journal, provider="scripted"
    )
    with scoped(CallScope(run_id=RUN, branch_id="br-canon", step=0, node_id="agent")):
        async for _event in journaled.stream(envelope()):
            pass
    replay = ReplayModel(journal=journal, run_id=RUN)
    events = []
    with scoped(CallScope(run_id=RUN, node_id="agent", step=0)):
        async for event in replay.stream(envelope()):
            events.append(event)
    assert sum(isinstance(e, ToolUseComplete) for e in events) == 2
    assert isinstance(events[-1], TurnComplete)


async def test_a_turn_handed_over_whole_is_replayed_whole(tmp_path: Path) -> None:
    """Asked with ``complete()``, the caller had none of the turn until all of it: a stream
    that replays it hands over no pieces either -- an early read issued on one would be a
    call the recorded node never made at that point."""
    journal = Journal(tmp_path / "j.db")
    await _drive(journal, ScriptedModel(turns=[tool_turn(("a", {}), ("b", {}))]), [envelope()])
    replay = ReplayModel(journal=journal, run_id=RUN)
    events = []
    with scoped(CallScope(run_id=RUN, node_id="agent", step=0)):
        async for event in replay.stream(envelope()):
            events.append(event)
    assert len(events) == 1 and isinstance(events[0], TurnComplete)
    assert len(events[0].response.tool_uses) == 2


async def test_a_squashed_branch_s_turn_is_never_served_back(tmp_path: Path) -> None:
    """The retired-chain rule: speculative output is evidence, never an input."""
    journal = Journal(tmp_path / "j.db")
    journaled = JournaledModel(ScriptedModel(turns=[tool_turn(("a", {}))]), journal)
    with scoped(CallScope(run_id=RUN, branch_id="br-doomed", step=0, speculative=True)):
        await journaled.complete(envelope())

    assert ReplayModel(journal=journal, run_id=RUN).steps == ()
    retired = ReplayModel(journal=journal, run_id=RUN, retired_branches=frozenset({"br-doomed"}))
    assert retired.steps == (0,)


async def test_a_replay_waits_as_long_as_the_question_took_to_write(tmp_path: Path) -> None:
    """A replay writes no question, and started its clock at once: its answers came back a
    write's time sooner than the run's, and a deadline inside that gap decided otherwise.
    Suspected by the eighteenth review."""
    import time

    class SlowToAsk(Journal):
        async def append_async(self, run_id: str, kind: str, payload: Any) -> int:
            if kind == "model_request":
                await asyncio.sleep(0.2)
            return await super().append_async(run_id, kind, payload)

    journal = SlowToAsk(tmp_path / "j.db")
    journaled = JournaledModel(ScriptedModel(turns=[tool_turn(("a", {}))]), journal)
    with scoped(CallScope(run_id=RUN, branch_id="br-canon", step=0, node_id="agent")):
        async for _event in journaled.stream(envelope()):
            pass
    (outcome,) = list(journal.read(RUN, kinds=["model_response"]))
    assert int(outcome.payload["ask_ms"]) >= 200, outcome.payload  # type: ignore[arg-type]

    replay = ReplayModel(journal=journal, run_id=RUN)
    began = time.monotonic()
    with scoped(CallScope(run_id=RUN, node_id="agent", step=0)):
        first = await replay.stream(envelope()).__anext__()
    assert isinstance(first, ToolUseComplete)
    assert time.monotonic() - began >= 0.2
