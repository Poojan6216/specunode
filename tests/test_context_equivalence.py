"""THE CONTEXT-EQUIVALENCE TEST (spec task 5.5, Hard Rule 13). Mandatory. Never skipped.

Hard Rule 9 proves the effect plumbing on journaled outputs. This proves something Rule 9
cannot: that **the model was never asked a different question**. A speculative branch may send
a model request only if the prompt it would send is one the sequential run could send -- no
placeholder anywhere in it, tool results assembled in the order the model asked for them, and
no message from a branch whose output may not be read back.

**These tests assert on what a model RECEIVED, never on what the runtime says it sent.** The
live check and the retirement-time rebuild share a prompt builder, so they can be wrong in the
same way and still agree with each other: an implementation that hashes a clean envelope while
a dirty one goes to the wire reports zero divergences forever and passes any test written
against its own bookkeeping. ``RecordingModel`` sits at the wire and keeps the projection of
every request, and that is what is compared.

It runs over every workload in ``bench/workloads/``, at tier 0 and tier 1. That matters most
for ``ops_agent``, which is the one workload that hands its turn to the runtime and therefore
the only one where a speculative branch exists to send a prompt at all -- on the others the
comparison is still made, and still has to hold, but it is the easy direction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bench.workloads import WORKLOADS, Workload

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.branch import Branch, BranchStatus, ResultSlot, SlotStatus, TurnFrame
from specunode.core.hazards import Hazard, analyse_model_request, handle_for
from specunode.core.model import (
    ContextDivergence,
    JournaledModel,
    Message,
    PromptBuilder,
    RequestEnvelope,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    check_structural,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import RecordingModel
from specunode.testing.world import World, standard_world

ATTESTED = frozenset({"br-1", "br-2"})
TIERS = ("t0", "t1")


def _cases() -> list[object]:
    return [
        pytest.param(w, tier, id=f"{w.name}-{tier}") for w in WORKLOADS for tier in TIERS
    ]


def run_with(
    tmp_path: Path,
    workload: Workload,
    world: World,
    *,
    speculation: bool,
    tier: str = "t0",
    db: str,
) -> tuple[Scheduler, RecordingModel, str]:
    adapter, registry = workload.make(world)
    journal = Journal(tmp_path / db)
    recorder = RecordingModel(workload.model())
    predictor = None
    if speculation and tier == "t1":
        index = PatternIndex(order=2)
        index.train([workload.decisions()])
        predictor = PatternDrafter(index=index)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(recorder, journal, provider="scripted"),
        policy=Policy(speculation=speculation),
        predictor=predictor,
    )
    return scheduler, recorder, new_ulid()


@pytest.mark.parametrize(("workload", "tier"), _cases())
async def test_the_speculative_arm_asks_the_same_questions_as_the_sequential_one(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    """The claim, asserted at the wire on both arms."""
    sequential, seq_recorder, seq_run = run_with(
        tmp_path,
        workload,
        standard_world(),
        speculation=False,
        db=f"{workload.name}-{tier}-seq.db",
    )
    speculative, spec_recorder, spec_run = run_with(
        tmp_path,
        workload,
        standard_world(),
        speculation=True,
        tier=tier,
        db=f"{workload.name}-{tier}-spec.db",
    )

    seq_result = await sequential.run(seq_run, dict(workload.seed))
    spec_result = await speculative.run(spec_run, dict(workload.seed))
    assert seq_result.ok and spec_result.ok

    assert seq_recorder.digests, "the run asked the model nothing, so this proves nothing"
    assert spec_recorder.digests == seq_recorder.digests, (
        "the speculative arm asked the model a different question than the sequential arm"
    )


@pytest.mark.parametrize(("workload", "tier"), _cases())
async def test_no_request_ever_carried_a_placeholder(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    """A model conditioned on a placeholder is deciding on a premise no real run had."""
    scheduler, recorder, run_id = run_with(
        tmp_path,
        workload,
        standard_world(),
        speculation=True,
        tier=tier,
        db=f"{workload.name}-{tier}-handles.db",
    )
    await scheduler.run(run_id, dict(workload.seed))
    assert recorder.calls
    for call in recorder.calls:
        assert "$specunode.handle:" not in str(call.projection)


# -- the structural check, and its planted bugs -------------------------------------------------


def _envelope(*messages: Message) -> RequestEnvelope:
    return RequestEnvelope(model="scripted", messages=messages, max_tokens=64)


def test_a_placeholder_anywhere_in_a_request_is_a_divergence() -> None:
    envelope = _envelope(
        Message(role="user", content=(TextBlock(text=f"use {handle_for('01ABC')}"),))
    )
    problems = check_structural(envelope, ATTESTED)
    assert any("placeholder" in problem for problem in problems)


def test_a_message_from_an_unattested_branch_is_a_divergence() -> None:
    """The Hard Rule 6 channel: a sibling that has not resolved yet is not attested either."""
    envelope = _envelope(
        Message(role="user", content=(TextBlock(text="hi"),), origin_branch="br-sibling")
    )
    problems = check_structural(envelope, ATTESTED)
    assert any("br-sibling" in problem for problem in problems)


def test_results_in_completion_order_rather_than_program_order_are_a_divergence() -> None:
    """The bug task 3.8 plants: the model asked for these in one order and got another."""
    assistant = Message(
        role="assistant",
        content=(
            ToolUseBlock(id="u1", name="a", args={}),
            ToolUseBlock(id="u2", name="b", args={}),
        ),
        origin_branch="br-1",
    )
    in_order = Message(
        role="user",
        content=(
            ToolResultBlock(tool_use_id="u1", content=1),
            ToolResultBlock(tool_use_id="u2", content=2),
        ),
        origin_branch="br-1",
    )
    reversed_order = Message(
        role="user",
        content=(
            ToolResultBlock(tool_use_id="u2", content=2),
            ToolResultBlock(tool_use_id="u1", content=1),
        ),
        origin_branch="br-1",
    )
    assert check_structural(_envelope(assistant, in_order), ATTESTED) == []
    problems = check_structural(_envelope(assistant, reversed_order), ATTESTED)
    assert any("program order" in problem for problem in problems)


def test_the_builder_refuses_to_send_a_diverging_request() -> None:
    """The check runs before the hash and before anything leaves, not after the fact."""
    builder = PromptBuilder(base=RequestEnvelope(model="scripted", max_tokens=64))
    with pytest.raises(ContextDivergence):
        builder.build(
            [Message(role="user", content=(TextBlock(text=handle_for("01ABC")),))],
            attested=ATTESTED,
        )


def test_a_clean_request_builds_and_records_no_injection() -> None:
    builder = PromptBuilder(base=RequestEnvelope(model="scripted", max_tokens=64))
    built = builder.build(
        [Message(role="user", content=(TextBlock(text="hello"),))], attested=ATTESTED
    )
    assert built.request_hash
    assert built.injected_index == ()


def test_material_the_runtime_did_not_derive_is_declared_not_hidden() -> None:
    """Correction C10: without this, Rule 13 false-positives on every ordinary LangGraph app.

    A node that prepends a system message assembled from graph state produces a prompt the
    runtime cannot rebuild from the journal. Declaring it keeps the derivable part under an
    exact comparison and counts the rest, instead of degrading the whole check.
    """
    builder = PromptBuilder(base=RequestEnvelope(model="scripted", max_tokens=64))
    builder.inject(0, TextBlock(text="a document retrieved from state"))
    built = builder.build(
        [Message(role="user", content=(TextBlock(text="hello"),))], attested=ATTESTED
    )
    assert built.injected_index == (0,)
    assert built.injected_hash


# -- the write barrier ----------------------------------------------------------------------------


def test_a_branch_holding_a_staged_slot_may_not_call_the_model() -> None:
    """A staged slot never fills, so no honest turn can include it."""
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    frame = TurnFrame(step=1, slots=[ResultSlot(ordinal=0, tool_use_id="u1")])
    frame.slots[0].status = SlotStatus.STAGED
    branch.frames.append(frame)
    assert analyse_model_request(branch, b"{}", Policy()) is Hazard.MODEL_TURN_AFTER_STAGED_WRITE


def test_the_barrier_holds_for_a_later_turn_too_not_just_the_next_one() -> None:
    """The two-turn variant. A flag reading 'the next call' would pass this and be wrong.

    The message list is append-only and a branch cannot drain before it retires, so a write
    staged at turn one leaves its unfillable slot in every later turn's request.
    """
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    staged = TurnFrame(step=1, slots=[ResultSlot(ordinal=0, tool_use_id="u1")])
    staged.slots[0].status = SlotStatus.STAGED
    branch.frames.append(staged)
    branch.frames.append(TurnFrame(step=2, slots=[], complete=True))
    branch.frames.append(TurnFrame(step=3, slots=[], complete=True))
    assert analyse_model_request(branch, b"{}", Policy()) is Hazard.MODEL_TURN_AFTER_STAGED_WRITE


def test_a_branch_with_only_filled_slots_may_call_the_model() -> None:
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    frame = TurnFrame(step=1, slots=[ResultSlot(ordinal=0, tool_use_id="u1")])
    frame.slots[0].status = SlotStatus.FILLED
    branch.frames.append(frame)
    assert analyse_model_request(branch, b"{}", Policy()) is None
