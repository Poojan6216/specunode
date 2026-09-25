"""A resumed run is served the model turns the journal holds, rather than asking again.

A resumed node asks the model what it asked before the crash. Asking again was at best a wasted
call and at worst a different answer: a different call at the same position derives a different
idempotency key, and the dedupe table cannot connect it to what the dead process already sent.
Each test here records a turn with one model and resumes with another that would answer
differently, so a turn that was asked again rather than served shows up as a different decision.
"""

from __future__ import annotations

from pathlib import Path

from specunode.core.decision import ToolCall
from specunode.core.model import (
    CallScope,
    JournaledModel,
    Message,
    RequestEnvelope,
    TextBlock,
    ToolUseComplete,
    TurnComplete,
    decisions_of,
    scoped,
)
from specunode.journal.journal import Journal
from specunode.journal.replay import RecordedTurns
from specunode.testing.models import ScriptedModel, tool_turn

RUN = "01RECORDEDTURNSAAAAAAAAAAA"
CHARGE_25 = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})
CHARGE_30 = ("charge_card", {"customer_id": "cus-1", "amount": 30.0})
RECEIPT = ("send_receipt", {"customer_id": "cus-1"})


def ask(text: str) -> RequestEnvelope:
    return RequestEnvelope(
        model="scripted",
        messages=(Message(role="user", content=(TextBlock(text=text),)),),
        max_tokens=64,
    )


def scope(branch: str, *, run_id: str = RUN, speculative: bool = False) -> CallScope:
    return CallScope(
        run_id=run_id, branch_id=branch, step=1, node_id="decide#0", speculative=speculative
    )


def resumed_with(*turns: tuple[str, dict[str, object]]) -> ScriptedModel:
    """A model that would answer ``turns`` if it were asked -- which it should not be."""
    return ScriptedModel(turns=[tool_turn(turn, turn=index) for index, turn in enumerate(turns)])


def calls(response: object) -> list[ToolCall]:
    return [d for d in decisions_of(response) if isinstance(d, ToolCall)]  # type: ignore[arg-type]


async def test_a_turn_the_journal_holds_is_served_rather_than_asked(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE_25, turn=0)]), journal)
    with scoped(scope("DEAD")):
        decided = await first.complete(ask("bill cus-1"))

    changed = resumed_with(CHARGE_30)
    resumed = JournaledModel(changed, journal)
    resumed.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("RESUMED")):
        again = await resumed.complete(ask("bill cus-1"))

    assert calls(again) == calls(decided), "the resumed node decided something else"
    assert changed.calls == 0, "the model was asked a question the journal had answered"
    replies = list(journal.read(RUN, kinds=["model_response"]))
    assert len(replies) == 2
    served = replies[1].payload
    assert served["recorded_from"] == replies[0].offset
    assert served["branch_id"] == "RESUMED", "a served turn belongs to the branch it served"
    requests = list(journal.read(RUN, kinds=["model_request"]))
    assert requests[1].payload["recorded_from"] == replies[0].offset


async def test_a_changed_question_is_asked_and_the_rest_of_that_conversation_too(
    tmp_path: Path,
) -> None:
    """A recorded answer answers the question it was given. From the first one that differs,
    the node and position go back to the model -- including later turns that do match, which
    belong to a conversation that has already gone another way."""
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(
        ScriptedModel(turns=[tool_turn(CHARGE_25, turn=0), tool_turn(RECEIPT, turn=1)]), journal
    )
    with scoped(scope("DEAD")):
        await first.complete(ask("bill cus-1"))
        await first.complete(ask("and the receipt"))

    changed = resumed_with(CHARGE_30, RECEIPT)
    resumed = JournaledModel(changed, journal)
    resumed.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("RESUMED")):
        different = await resumed.complete(ask("bill cus-1, who now owes 30"))
        await resumed.complete(ask("and the receipt"))

    assert calls(different) == [ToolCall(*CHARGE_30)]
    assert changed.calls == 2, "a turn was served in answer to a question it was not given"
    replies = [entry.payload for entry in journal.read(RUN, kinds=["model_response"])]
    assert not any("recorded_from" in reply for reply in replies)


async def test_a_streamed_turn_is_served_as_a_stream(tmp_path: Path) -> None:
    """Early issue reads a served turn's blocks as it read the original's."""
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE_25, RECEIPT, turn=0)]), journal)
    with scoped(scope("DEAD")):
        recorded = [event async for event in first.stream(ask("bill cus-1"))]

    changed = resumed_with(CHARGE_30)
    resumed = JournaledModel(changed, journal)
    resumed.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("RESUMED")):
        served = [event async for event in resumed.stream(ask("bill cus-1"))]

    assert changed.calls == 0
    blocks = [e.block for e in served if isinstance(e, ToolUseComplete)]
    assert blocks == [e.block for e in recorded if isinstance(e, ToolUseComplete)]
    assert isinstance(served[-1], TurnComplete)
    assert calls(served[-1].response) == [ToolCall(*CHARGE_25), ToolCall(*RECEIPT)]
    replies = list(journal.read(RUN, kinds=["model_response"]))
    assert replies[1].payload["recorded_from"] == replies[0].offset


async def test_a_guess_is_never_served(tmp_path: Path) -> None:
    """A speculative branch's model call is a guess, not a decision the run made."""
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE_25, turn=0)]), journal)
    with scoped(scope("DEAD")):
        await first.complete(ask("bill cus-1"))

    changed = resumed_with(CHARGE_30)
    resumed = JournaledModel(changed, journal)
    resumed.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("GUESS", speculative=True)):
        await resumed.complete(ask("bill cus-1"))
    assert changed.calls == 1


async def test_another_run_is_not_served_this_ones_turns(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE_25, turn=0)]), journal)
    with scoped(scope("DEAD")):
        await first.complete(ask("bill cus-1"))

    changed = resumed_with(CHARGE_30)
    resumed = JournaledModel(changed, journal)
    resumed.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("FRESH", run_id="01ANOTHERRUNAAAAAAAAAAAAAA")):
        await resumed.complete(ask("bill cus-1"))
    assert changed.calls == 1


async def test_a_second_resume_is_served_the_whole_conversation(tmp_path: Path) -> None:
    """The first resume was served one turn and died; the second must still be served both.

    Each attempt journals the turns it is served, so the journal holds the dead attempt's two
    turns and the first resume's one. Keeping the latest attempt -- which is what replay does --
    would serve the one and ask the model for the other.
    """
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(
        ScriptedModel(turns=[tool_turn(CHARGE_25, turn=0), tool_turn(RECEIPT, turn=1)]), journal
    )
    with scoped(scope("DEAD")):
        await first.complete(ask("bill cus-1"))
        await first.complete(ask("and the receipt"))

    died_again = JournaledModel(resumed_with(CHARGE_30), journal)
    died_again.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("RESUMED-1")):
        await died_again.complete(ask("bill cus-1"))

    changed = resumed_with(CHARGE_30, RECEIPT)
    resumed = JournaledModel(changed, journal)
    resumed.serve_recorded(RecordedTurns(journal, RUN))
    with scoped(scope("RESUMED-2")):
        charge = await resumed.complete(ask("bill cus-1"))
        receipt = await resumed.complete(ask("and the receipt"))

    assert changed.calls == 0, "the second resume asked for a turn the journal held"
    assert calls(charge) == [ToolCall(*CHARGE_25)]
    assert calls(receipt) == [ToolCall(*RECEIPT)]
    replies = list(journal.read(RUN, kinds=["model_response"]))
    origins = [entry.payload.get("recorded_from") for entry in replies]
    # Asked, asked; the first resume served the first turn; the second resume served both, each
    # naming the entry it was first recorded in rather than the copy it may have been read from.
    first_turn, second_turn = replies[0].offset, replies[1].offset
    assert origins == [None, None, first_turn, first_turn, second_turn]
