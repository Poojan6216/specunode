"""A resumed run is served the model turns whose decisions may already have sent something.

A resumed node asks the model what it asked before the crash. Where the dead attempt's answer
may have sent something, asking again risked a different answer: a different call at the same
position derives a different idempotency key, and the dedupe table cannot connect it to what
already went out. Where the answer sent nothing, there is nothing to protect, and asking again
lets a run recover from an answer that failed.

Each test records a turn with one model and resumes with another that would answer differently,
so a turn that was asked again rather than served shows up as a different decision.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from specunode.core.decision import ToolCall
from specunode.core.model import (
    CallScope,
    JournaledModel,
    Message,
    ModelResponse,
    RequestEnvelope,
    TextBlock,
    ToolUseComplete,
    TurnComplete,
    decisions_of,
    scoped,
)
from specunode.journal.journal import Journal, PendingClaim
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


def model(*turns: tuple[str, dict[str, object]]) -> ScriptedModel:
    return ScriptedModel(turns=[tool_turn(turn, turn=index) for index, turn in enumerate(turns)])


def calls(response: ModelResponse) -> list[ToolCall]:
    return [d for d in decisions_of(response) if isinstance(d, ToolCall)]


async def record(
    journal: Journal, branch: str, *questions: tuple[str, tuple[str, dict[str, object]]]
) -> None:
    """``branch`` asks each question and is given its answer, as a process before the crash."""
    first = JournaledModel(model(*(answer for _, answer in questions)), journal)
    with scoped(scope(branch)):
        for question, _ in questions:
            await first.complete(ask(question))


async def sent(journal: Journal, branch: str, name: str = "") -> None:
    """``branch`` staged an effect and it reached the world, as the drain records one."""
    effect_id = f"effect-{branch}{name}"
    await journal.append_async(
        RUN,
        "effect_staged",
        {
            "v": 1,
            "effect_id": effect_id,
            "branch_id": branch,
            "step": 2,
            "tool": "charge_card",
            "args_hash": "h",
            "key": f"key-{effect_id}",
            "nkey": f"nkey-{effect_id}",
            "stage_index": 0,
        },
    )
    await journal.append_async(
        RUN,
        "effect_dispatched",
        {
            "v": 1,
            "effect_id": effect_id,
            "branch_id": branch,
            "nkey": f"nkey-{effect_id}",
            "stage_index": 0,
            "dispatch_index": 0,
            "authorised_by_offset": 0,
            "deduped": False,
        },
    )


async def dead_lettered(journal: Journal, branch: str, *, left: str) -> None:
    await journal.append_async(
        RUN,
        "effect_dead_lettered",
        {
            "v": 1,
            "effect_id": f"effect-{branch}",
            "branch_id": branch,
            "nkey": f"nkey-{branch}",
            "attempts": 1,
            "last_error": {"type": "ToolDispatchError", "message": "no such tool"},
            "sent": left,
        },
    )


def resumed(journal: Journal, changed: ScriptedModel) -> JournaledModel:
    target = JournaledModel(changed, journal)
    target.serve_recorded(RUN, RecordedTurns(journal, RUN))
    return target


async def test_a_decision_that_sent_something_is_served_rather_than_asked(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25))
    await sent(journal, "DEAD")

    changed = model(CHARGE_30)
    with scoped(scope("RESUMED")):
        again = await resumed(journal, changed).complete(ask("bill cus-1"))

    assert calls(again) == [ToolCall(*CHARGE_25)], "the resumed node decided something else"
    assert changed.calls == 0, "the model was asked for a decision that had sent something"
    replies = list(journal.read(RUN, kinds=["model_response"]))
    assert replies[1].payload["recorded_from"] == replies[0].offset
    assert replies[1].payload["branch_id"] == "RESUMED", "a served turn belongs to its branch"
    requests = list(journal.read(RUN, kinds=["model_request"]))
    assert requests[1].payload["recorded_from"] == replies[0].offset


async def test_a_decision_that_sent_nothing_is_asked_again(tmp_path: Path) -> None:
    """Nothing went out on it, so nothing needs protecting from a different answer."""
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25))

    changed = model(CHARGE_30)
    with scoped(scope("RESUMED")):
        again = await resumed(journal, changed).complete(ask("bill cus-1"))
    assert changed.calls == 1
    assert calls(again) == [ToolCall(*CHARGE_30)]


async def test_a_dead_letter_that_never_left_does_not_pin_the_decision(tmp_path: Path) -> None:
    """The model named a tool that does not exist; the call failed before it left the process.

    Serving that answer again would retry the same failed call on every resume, with the model
    never asked: a run stuck until its code or prompt changed. Nothing left, so it is asked.
    """
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", ("chrage_card", {"amount": 25.0})))
    await dead_lettered(journal, "DEAD", left="no")

    changed = model(CHARGE_25)
    with scoped(scope("RESUMED")):
        again = await resumed(journal, changed).complete(ask("bill cus-1"))
    assert changed.calls == 1
    assert calls(again) == [ToolCall(*CHARGE_25)]


async def test_a_dead_letter_that_may_have_left_pins_the_decision(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25))
    await dead_lettered(journal, "DEAD", left="maybe")

    changed = model(CHARGE_30)
    with scoped(scope("RESUMED")):
        again = await resumed(journal, changed).complete(ask("bill cus-1"))
    assert changed.calls == 0
    assert calls(again) == [ToolCall(*CHARGE_25)]


async def test_a_claim_with_no_outcome_pins_the_decision_unless_it_never_left(
    tmp_path: Path,
) -> None:
    """The process died after claiming the charge: sent, or maybe not -- unless marked unsent."""
    for marked_unsent, served in ((False, True), (True, False)):
        journal = Journal(tmp_path / f"claim-{marked_unsent}.db")
        await record(journal, "DEAD", ("bill cus-1", CHARGE_25))
        claim = PendingClaim(
            run_id=RUN,
            nkey="nkey-charge",
            idem_key="key-charge",
            effect_id="effect-charge",
            branch_id="DEAD",
            tool="charge_card",
        )
        await journal.claim_dispatch(claim)
        if marked_unsent:
            await journal.mark_not_sent(RUN, "nkey-charge", 1)

        changed = model(CHARGE_30)
        with scoped(scope("RESUMED")):
            again = await resumed(journal, changed).complete(ask("bill cus-1"))
        assert (changed.calls == 0) is served, (marked_unsent, changed.calls)
        assert calls(again) == [ToolCall(*(CHARGE_25 if served else CHARGE_30))]


async def test_a_changed_question_is_asked_and_the_rest_of_that_conversation_too(
    tmp_path: Path,
) -> None:
    """A recorded answer answers the question it was given. From the first one that differs,
    the node and position go back to the model -- including later turns that do match, which
    belong to a conversation that has already gone another way."""
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25), ("and the receipt", RECEIPT))
    await sent(journal, "DEAD")

    changed = model(CHARGE_30, RECEIPT)
    target = resumed(journal, changed)
    with scoped(scope("RESUMED")):
        different = await target.complete(ask("bill cus-1, who now owes 30"))
        await target.complete(ask("and the receipt"))

    assert calls(different) == [ToolCall(*CHARGE_30)]
    assert changed.calls == 2, "a turn was served in answer to a question it was not given"


async def test_after_a_question_changed_the_latest_attempt_to_ask_it_answers(
    tmp_path: Path,
) -> None:
    """Two crashes. The first attempt charged 25 and notified; the price then changed, so the
    second attempt's question was new, and it charged 30 before dying. The third asks the
    second's question. It must be served the second's answer: the charge that went out under
    the question being asked now is 30, and a fresh answer would be a third charge.

    Keeping the attempt with the most turns -- the first -- matched nothing, asked the model,
    and charged again.
    """
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "FIRST", ("the price is 10", CHARGE_25), ("notify", RECEIPT))
    await sent(journal, "FIRST")
    await record(journal, "SECOND", ("the price is 12", CHARGE_30))
    await sent(journal, "SECOND")

    changed = model(("charge_card", {"customer_id": "cus-1", "amount": 35.0}))
    with scoped(scope("THIRD")):
        again = await resumed(journal, changed).complete(ask("the price is 12"))
    assert changed.calls == 0
    assert calls(again) == [ToolCall(*CHARGE_30)]


async def test_a_streamed_turn_is_served_as_a_stream(tmp_path: Path) -> None:
    """Early issue reads a served turn's blocks as it read the original's."""
    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE_25, RECEIPT, turn=0)]), journal)
    with scoped(scope("DEAD")):
        recorded = [event async for event in first.stream(ask("bill cus-1"))]
    await sent(journal, "DEAD")

    changed = model(CHARGE_30)
    with scoped(scope("RESUMED")):
        served = [event async for event in resumed(journal, changed).stream(ask("bill cus-1"))]

    assert changed.calls == 0
    blocks = [e.block for e in served if isinstance(e, ToolUseComplete)]
    assert blocks == [e.block for e in recorded if isinstance(e, ToolUseComplete)]
    assert isinstance(served[-1], TurnComplete)
    assert calls(served[-1].response) == [ToolCall(*CHARGE_25), ToolCall(*RECEIPT)]


async def test_a_guess_is_never_served(tmp_path: Path) -> None:
    """A speculative branch's model call is a guess, not a decision the run made."""
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25))
    await sent(journal, "DEAD")

    changed = model(CHARGE_30)
    with scoped(scope("GUESS", speculative=True)):
        await resumed(journal, changed).complete(ask("bill cus-1"))
    assert changed.calls == 1


async def test_serving_is_per_run_on_a_shared_model(tmp_path: Path) -> None:
    """One model object serves several runs. A fresh run, or another run finishing its resume,
    must neither be served this run's turns nor switch off this run's serving."""
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25))
    await sent(journal, "DEAD")

    changed = model(CHARGE_30, CHARGE_30)
    shared = resumed(journal, changed)
    shared.serve_recorded("01ANOTHERRUNAAAAAAAAAAAAAA", None)  # another run's resume ending
    with scoped(scope("FRESH", run_id="01ANOTHERRUNAAAAAAAAAAAAAA")):
        fresh = await shared.complete(ask("bill cus-1"))
    with scoped(scope("RESUMED")):
        again = await shared.complete(ask("bill cus-1"))
    assert calls(fresh) == [ToolCall(*CHARGE_30)], "a fresh run was served another run's turn"
    assert calls(again) == [ToolCall(*CHARGE_25)], "the resume stopped being served"
    assert changed.calls == 1


async def test_a_second_resume_is_served_the_whole_conversation(tmp_path: Path) -> None:
    """The first resume was served one turn and died having sent nothing more; the second must
    still be served both of the dead attempt's turns."""
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25), ("and the receipt", RECEIPT))
    await sent(journal, "DEAD")

    with scoped(scope("RESUMED-1")):
        await resumed(journal, model(CHARGE_30)).complete(ask("bill cus-1"))

    changed = model(CHARGE_30, RECEIPT)
    target = resumed(journal, changed)
    with scoped(scope("RESUMED-2")):
        charge = await target.complete(ask("bill cus-1"))
        receipt = await target.complete(ask("and the receipt"))

    assert changed.calls == 0, "the second resume asked for a turn the journal held"
    assert calls(charge) == [ToolCall(*CHARGE_25)]
    assert calls(receipt) == [ToolCall(*RECEIPT)]
    replies = list(journal.read(RUN, kinds=["model_response"]))
    origins = [entry.payload.get("recorded_from") for entry in replies]
    # Asked, asked; the first resume served the first turn; the second resume served both, each
    # naming the entry it was first recorded in.
    first_turn, second_turn = replies[0].offset, replies[1].offset
    assert origins == [None, None, first_turn, first_turn, second_turn]


async def test_only_the_turns_up_to_the_last_send_are_served(tmp_path: Path) -> None:
    """A conversation: the first answer charged, and the charge went out; the second named a
    tool that does not exist, and nothing went out on it. The first is served -- the charge is
    out -- and the second is asked again, so the run can recover from it. Serving the whole
    attempt pinned the bad answer to every resume."""
    journal = Journal(tmp_path / "journal.db")
    await record(journal, "DEAD", ("bill cus-1", CHARGE_25))
    await sent(journal, "DEAD")
    await record(journal, "DEAD", ("now tell them", ("notfy", {"customer_id": "cus-1"})))

    changed = model(("notify", {"customer_id": "cus-1"}))
    target = resumed(journal, changed)
    with scoped(scope("RESUMED")):
        charge = await target.complete(ask("bill cus-1"))
        notify = await target.complete(ask("now tell them"))
    assert calls(charge) == [ToolCall(*CHARGE_25)]
    assert calls(notify) == [ToolCall("notify", {"customer_id": "cus-1"})]
    assert changed.calls == 1


async def test_calls_made_at_once_are_matched_in_the_order_they_were_asked(
    tmp_path: Path,
) -> None:
    """Two questions asked together; the second's answer came back first. The resume asks them
    in the same order, and each is served its own answer. Matched in the order the answers came
    back, neither was served."""

    class ByQuestion:
        def __init__(self, answers: dict[str, tuple[float, ModelResponse]]) -> None:
            self.answers, self.calls = answers, 0

        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            self.calls += 1
            block = envelope.messages[-1].content[0]
            assert isinstance(block, TextBlock)
            delay, answer = self.answers[block.text]
            await asyncio.sleep(delay)
            return answer

        def stream(self, envelope: RequestEnvelope) -> object:
            raise NotImplementedError

    journal = Journal(tmp_path / "journal.db")
    first = JournaledModel(
        ByQuestion(
            {
                "bill cus-1": (0.05, tool_turn(CHARGE_25, turn=0)),
                "tell cus-1": (0.0, tool_turn(RECEIPT, turn=1)),
            }
        ),  # type: ignore[arg-type]
        journal,
    )
    with scoped(scope("DEAD")):
        await asyncio.gather(first.complete(ask("bill cus-1")), first.complete(ask("tell cus-1")))
    await sent(journal, "DEAD")
    replies = [e.payload["decision"] for e in journal.read(RUN, kinds=["model_response"])]
    assert replies[0]["name"] == "send_receipt", "the answers did not come back out of order"

    changed = ByQuestion(
        {"bill cus-1": (0.0, tool_turn(CHARGE_30, turn=0)), "tell cus-1": (0.0, tool_turn(RECEIPT))}
    )
    target = JournaledModel(changed, journal)  # type: ignore[arg-type]
    target.serve_recorded(RUN, RecordedTurns(journal, RUN))
    with scoped(scope("RESUMED")):
        charge, _receipt = await asyncio.gather(
            target.complete(ask("bill cus-1")), target.complete(ask("tell cus-1"))
        )
    assert changed.calls == 0
    assert calls(charge) == [ToolCall(*CHARGE_25)]


# -- replay: one outcome per question, in the order asked, and no wait without end --------------


async def journal_two_questions(journal: Journal, *, a_cancelled: bool) -> None:
    """Questions A then B, asked at once; B answered first, then A -- and, if ``a_cancelled``,
    A's caller had stopped waiting while its answer was written, so a marker follows B's."""
    from specunode.core.model import CANCELLED, request_hash

    writer = JournaledModel(ScriptedModel(turns=[]), journal)
    here = scope("br-1")
    a, b = ask("question A"), ask("question B")
    request_a = await writer._journal_request(a, here, request_hash(a), None)
    request_b = await writer._journal_request(b, here, request_hash(b), None)
    answer_a, answer_b = tool_turn(CHARGE_25), tool_turn(CHARGE_30)
    if a_cancelled:
        await writer._journal_response(answer_a, here, request_a, request_hash(a), 50)
        await writer._journal_response(answer_b, here, request_b, request_hash(b), 5)
        await writer._journal_response(
            answer_a, here, request_a, request_hash(a), 50, failed=CANCELLED, cancelled=True
        )
    else:
        await writer._journal_response(answer_b, here, request_b, request_hash(b), 5)
        await writer._journal_response(answer_a, here, request_a, request_hash(a), 50)


async def test_replay_matches_questions_in_the_order_they_were_asked(tmp_path: Path) -> None:
    """Answered the other way round, two questions asked at once were each matched with the
    other's answer, and the replay of a faithful run was refused. Found by the fourteenth
    review."""
    from specunode.journal.replay import ReplayModel

    journal = Journal(tmp_path / "journal.db")
    await journal_two_questions(journal, a_cancelled=False)
    replay = ReplayModel(journal=journal, run_id=RUN)
    with scoped(scope("br-2")):
        first = await replay.complete(ask("question A"))
        second = await replay.complete(ask("question B"))
    assert calls(first) == [ToolCall(*CHARGE_25)] and calls(second) == [ToolCall(*CHARGE_30)]


async def test_replay_never_hands_over_an_answer_its_node_never_saw(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A's answer, B's answer, then the marker that A's node had stopped waiting: the marker was
    merged only when it sat next to the answer, so replay handed A's answer over -- and
    ``replay --dispatch`` sent a charge the run never made. Nor does it wait without end for a
    question nothing asks it to stop waiting on. Found by the fourteenth review."""
    import pytest

    from specunode.core import model as model_module
    from specunode.core.model import TurnAbandoned
    from specunode.journal.replay import ReplayModel

    monkeypatch.setattr(model_module, "_ABANDON_MARGIN_S", 0.05)  # type: ignore[attr-defined]
    journal = Journal(tmp_path / "journal.db")
    await journal_two_questions(journal, a_cancelled=True)
    replay = ReplayModel(journal=journal, run_id=RUN)
    with scoped(scope("br-2")):
        with pytest.raises(TurnAbandoned):
            await replay.complete(ask("question A"))
        answer_b = await replay.complete(ask("question B"))
    assert calls(answer_b) == [ToolCall(*CHARGE_30)]


async def test_a_turn_cancelled_again_and_again_is_still_recorded(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A third cancel while the "cancelled" outcome was being written cancelled the write
    itself, and the question was left with no outcome. Found by the fourteenth review."""
    import contextlib

    journal = Journal(tmp_path / "journal.db")
    real = JournaledModel._journal_response

    async def slow(self: JournaledModel, *args: object, **kwargs: object) -> None:
        await asyncio.sleep(0.2)  # the writer is busy
        await real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(JournaledModel, "_journal_response", slow)  # type: ignore[attr-defined]

    class Hangs:
        async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    model_ = JournaledModel(Hangs(), journal)  # type: ignore[arg-type]

    async def asking() -> None:
        with scoped(scope("br-1")):
            await model_.complete(ask("question A"))

    task = asyncio.create_task(asking())
    for _ in range(1000):  # the question is on disk and the model is being asked
        if list(journal.read(RUN, kinds=["model_request"])):
            break
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0.02)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    outcomes = [e.payload for e in journal.read(RUN, kinds=["model_response"])]
    assert outcomes and outcomes[-1].get("cancelled") is True, outcomes


async def test_a_turn_cancelled_while_its_question_is_written_has_an_outcome(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The caller stopped waiting while the question itself was being written: the write
    finished anyway, and a question on disk with no outcome sent every later question at its
    position to the live model on resume. Found by the slow-disk run."""
    import contextlib

    journal = Journal(tmp_path / "journal.db")
    real = JournaledModel._journal_request

    async def slow(self: JournaledModel, *args: object, **kwargs: object) -> str:
        await asyncio.sleep(0.2)
        return await real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(JournaledModel, "_journal_request", slow)  # type: ignore[attr-defined]
    model_ = JournaledModel(ScriptedModel(turns=[tool_turn(CHARGE_25)]), journal)

    async def asking() -> None:
        with scoped(scope("br-1")):
            await model_.complete(ask("question A"))

    task = asyncio.create_task(asking())
    await asyncio.sleep(0.05)  # the question is being written
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    requests = list(journal.read(RUN, kinds=["model_request"]))
    outcomes = [e.payload for e in journal.read(RUN, kinds=["model_response"])]
    assert len(requests) == 1 and outcomes and outcomes[-1].get("cancelled") is True, outcomes
