"""Hard Rule 10 -- "wasted work is bounded and measured" -- as behaviour, not as a docstring.

Every one of these was dead before this file existed. ``Budget.record_speculative_read`` had no
caller, so reads were never counted. ``Budget.inflight_branches`` was never incremented, so the
inflight limit could not be reached. ``record_resolution`` was always passed ``tokens=0``, and
``Prediction`` had no cost field, so ``max_wasted_tokens`` could not be reached either.
``Budget.disable`` had no caller and no ``policy_event`` was ever journaled, so the half of the
rule that says "**and the ledger says so**" never happened -- and the ledger read alpha from
those missing events, so it printed ``n/a`` on every run while the runtime measured it the whole
time. End-of-turn squashes were never fed to the alpha window at all.

The gate half of the rule did work throughout, because ``may_speculate`` re-evaluates. That is
the shape every audit of this project found: a mechanism that half-works reads as working.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.integration.test_speculation import TURN, FixedDrafter, OneTurnGraph, registry_for

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolSpec
from specunode.core.model import (
    JournaledModel,
    Message,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolUseComplete,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.drafters.t2_model import ModelDrafter
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger, render_ledger
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world


async def run(
    tmp_path: Path, *, policy: Policy, predictor: object, db: str
) -> tuple[RunResult, World, Journal, str, Scheduler]:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=15.0),
            journal,
            provider="scripted",
        ),
        policy=policy,
        predictor=predictor,  # type: ignore[arg-type]
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    return result, world, journal, run_id, scheduler


def events(journal: Journal, run_id: str, kind: str) -> list[dict[str, object]]:
    return [
        dict(e.payload)
        for e in journal.read(run_id, kinds=["policy_event"])
        if e.payload.get("event") == kind
    ]


@pytest.mark.timeout(60)
async def test_speculative_reads_are_counted_against_the_budget(tmp_path: Path) -> None:
    result, _, _, _, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="reads.db",
    )
    assert result.ok
    assert scheduler.budget.speculative_reads_used > 0, (
        "a guess made a read and the budget's counter never moved"
    )
    # Two counters, two questions, and the difference is deliberate. The report answers attack
    # 7.2's -- "what reached upstream without a durable decision behind it?" -- and includes
    # reads issued early for a turn that is not yet journaled. The budget answers "what may a
    # wrong guess cost?" and charges only reads made on a forked guess. Holding them equal is
    # what made a run with no predictor at all close its own gate.
    assert (
        scheduler.budget.speculative_reads_used <= scheduler.counters.speculative_reads_upstream
    ), "the budget charged reads the report does not even count"
    assert (
        scheduler.counters.speculative_reads_upstream > scheduler.budget.speculative_reads_used
    ), "this turn issues a read early as well as on the guess; the two counts should differ"


@pytest.mark.timeout(60)
async def test_inflight_branches_return_to_zero(tmp_path: Path) -> None:
    """Counted up at fork and down at resolution, on both paths -- previously never at all."""
    result, _, _, _, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="inflight.db",
    )
    assert result.ok
    assert scheduler.counters.branches_forked > 0, "nothing was forked, so this proves nothing"
    assert scheduler.budget.inflight_branches == 0, (
        f"{scheduler.budget.inflight_branches} guess(es) still counted as open after the run"
    )


@pytest.mark.timeout(60)
async def test_every_resolution_is_journaled_with_the_windows_state(tmp_path: Path) -> None:
    """The ledger reads alpha from ``policy_event`` and from nowhere else."""
    result, _, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="alpha.db",
    )
    assert result.ok
    observed = events(journal, run_id, "alpha_observed")
    assert len(observed) == scheduler.budget.window.samples > 0
    last = observed[-1]
    assert last["hits"] == scheduler.budget.window.hits
    assert last["samples"] == scheduler.budget.window.samples
    assert last["window"] == scheduler.policy.alpha_window


@pytest.mark.timeout(60)
async def test_the_ledger_reports_what_was_graded_even_before_the_window_fills(
    tmp_path: Path,
) -> None:
    """``alpha`` is None until 20 samples, on purpose. The receipt still has to say something.

    "n/a" read as "nothing was measured", and that was never true of any speculative run.
    """
    result, _, journal, run_id, _ = await run(
        tmp_path,
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="receipt.db",
    )
    assert result.ok
    ledger = build_ledger(journal, run_id)
    assert ledger.alpha is None, "one sample should not be judged as a rate"
    assert ledger.alpha_samples > 0
    assert ledger.alpha_window == 20, "the window size never reached the ledger before"
    text = render_ledger(ledger)
    # "hits/samples graded" -- the denominator is what was graded, not the window size.
    assert re.search(r"unjudged \(\d+/\d+ graded\)", text), text
    assert f"({ledger.alpha_hits}/{ledger.alpha_samples} graded)" in text
    assert "n/a" not in text.split("alpha (window")[1].split("context")[0]


@pytest.mark.timeout(60)
async def test_an_unmeasured_floor_is_journaled_and_rendered(tmp_path: Path) -> None:
    """The ledger knew how to render this event. Nothing ever emitted it."""
    result, _, journal, run_id, _ = await run(
        tmp_path,
        policy=Policy(speculation=True),  # alpha_floor defaults to None
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="unmeasured.db",
    )
    assert result.ok
    assert events(journal, run_id, "alpha_floor_unmeasured")
    assert "[gate inactive: unmeasured]" in render_ledger(build_ledger(journal, run_id))


@pytest.mark.timeout(60)
async def test_the_gate_closing_is_journaled_exactly_once(tmp_path: Path) -> None:
    result, _, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True, max_speculative_reads=1),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="closed.db",
    )
    assert result.ok
    closed = events(journal, run_id, "speculation_disabled")
    assert len(closed) == 1
    assert closed[0]["reason"] == "max_speculative_reads"
    assert scheduler.counters.speculation_disabled_reason == "max_speculative_reads"
    # The summary entry says it too, where a reader looks first.
    finished = [e.payload for e in journal.read(run_id, kinds=["run_finished"])]
    assert finished, "the run never wrote run_finished"
    assert finished[-1]["counters"]["speculation_disabled_reason"] == "max_speculative_reads"


class _CostlyDraftClient:
    """A draft model whose every answer costs tokens, and always guesses wrong."""

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        # A tool call the model will never emit, so the prediction is always squashed.
        return tool_turn(("charge_card", {"customer_id": "cus-1", "amount": 99.0}), turn=0)


@pytest.mark.timeout(60)
async def test_a_squashed_tier_2_guess_wastes_the_tokens_it_cost(tmp_path: Path) -> None:
    """``record_resolution`` was always passed ``tokens=0``. ``Prediction`` had no cost field.

    ``tool_turn`` stamps every response with 128 input and 64 output tokens, so a squashed
    tier-2 guess must add exactly that to ``wasted_tokens`` -- and with the limit set below it,
    the gate must close for ``max_wasted_tokens``, which was unreachable before.
    """
    drafter = ModelDrafter(client=_CostlyDraftClient())  # type: ignore[arg-type]
    result, world, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True, max_wasted_tokens=100),
        predictor=drafter,
        db="wasted.db",
    )
    assert result.ok, result.error
    assert scheduler.counters.branches_squashed > 0, "no guess was squashed, so nothing wasted"
    assert scheduler.budget.wasted_tokens >= 128 + 64, (
        f"wasted_tokens is {scheduler.budget.wasted_tokens}; the draft model's usage was lost"
    )
    closed = events(journal, run_id, "speculation_disabled")
    assert closed and closed[0]["reason"] == "max_wasted_tokens"
    # The receipt, not only the runtime's own memory. The ledger sums wasted tokens from
    # ``branch_resolved`` and the squash payload did not carry them, so a run whose gate had
    # closed for max_wasted_tokens printed "wasted tokens: 0" on the line above the closure.
    ledger = build_ledger(journal, run_id)
    assert ledger.wasted_tokens == scheduler.budget.wasted_tokens, (
        f"budget says {scheduler.budget.wasted_tokens}, the signed ledger says "
        f"{ledger.wasted_tokens}"
    )
    finished = [e.payload for e in journal.read(run_id, kinds=["run_finished"])]
    assert finished[-1]["counters"]["wasted_tokens"] == scheduler.budget.wasted_tokens
    assert f"wasted tokens: {scheduler.budget.wasted_tokens:,}" in render_ledger(ledger)
    # And the wrong guess never reached the world.
    assert [m.tool for m in world.mutations] == ["restart_job"]


@pytest.mark.timeout(60)
async def test_run_started_carries_the_alpha_configuration(tmp_path: Path) -> None:
    result, _, journal, run_id, _ = await run(
        tmp_path,
        policy=Policy(speculation=True, alpha_window=7, max_wasted_tokens=5000),
        predictor=None,
        db="started.db",
    )
    assert result.ok
    started = next(iter(journal.read(run_id, kinds=["run_started"])))
    policy = started.payload["policy"]
    assert isinstance(policy, dict)
    assert policy["alpha_window"] == 7
    assert policy["max_wasted_tokens"] == 5000
    assert "max_speculative_reads" in policy and "alpha_floor" in policy


@pytest.mark.timeout(60)
async def test_the_receipt_says_which_tier_the_graded_guesses_belong_to(tmp_path: Path) -> None:
    """``AlphaWindow`` graded per tier from the start and nothing reported it.

    The gate consults one rate. A run with two predictors needs to know which one is missing,
    so the event, the ledger and the rendered receipt all carry the per-tier counts.
    """
    result, _, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="tiers.db",
    )
    assert result.ok
    window = scheduler.budget.window
    assert window.samples > 0, "nothing was graded, so this proves nothing"
    last = events(journal, run_id, "alpha_observed")[-1]
    assert last["by_tier"] == {"1": {"hits": window.hits, "samples": window.samples}}
    ledger = build_ledger(journal, run_id)
    assert ledger.alpha_by_tier == {1: (window.hits, window.samples)}
    assert f"by tier: T1 {window.hits}/{window.samples}" in render_ledger(ledger)


class _BlockingRead:
    """A read tool that parks until released, so a guess can be observed while it is open."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, section: str) -> dict[str, object]:
        self.started.set()
        await self.release.wait()
        return {"section": section, "text": "restart it"}


@pytest.mark.timeout(60)
async def test_an_open_guess_is_counted_in_flight_until_it_resolves(tmp_path: Path) -> None:
    """The count has to be 1 *while* the guess is open; the old test only checked it ended at 0.

    ``inflight_branches`` defaults to 0, so a test that runs to completion and asserts zero
    passes just as well with the increment and both decrements deleted -- which is exactly what
    it did. This one blocks the predicted read, looks at the count while the branch is alive,
    then releases it.
    """
    world = standard_world()
    registry = registry_for(world)
    blocking = _BlockingRead()
    registry.register(
        ToolSpec(name="fetch_runbook", effect=EffectClass.READ, fn=blocking, witness=False)
    )
    journal = Journal(tmp_path / "open.db")
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=15.0),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
    )
    task = asyncio.create_task(scheduler.run(new_ulid(), {}))
    await asyncio.wait_for(blocking.started.wait(), timeout=10)
    assert scheduler.budget.inflight_branches == 1, (
        f"a guess is open and running, and the budget counts {scheduler.budget.inflight_branches}"
    )
    blocking.release.set()
    result = await asyncio.wait_for(task, timeout=30)
    assert result.ok, result.error
    assert scheduler.counters.branches_forked > 0
    assert scheduler.budget.inflight_branches == 0, "the guess resolved and was never counted down"


class _FailingStream(ScriptedModel):
    """Emits one block, then raises -- a turn that dies with a guess still open."""

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        emitted = False
        async for event in super().stream(envelope):
            yield event
            if isinstance(event, ToolUseComplete):
                emitted = True
                break
        assert emitted
        raise RuntimeError("the target died mid-turn")


@pytest.mark.timeout(60)
async def test_a_turn_that_dies_with_a_guess_open_still_grades_and_discards_it(
    tmp_path: Path,
) -> None:
    """Every branch ends retired, squashed or stalled -- including when the turn raises.

    Without this, a stream that raised left the guess ungraded, its in-flight slot held for the
    rest of the scheduler's life, its staged effect never discarded, and the journal holding a
    ``branch_forked`` with no ``branch_resolved``.
    """
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "died.db")
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            _FailingStream(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=5.0),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("restart_job", {"job_id": "etl-1"})),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    assert not result.ok, "the target raised; the run cannot have succeeded"

    forked = {str(e.payload["branch_id"]) for e in journal.read(run_id, kinds=["branch_forked"])}
    resolved = {
        str(e.payload["branch_id"]) for e in journal.read(run_id, kinds=["branch_resolved"])
    }
    assert forked, "nothing was forked, so this proves nothing"
    # Both branches: the guess (squashed on the way out of the turn) and the canonical branch
    # of the node visit (faulted). The driven loop journaled the fork and then broke straight
    # out, so the canonical one was left open too.
    assert forked <= resolved, f"{forked - resolved} forked and never resolved"
    statuses = {str(e.payload["status"]) for e in journal.read(run_id, kinds=["branch_resolved"])}
    assert statuses == {"squashed", "faulted"}, statuses
    assert scheduler.budget.inflight_branches == 0, "the guess is still counted as open"
    assert scheduler.budget.window.samples > 0, "the guess was never graded"
    assert [m.tool for m in world.mutations] == [], "a dead turn put something in the world"


@pytest.mark.timeout(60)
async def test_a_run_with_no_predictor_never_charges_the_read_budget(tmp_path: Path) -> None:
    """Early-issued reads are not guesses, and were closing the gate as though they were.

    A read issued for a turn that is not yet durable counts as speculative in the *report*
    (attack 7.2 counts it: it reached upstream without a durable decision). It is not a guess,
    and charging it to ``max_speculative_reads`` disabled speculation on runs that had no
    predictor at all.
    """
    result, _, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True, max_speculative_reads=1),
        predictor=None,
        db="nopredictor.db",
    )
    assert result.ok, result.error
    assert scheduler.counters.speculative_reads_upstream > 0, "no read was issued early"
    assert scheduler.budget.speculative_reads_used == 0, (
        "a run with no predictor charged the guess budget"
    )
    assert not events(journal, run_id, "speculation_disabled"), (
        "a run that never guessed journaled the gate closing"
    )


@pytest.mark.timeout(60)
async def test_the_receipt_keeps_the_closure_and_separates_the_two_read_counts(
    tmp_path: Path,
) -> None:
    """Two defects in one run: the closure scrolled out of the receipt, and the counts disagreed.

    The rendered receipt showed the first three policy events, and alpha observations come
    first, so the line saying speculation had been switched off was dropped after two
    resolutions. The ledger also counted a narrower set of reads than the budget charged, so a
    gate that closed for ``max_speculative_reads`` sat beside "speculative reads upstream: 0".
    """
    result, _, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True, max_speculative_reads=1),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}), limit=3),
        db="receipt.db",
    )
    assert result.ok, result.error
    ledger = build_ledger(journal, run_id)
    assert ledger.speculation_disabled_reason == "max_speculative_reads"
    assert ledger.speculative_reads_charged == scheduler.budget.speculative_reads_used > 0
    assert ledger.speculative_reads_upstream >= ledger.speculative_reads_charged
    rendered = render_ledger(ledger)
    assert "policy: speculation_disabled - max_speculative_reads" in rendered, rendered
    assert f"{ledger.speculative_reads_charged} charged to the read budget" in rendered
    # The window's hits survive the closure entry, which used to reset them to zero.
    assert ledger.alpha_hits == scheduler.budget.window.hits
    assert ledger.alpha_samples == scheduler.budget.window.samples


class ManyTurnsGraph(OneTurnGraph):
    """Four turns through the same node, so a run produces several resolutions."""

    turns = 4

    async def run_node(self, node: object, session: object) -> ToolCall:  # type: ignore[override]
        assert session.call_turn is not None  # type: ignore[attr-defined]
        await session.call_turn(  # type: ignore[attr-defined]
            RequestEnvelope(
                model="scripted",
                messages=(Message(role="user", content=(TextBlock(text="go"),)),),
                max_tokens=128,
                stream=True,
            )
        )
        state = session.state  # type: ignore[attr-defined]
        state["turns"] = int(state.get("turns", 0)) + 1
        if state["turns"] >= self.turns:
            state["done"] = True
        return ToolCall("restart_job", {"job_id": "etl-1"})


@pytest.mark.timeout(60)
async def test_the_closure_survives_a_run_with_more_policy_events_than_the_receipt_shows(
    tmp_path: Path,
) -> None:
    """The receipt showed the first three policy events, and observations come first.

    Every resolution writes an ``alpha_observed``, so on any run with three or more guesses the
    line saying speculation had been switched off -- the one thing in that list a reader is
    looking for -- scrolled off the end. The journal always had it; the rendered receipt did
    not.
    """
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "many.db")
    scheduler = Scheduler(
        graph=ManyTurnsGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(
                turns=[tool_turn(*TURN, turn=i) for i in range(ManyTurnsGraph.turns)],
                block_delay_ms=5.0,
            ),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True, max_speculative_reads=3),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}), limit=20),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    ledger = build_ledger(journal, run_id)
    observed = [e for e in ledger.policy_events if e.event == "alpha_observed"]
    closures = [e for e in ledger.policy_events if e.event == "speculation_disabled"]
    assert len(observed) >= 3, f"only {len(observed)} observations; this proves nothing"
    assert closures, "the gate never closed, so this proves nothing"
    assert ledger.policy_events.index(closures[0]) >= 3, "the closure was already in the first 3"
    rendered = render_ledger(ledger)
    assert "policy: speculation_disabled - max_speculative_reads" in rendered, rendered


@pytest.mark.timeout(60)
async def test_a_closure_journaled_without_its_counts_does_not_zero_the_window(
    tmp_path: Path,
) -> None:
    """Compatibility, and the shape of the original defect.

    Runs journaled before today wrote ``speculation_disabled`` with ``samples`` and no ``hits``,
    and the ledger took any event carrying ``samples`` as the latest state of the window -- so
    the closure, which is usually the last such event, reset the hits to zero and the receipt
    read ``unjudged (0/n graded)`` beside a per-tier line that still said ``T1 n/n``. The
    entries below are written by hand on purpose: they are the shape the runtime no longer
    produces, and the reader still has to be right about them.
    """
    journal = Journal(tmp_path / "legacy.db")
    run_id = new_ulid()
    await journal.append_async(
        run_id,
        "run_started",
        {
            "v": 1,
            "mode": "run",
            "config_hash": "0" * 64,
            "registry_hash": "0" * 64,
            "target": {"provider": "scripted", "model": "scripted"},
            "policy": {"alpha_window": 20},
        },
    )
    await journal.append_async(
        run_id,
        "policy_event",
        {
            "v": 1,
            "event": "alpha_observed",
            "reason": "a prediction resolved",
            "step": 1,
            "alpha": None,
            "hits": 2,
            "samples": 3,
            "window": 20,
            "by_tier": {"1": {"hits": 2, "samples": 3}},
        },
    )
    await journal.append_async(
        run_id,
        "policy_event",
        {
            "v": 1,
            "event": "speculation_disabled",
            "reason": "max_speculative_reads",
            "step": 2,
            "alpha": None,
            "samples": 3,
            "window": 20,
        },
    )
    ledger = build_ledger(journal, run_id)
    assert (ledger.alpha_hits, ledger.alpha_samples) == (2, 3)
    assert ledger.alpha_by_tier == {1: (2, 3)}
    assert "unjudged (2/3 graded)" in render_ledger(ledger)


@pytest.mark.timeout(60)
async def test_a_budget_spent_by_the_last_resolution_still_says_so(tmp_path: Path) -> None:
    """The closure was journaled only when the *next* guess was attempted.

    A budget spent by the final resolution of a run left ``may_speculate()`` False with no
    ``policy_event`` and no reason in the counters: the run stopped speculating and the durable
    record did not say why. One guess, whose read spends the budget, and then the turn ends.
    """
    # Three guesses at 192 tokens each. The first two are squashed by the block that follows
    # them, so a later ``_speculate_next`` would have announced any closure they caused; the
    # third is squashed when the turn ends, after the last block, and crosses the limit. From
    # there nothing asks the gate again.
    drafter = ModelDrafter(client=_CostlyDraftClient())  # type: ignore[arg-type]
    result, _, journal, run_id, scheduler = await run(
        tmp_path,
        policy=Policy(speculation=True, max_wasted_tokens=400),
        predictor=drafter,
        db="last.db",
    )
    assert result.ok, result.error
    assert not scheduler.budget.may_speculate(), "the budget was not spent; this proves nothing"
    closed = events(journal, run_id, "speculation_disabled")
    assert len(closed) == 1, f"{len(closed)} closures journaled"
    assert scheduler.counters.speculation_disabled_reason == "max_wasted_tokens"
    assert build_ledger(journal, run_id).speculation_disabled_reason == "max_wasted_tokens"


@pytest.mark.timeout(60)
async def test_a_resolution_and_its_alpha_event_name_the_same_branch_and_step(
    tmp_path: Path,
) -> None:
    """Two entries describe one resolution, and they have to be joinable.

    ``alpha_observed`` carried the *parent's* cursor, which early reads advance concurrently,
    while the matching ``branch_resolved`` carried the guess's fork step -- so the step numbers
    disagreed, by an amount that depended on which reads happened to have finished.
    """
    result, _, journal, run_id, _ = await run(
        tmp_path,
        policy=Policy(speculation=True),
        predictor=FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"})),
        db="join.db",
    )
    assert result.ok, result.error
    resolutions = {
        str(e.payload["branch_id"]): int(e.payload["step"])
        for e in journal.read(run_id, kinds=["branch_resolved"])
    }
    observed = events(journal, run_id, "alpha_observed")
    assert observed, "nothing was graded, so this proves nothing"
    for event in observed:
        branch_id = event["branch_id"]
        assert branch_id in resolutions, f"{branch_id} has no branch_resolved entry"
        assert event["step"] == resolutions[str(branch_id)], (
            "the alpha event and the resolution disagree about the step"
        )
