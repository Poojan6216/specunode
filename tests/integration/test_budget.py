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

import re
from pathlib import Path

import pytest
from tests.integration.test_speculation import TURN, FixedDrafter, OneTurnGraph, registry_for

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import ToolCall
from specunode.core.model import JournaledModel, ModelResponse, RequestEnvelope
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
        "speculative reads were made and the budget's counter never moved"
    )
    assert (
        scheduler.budget.speculative_reads_used == scheduler.counters.speculative_reads_upstream
    ), "the budget's counter and the report's counter disagree about the same reads"


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
