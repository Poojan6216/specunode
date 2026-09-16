"""Budgets and the acceptance rate (spec task 3.6, Hard Rule 10).

Speculation that misses costs real tokens and real upstream reads. A drafter that is wrong
often enough turns a latency win into a latency loss, so every limit here is enforced by
deterministic code and every overrun has a name -- the benchmark's histogram must be able to
say *which* budget stopped a run from speculating, not merely that something did.
"""

from __future__ import annotations

import pytest

from specunode.core.policy import AlphaWindow, Budget, Policy


def test_a_partial_window_has_no_rate_yet() -> None:
    """Disabling speculation on three samples is a worse error than speculating three more."""
    window = AlphaWindow(size=20)
    for _ in range(5):
        window.record(tier=1, confirmed=False)
    assert window.alpha is None
    assert not window.full
    assert not window.below(0.5)


def test_a_full_window_reports_the_rate() -> None:
    window = AlphaWindow(size=10)
    for index in range(10):
        window.record(tier=1, confirmed=index < 7)
    assert window.alpha == pytest.approx(0.7)
    assert window.below(0.8)
    assert not window.below(0.5)


def test_the_window_slides() -> None:
    window = AlphaWindow(size=4)
    for _ in range(4):
        window.record(tier=1, confirmed=False)
    assert window.alpha == 0.0
    for _ in range(4):
        window.record(tier=1, confirmed=True)
    assert window.alpha == 1.0


def test_tier_zero_is_counted_separately_and_never_enters_the_rate() -> None:
    """Its acceptance is 1 by construction; averaging it in holds the rate above any floor.

    Tier 0 only ever proposes calls the target has already emitted, so it cannot miss. Folding
    it into the gate's input would mean the gate never fires however badly the real predictors
    were doing -- the drafter would be measured by the one tier that is not predicting.
    """
    window = AlphaWindow(size=4)
    for _ in range(20):
        window.record(tier=0, confirmed=True)
    assert window.alpha is None, "tier 0 alone must not fill the window"

    for _ in range(4):
        window.record(tier=1, confirmed=False)
    assert window.alpha == 0.0
    assert window.alpha_for(0) == 1.0
    assert window.alpha_for(1) == 0.0


def test_an_unmeasured_rate_is_never_below_a_floor() -> None:
    assert not AlphaWindow(size=10).below(0.9)


# -- budgets ---------------------------------------------------------------------------------


def test_only_a_squashed_branchs_tokens_are_wasted() -> None:
    """A confirmed branch's tokens bought the answer the run needed."""
    budget = Budget(policy=Policy(max_wasted_tokens=1000))
    budget.record_resolution(tier=1, confirmed=True, tokens=400)
    assert budget.wasted_tokens == 0
    budget.record_resolution(tier=1, confirmed=False, tokens=400)
    assert budget.wasted_tokens == 400


def test_each_budget_is_named_when_it_runs_out() -> None:
    """The histogram must say which limit stopped the run, not that something did."""
    budget = Budget(policy=Policy(max_wasted_tokens=100, max_speculative_reads=2))
    assert budget.exhausted() is None

    budget.record_resolution(tier=1, confirmed=False, tokens=150)
    assert budget.exhausted() == "max_wasted_tokens"

    other = Budget(policy=Policy(max_speculative_reads=2))
    other.record_speculative_read()
    other.record_speculative_read()
    assert other.exhausted() == "max_speculative_reads"


def test_the_alpha_gate_disables_speculation_for_the_rest_of_the_run() -> None:
    budget = Budget(policy=Policy(alpha_window=4, alpha_floor=0.5))
    for _ in range(4):
        budget.record_resolution(tier=1, confirmed=False, tokens=1)
    assert budget.should_disable() == "alpha_below_floor"
    assert not budget.may_speculate()


def test_an_unset_floor_never_disables_on_the_rate() -> None:
    """alpha_floor None means 'use the measured break-even', which is a Phase 6 number."""
    budget = Budget(policy=Policy(alpha_window=4, alpha_floor=None))
    for _ in range(8):
        budget.record_resolution(tier=1, confirmed=False, tokens=1)
    assert budget.should_disable() is None
    assert budget.may_speculate()


def test_disabling_is_recorded_with_its_reason() -> None:
    """A run that stopped speculating and one that never started look the same otherwise."""
    budget = Budget(policy=Policy())
    budget.disable("alpha_below_floor")
    assert not budget.may_speculate()
    assert budget.disabled_reason == "alpha_below_floor"


def test_the_window_takes_its_size_from_the_policy() -> None:
    assert Budget(policy=Policy(alpha_window=7)).window.size == 7


def test_a_healthy_run_keeps_speculating() -> None:
    budget = Budget(policy=Policy(alpha_window=4, alpha_floor=0.5, max_wasted_tokens=10_000))
    for _ in range(4):
        budget.record_resolution(tier=1, confirmed=True, tokens=100)
    assert budget.may_speculate()
    assert budget.window.alpha == 1.0
