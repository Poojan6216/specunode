"""Budgets, and the rule that speculation may never cost more than it saves.

Hard Rule 10: wasted work is bounded and measured. Speculation that misses costs real tokens
and real upstream reads, and a drafter that is wrong often enough turns a latency win into a
latency loss. Every limit here is checked by deterministic code -- no model decides whether
to keep speculating.

The interesting one is ``alpha_floor``. The rolling acceptance rate alpha is the fraction of
recent predictions the gate confirmed. Below some workload-specific value, speculating is a
net loss; that value is the *break-even* alpha, and it is measured per workload in Phase 6
rather than guessed. ``alpha_floor=None`` means "use the measured break-even for this
workload"; a number overrides it. When alpha falls below the floor, speculation is switched
off for the rest of the run, a ``policy_event`` is journaled, and the ledger says so -- a run
that quietly stopped speculating and a run that never started look identical otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["AlphaWindow", "Budget", "Policy", "StaleReadAction"]

StaleReadAction = Literal["squash", "stall"]


@dataclass(frozen=True)
class Policy:
    """Limits on how far ahead, how wide and how expensively the runtime may run."""

    #: Master switch. Off means the scheduler takes the same code path with no candidates.
    speculation: bool = True
    #: Top-1 by default. Top-k multiplies the upstream reads a wrong guess pays for, so
    #: widening this is a decision about money as much as about latency.
    max_inflight_branches: int = 1
    max_speculation_depth: int = 3
    max_wasted_tokens: int = 20_000
    #: Upstream reads a squashed branch is allowed to have cost (attack 7.2's budget).
    max_speculative_reads: int = 50
    alpha_window: int = 20
    #: None means "use the break-even measured for this workload" rather than a guess.
    alpha_floor: float | None = None
    #: Default off: an irreversible effect is a barrier, not something to stage. Turning it
    #: on is legitimate and is measured in attack 7.9, and the docs say why it is not default.
    stage_irreversible: bool = False
    on_stale_read: StaleReadAction = "squash"

    def __post_init__(self) -> None:
        if self.max_inflight_branches < 1:
            raise ValueError("max_inflight_branches must be at least 1")
        if self.max_speculation_depth < 0:
            raise ValueError("max_speculation_depth must not be negative")
        if self.max_wasted_tokens < 0:
            raise ValueError("max_wasted_tokens must not be negative")
        if self.max_speculative_reads < 0:
            raise ValueError("max_speculative_reads must not be negative")
        if self.alpha_window < 1:
            raise ValueError("alpha_window must be at least 1")
        if self.alpha_floor is not None and not 0.0 <= self.alpha_floor <= 1.0:
            raise ValueError("alpha_floor is a rate and must lie in [0, 1]")

    @property
    def sequential(self) -> bool:
        """True when this policy asks for no speculation at all."""
        return not self.speculation


@dataclass
class AlphaWindow:
    """The drafter's rolling acceptance rate, over a fixed window.

    Alpha is the fraction of recent predictions the gate confirmed. Below some workload-specific
    value, speculating costs more than it saves -- wasted tokens, upstream reads a squashed
    branch paid for, scheduling overhead -- and that value is the *break-even*, measured per
    workload in Phase 6 rather than guessed at here.

    Tier 0 is counted separately and excluded from the gate's input. Its acceptance is 1 by
    construction, because it only ever proposes calls the target has already emitted; averaging
    it in would hold the rate above any floor no matter how badly the real predictors were
    doing, and the gate would never fire.
    """

    size: int = 20
    _samples: list[bool] = field(default_factory=list)
    _by_tier: dict[int, list[bool]] = field(default_factory=dict)

    def record(self, *, tier: int, confirmed: bool) -> None:
        self._by_tier.setdefault(tier, []).append(confirmed)
        if tier == 0:
            return
        self._samples.append(confirmed)
        if len(self._samples) > self.size:
            del self._samples[: len(self._samples) - self.size]

    @property
    def samples(self) -> int:
        return len(self._samples)

    @property
    def full(self) -> bool:
        """Whether enough predictions have been graded to judge the drafter at all."""
        return len(self._samples) >= self.size

    @property
    def alpha(self) -> float | None:
        """The rate, or ``None`` until the window is full.

        ``None`` rather than a partial average on purpose: disabling speculation on the
        evidence of three predictions is a worse error than speculating three more times.
        """
        if not self.full:
            return None
        return sum(1 for s in self._samples if s) / len(self._samples)

    def alpha_for(self, tier: int) -> float | None:
        graded = self._by_tier.get(tier, [])
        if not graded:
            return None
        return sum(1 for s in graded if s) / len(graded)

    def below(self, floor: float | None) -> bool:
        """Whether the measured rate has fallen under the floor. Unknown is never below."""
        current = self.alpha
        return floor is not None and current is not None and current < floor


@dataclass
class Budget:
    """What speculation has spent so far, and what it is still allowed to spend.

    Hard Rule 10 in one object. Every limit is checked by deterministic code and every overrun
    is a hazard with a name, so the benchmark's histogram can say *which* budget stopped a run
    from speculating rather than reporting that something did.
    """

    policy: Policy
    wasted_tokens: int = 0
    speculative_reads_used: int = 0
    inflight_branches: int = 0
    window: AlphaWindow = field(default_factory=AlphaWindow)
    #: Set once the alpha gate fires. A run that stopped speculating and a run that never
    #: started look identical in the effect ledger, so the difference is recorded explicitly.
    speculation_disabled: bool = False
    disabled_reason: str | None = None

    def __post_init__(self) -> None:
        self.window.size = self.policy.alpha_window

    def record_resolution(self, *, tier: int, confirmed: bool, tokens: int = 0) -> None:
        self.window.record(tier=tier, confirmed=confirmed)
        if not confirmed:
            # Only a squashed branch's tokens are wasted. A confirmed branch's tokens bought
            # the answer the run needed.
            self.wasted_tokens += tokens

    def record_speculative_read(self) -> None:
        self.speculative_reads_used += 1

    def disable(self, reason: str) -> None:
        self.speculation_disabled = True
        self.disabled_reason = reason

    def exhausted(self) -> str | None:
        """The name of the first budget that is spent, or ``None``."""
        if self.wasted_tokens >= self.policy.max_wasted_tokens:
            return "max_wasted_tokens"
        if self.speculative_reads_used >= self.policy.max_speculative_reads:
            return "max_speculative_reads"
        return None

    def should_disable(self) -> str | None:
        """Whether speculation should stop for the rest of this run, and why."""
        if self.speculation_disabled:
            return self.disabled_reason
        spent = self.exhausted()
        if spent is not None:
            return spent
        if self.window.below(self.policy.alpha_floor):
            return "alpha_below_floor"
        return None

    def may_speculate(self) -> bool:
        return not self.speculation_disabled and self.should_disable() is None
