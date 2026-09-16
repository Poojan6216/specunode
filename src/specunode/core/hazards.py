"""Hazards: the places where the runtime refuses to guess (Hard Rule 7).

A hazard stalls a branch. It does not degrade into a heuristic, a projection or a best guess.
The list is closed and each member fires at exactly one site, so the benchmark's hazard
histogram has disjoint buckets and a run's stalls can be attributed rather than lumped.

Two detectors do most of the work, and they catch different things:

**Handle scanning** catches a *value* dependency -- a later call whose arguments contain the
placeholder a staged write returned. The predicate is a loose, case-insensitive byte scan over
the exact bytes ``canonical()`` produced, because a mangled prefix is not evidence of safety.
A structural walk runs alongside it to attribute the hit to a path and an effect, but the scan
is what decides. A prefix that resolves to no staged effect in this branch's lineage is still
a hazard: a sibling's handle appearing here would be a Hard Rule 6 violation, and swallowing
it would hide that.

**forward_keys** catches a *key* dependency -- a read touching a resource a staged write
touches, with no placeholder anywhere. Neither detector substitutes for the other, and this
one has no backstop. Witness validation at retirement cannot cover it, because the branch's
own staged write has not been dispatched at that moment and so cannot have made anything
stale. That is why an undeclared ``forward_keys`` fails closed: stage ``close_ticket(T1)``,
read an undeclared ``list_open_tickets()``, see T1 still open, stage ``escalate(T1)`` computed
from that list -- the model confirms, both writes drain, and the world receives an escalation
the sequential run would never have issued, with the leak test green and the read counted
fresh.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from specunode.canonical import JsonValue, canonical
from specunode.core.branch import Branch, BranchStatus
from specunode.core.decision import Decision, ToolCall, is_barrier
from specunode.core.effects import EffectClass, ToolSpec
from specunode.core.policy import Policy

__all__ = [
    "WILDCARD_KEY",
    "BudgetView",
    "HandleRef",
    "Hazard",
    "HazardViolation",
    "analyse",
    "analyse_decision",
    "analyse_model_request",
    "handle_for",
    "has_handle",
    "keys_conflict",
    "scan_handles",
]

#: The placeholder a staged write returns in place of a value it does not have.
HANDLE_PREFIX = "$specunode.handle:"

#: "this call touches something I cannot name". Conflicts with everything, including itself.
WILDCARD_KEY = "*"

#: Loose and case-insensitive: the predicate. A mangled or re-cased prefix is not evidence
#: that no dependency exists, so it stalls too.
_LOOSE_HANDLE = re.compile(rb"(?i)\$specunode\.handle:")
#: Strict: used only to attribute a hit to an effect id.
_STRICT_HANDLE = re.compile(r"\$specunode\.handle:([0-9A-HJKMNP-TV-Z]{26})")


class HazardViolation(RuntimeError):
    """A hazard that should have stalled the branch reached code that assumed it had not."""


class Hazard(Enum):
    """Why a branch stopped guessing. Each fires at exactly one site."""

    RETURN_VALUE_DEPENDENCY = "depends on staged write's return value"
    UNDECLARED_TOOL = "tool has no declared effect class"
    FREE_TEXT_NODE = "node emits free text"
    READ_AFTER_STAGED_WRITE = "read touches a key a staged write touches; no forwarding"
    BUDGET = "speculation budget exhausted"
    IRREVERSIBLE_ON_PATH = "irreversible effect would need staging"
    MODEL_TURN_AFTER_STAGED_WRITE = "next model call would contain a placeholder"
    READ_BUDGET = "speculative read budget exhausted"
    #: Added beyond the spec's eight. A predicted route into a node the adapter cannot run
    #: speculatively must be *named* rather than silently not attempted, or the benchmark's
    #: hazard histogram is incomplete and the honest answer about available speculation is
    #: understated. Logged as a decision in the Progress Log.
    NODE_NOT_SPECULABLE = "predicted route enters a node that cannot run speculatively"


class BudgetView(Protocol):
    """Read-only budget counters. A Protocol so ``Policy`` can stay frozen and immutable."""

    @property
    def speculative_reads_used(self) -> int: ...
    @property
    def wasted_tokens(self) -> int: ...
    @property
    def inflight_branches(self) -> int: ...


@dataclass(frozen=True, slots=True)
class HandleRef:
    """Where a placeholder was found, and which staged effect it names."""

    path: str
    effect_id: str | None
    resolvable: bool


def handle_for(effect_id: str) -> str:
    return f"{HANDLE_PREFIX}{effect_id}"


def has_handle(canonical_bytes: bytes) -> bool:
    """The predicate. Runs over the exact bytes ``canonical()`` produced, never a re-encoding."""
    return _LOOSE_HANDLE.search(canonical_bytes) is not None


def _walk(value: JsonValue, path: str, known: frozenset[str], out: list[HandleRef]) -> None:
    if isinstance(value, str):
        for match in _STRICT_HANDLE.finditer(value):
            effect_id = match.group(1)
            out.append(HandleRef(path, effect_id, effect_id in known))
        if HANDLE_PREFIX in value and not _STRICT_HANDLE.search(value):
            out.append(HandleRef(path, None, False))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            # Keys as well as values: a handle used as an object key is still a dependency.
            _walk(key, f"{path}.<key>", known, out)
            _walk(item, f"{path}.{key}", known, out)
    elif isinstance(value, Sequence):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", known, out)


def scan_handles(args: JsonValue, known: frozenset[str] = frozenset()) -> tuple[HandleRef, ...]:
    """Attribute every placeholder to a path and, where possible, to a staged effect."""
    found: list[HandleRef] = []
    _walk(args, "args", known, found)
    return tuple(found)


def keys_conflict(left: frozenset[str], right: frozenset[str]) -> bool:
    """Whether two resource-key sets may touch the same thing.

    Never a bare intersection: an unnameable key conflicts with everything, because "I cannot
    tell what this touches" and "this touches nothing" are different answers and only one of
    them is safe.
    """
    return WILDCARD_KEY in left or WILDCARD_KEY in right or bool(left & right)


def keys_touched(spec: ToolSpec, args: Mapping[str, JsonValue]) -> frozenset[str]:
    """Resource keys a call touches. An undeclared tool touches the wildcard."""
    declared = spec.keys_touched(args)
    return (
        frozenset({WILDCARD_KEY}) if declared is None else (declared or frozenset({WILDCARD_KEY}))
    )


def analyse_decision(branch: Branch, predicted: Decision, policy: Policy) -> Hazard | None:
    """Filter a *candidate* before a branch is forked on it."""
    if is_barrier(predicted):
        # Free text cannot be confirmed by equality, so speculating on it would mean retiring
        # on an approximate match. This must fire here: analyse() only sees a ToolCall, so
        # leaving it there means it never fires, every free-text node gets speculated on and
        # squashed, and the benchmark's free-text-barrier fraction reads zero.
        return Hazard.FREE_TEXT_NODE
    if branch.depth >= policy.max_speculation_depth:
        return Hazard.BUDGET
    return None


def analyse_model_request(
    branch: Branch, canonical_request: bytes, policy: Policy
) -> Hazard | None:
    """Hard Rule 13's gate, checked before any request leaves the runtime."""
    # Two predicates, and the second is the structural one. A staged slot never fills -- the
    # effect cannot dispatch before the branch retires -- so a request that must include it
    # would either deadlock or carry a synthetic value. Checking only the bytes would miss the
    # case where the staged result is not rendered into the prompt at all but the turn still
    # depends on it having happened.
    if branch.has_staged_slot() or has_handle(canonical_request):
        return Hazard.MODEL_TURN_AFTER_STAGED_WRITE
    return None


def analyse(
    branch: Branch,
    next_call: ToolCall,
    spec: ToolSpec,
    policy: Policy,
    *,
    staged_keys: Sequence[frozenset[str]] = (),
    staged_effect_ids: frozenset[str] = frozenset(),
    budget: BudgetView | None = None,
) -> Hazard | None:
    """Whether this call may run on this branch, and if not, why.

    Precedence is first-match-wins and correctness-bearing predicates come before
    cost-bearing ones, for two reasons. An undeclared tool must be caught first because every
    later predicate reads ``ToolSpec`` fields the registry *synthesised* -- its
    ``forward_keys`` and ``idempotent`` are fiction. And data hazards must be decided before
    budget hazards so the offline opportunity analysis, which has no scheduler and therefore
    no budget state, reproduces the same histogram the online run does.
    """
    canonical_args = canonical(dict(next_call.args))

    if spec.synthesised:
        return Hazard.UNDECLARED_TOOL
    if has_handle(canonical_args):
        return Hazard.RETURN_VALUE_DEPENDENCY
    if (
        spec.effect is EffectClass.IRREVERSIBLE
        and not policy.stage_irreversible
        # Only a guess needs refusing. On a CONFIRMED branch the decision is already the
        # model's and staging is followed immediately by an in-order retirement whose
        # confirming entry exists, so refusing there would mean the call is never dispatched
        # at all and any workload with an irreversible tool could not complete.
        and branch.status is BranchStatus.SPECULATIVE
    ):
        return Hazard.IRREVERSIBLE_ON_PATH
    if spec.effect is EffectClass.READ and staged_keys:
        touched = keys_touched(spec, next_call.args)
        if any(keys_conflict(touched, staged) for staged in staged_keys):
            return Hazard.READ_AFTER_STAGED_WRITE

    # Budget hazards never apply to the canonical path: a CONFIRMED branch that stopped
    # advancing because the speculation budget is spent would make the run hang rather than
    # simply stop speculating.
    if branch.status is not BranchStatus.SPECULATIVE or budget is None:
        return None
    if (
        spec.effect is EffectClass.READ
        and budget.speculative_reads_used >= policy.max_speculative_reads
    ):
        return Hazard.READ_BUDGET
    if (
        branch.depth >= policy.max_speculation_depth
        or budget.inflight_branches > policy.max_inflight_branches
        or budget.wasted_tokens >= policy.max_wasted_tokens
    ):
        return Hazard.BUDGET
    return None
