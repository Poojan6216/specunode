"""The effect ledger: which model decision authorised each effect that left the runtime.

This is the artifact the project produces. Anything can print "done"; a ledger says which
decision authorised each effect, what was thrown away, where the runtime had to fall back to
sequential execution, and what the speculation cost. Section 2's Demo 3 shows the rendered
form, and that rendering is a contract.

Four properties carry the whole module, and each one exists because the obvious alternative
fails silently.

**A ledger is a pure function of journal entries, and of nothing else.**
:func:`build_ledger` never consults a :class:`~specunode.buffer.store_buffer.DrainReport`,
a live :class:`~specunode.core.branch.Branch`, or any in-memory counter. Spec task 2.5
requires a resumed run's ledger to equal the uninterrupted run's after normalisation, and on
a resume the in-memory objects simply do not exist -- but worse than absent, they disagree.
When a process dies between sending an effect and recording that it sent it, the resumed run
re-stages the effect, meets ``ALREADY_DISPATCHED`` at the dedupe check and sends nothing; its
``DrainReport`` says ``SKIPPED_DEDUPE`` where the uninterrupted run's said ``DISPATCHED``. The
journal says the same thing in both runs -- one ``effect_dispatched`` entry, written in the
same transaction as the dedupe row -- so a ledger built from entries renders one
``DISPATCHED`` row either way, and a ledger built from the report renders a status that
exists in no other run of the same workload.

**Rows are ordered by** ``(retire_seq, dispatch_index)``, **never by** ``stage_index``.
``stage_index`` is recorded when an effect is *staged*; ``dispatch_index`` when it is sent.
A ledger sorted by the former takes a drain that iterated its buffer backwards and puts the
rows back into stage order, so every column matches the sequential arm and the equivalence
test's mandated "dispatch order reversed" planted bug passes green. Sorting by the order
things actually left is the only ordering under which that bug is visible.

**Nothing signed depends on** ``run_finished``. ``run_finished`` carries the ledger's own
signature, so any payload field derived from it would be defined in terms of a value that
does not exist until after the signature is computed. ``journal_head`` is therefore the chain
head immediately *before* ``run_finished``, ``journal_entries`` excludes it, and every
aggregate below is recomputed from the entries that record the underlying event --
``branch_resolved`` for tokens and squashes, ``policy_event`` for alpha, ``tool_result`` for
upstream reads. ``run_finished``'s own ``counters`` are the runtime's self-report and are
deliberately not used: a ledger that copied them would be corroborating the bookkeeping with
the bookkeeping.

**A stamp for a property nobody checked is worse than no stamp.** ``context_identity`` renders
``unchecked`` when ``context_checks[0] == 0``, ``enforced(derivable)`` when the run injected
prompt material the runtime cannot re-derive, and ``unenforced`` on the MCP proxy path where
the runtime never sees the request at all. The same reasoning is why :func:`render_ledger`
prints the journal head rather than the words "chain ok": it never walked the chain.

**What the signature proves, and what it does not.** SpecuNode is a local research tool with
no accounts (Hard Rule 11). The Ed25519 key lives unencrypted under ``.specunode/`` and is
minted on first use with no passphrase and no prompt, because a prompt breaks CI and
non-interactive runs. A valid signature therefore shows that these rows are the bytes that
were signed by whoever held that key file -- tamper-evidence for a ledger that leaves the
machine. It says nothing about *who* that was, nothing about whether the effects happened in
the world (only that the runtime recorded them), and nothing about freshness: an older valid
ledger can be presented again, and only the journal binding in check 3 ties a ledger to a
run. Anyone who can read the key file can sign as its owner, which defeats checks 1 and 2 and
leaves only check 3.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import blake2b
from pathlib import Path
from typing import Final, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from specunode.canonical import JsonValue, canonical, chash
from specunode.core.decision import ToolCall, decision_payload
from specunode.core.hazards import Hazard
from specunode.journal.entries import Entry, entry_hash, genesis_prev_hash
from specunode.journal.journal import Journal

__all__ = [
    "DEFAULT_KEYSTORE",
    "EffectStatus",
    "Ledger",
    "LedgerRow",
    "LedgerVerification",
    "PolicyEventRow",
    "ReadTally",
    "SigningKey",
    "Stall",
    "build_ledger",
    "build_ledger_from_entries",
    "journal_head",
    "key_id_for",
    "ledger_payload",
    "load_or_create_key",
    "render_ledger",
    "sign_ledger",
    "signed_bytes",
    "trusted_key_ids",
    "verify_ledger",
]

#: Domain separator. The same blake2b construction hashes journal entries and idempotency
#: keys, so a ledger signature preimage is prefixed to make a cross-context collision
#: impossible even if identical bytes were fed to all three.
SIGN_DOMAIN: Final = b"specunode/ledger/v1\x00"

#: Where a run's key material lives. Gitignored; the only committed key is the insecure test
#: fixture. Overridable by ``$SPECUNODE_SIGNING_KEY`` (a path to a PEM).
DEFAULT_KEYSTORE: Final = Path("./.specunode")

_KEY_ENV: Final = "SPECUNODE_SIGNING_KEY"
_PAYLOAD_VERSION: Final = 1

EffectStatus = Literal["DISPATCHED", "DEAD_LETTER", "COMPENSATED"]

_ELLIPSIS: Final = "…"
_MAX_EFFECT_CELL: Final = 60
#: 8 hex, not section 2's illustrative 4: four hex digits collide with probability ~26% over
#: a 200-effect run, and two ledger rows that render the same key are two rows an auditor
#: cannot tell apart.
_KEY_PREFIX: Final = 8

_HAZARD_BY_VALUE: Final = {hazard.value: hazard for hazard in Hazard}
_HAZARD_BY_NAME: Final = {hazard.name: hazard for hazard in Hazard}


# -- small typed accessors over journal payloads ------------------------------------------
#
# Journal payloads are Mappings of JsonValue, so every read needs narrowing. Doing it once
# here keeps the builder readable and, more usefully, makes "this field was absent or the
# wrong type" degrade to a stated default rather than to a TypeError halfway through
# rendering an audit artifact.


def _as_str(payload: Mapping[str, JsonValue], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _as_int(payload: Mapping[str, JsonValue], key: str, default: int = -1) -> int:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _as_bool(payload: Mapping[str, JsonValue], key: str, default: bool = False) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else default


def _as_map(payload: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _as_seq(payload: Mapping[str, JsonValue], key: str) -> Sequence[JsonValue]:
    value = payload.get(key)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _as_float(payload: Mapping[str, JsonValue], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


# -- the rows and the ledger ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One effect that left the runtime, and the decision that authorised it."""

    effect_id: str
    call: ToolCall
    #: Hard Rule 8's lineage-bearing key. Rendered and signed; never compared across runs,
    #: because the lineage is a property of how the runtime got here, not of the effect.
    key: str
    #: Lineage-free. The token the tool adapter received as its idempotency key, and the only
    #: one of the three keys the world ever sees.
    nkey: str
    node_id: str
    step_index: int
    branch_id: str
    #: Fork-order ordinal, rendered as ``br-NN``. A ULID is unreadable in a table.
    branch_ord: int
    stage_index: int
    #: Position in the order effects actually left. The row sort key, with ``retire_seq``.
    dispatch_index: int
    retire_seq: int
    #: Section 7 calls this "journal offset of the confirming model_response"; Demo 3 prints
    #: "step 4". Hard Rule 9 decides between them: a speculative run journals entries a
    #: sequential run never writes, so offsets for the same decision differ by an amount
    #: proportional to how much speculation happened, and a ledger field carrying one could
    #: never compare equal across the two arms. It is the confirming turn's ``step_index``.
    authorised_by_step: int
    status: EffectStatus
    reason: str | None = None
    ack: JsonValue = None
    ack_hash: str | None = None
    #: Set on a compensation row (correction C18); names the effect being undone.
    compensates: str | None = None
    #: True when ``effect_staged`` and ``effect_dispatched`` disagree about ``args_hash``.
    #: The two entries store the arguments once and the digest twice precisely so that a
    #: back-patched argument between staging and dispatch has somewhere to show up.
    args_mismatch: bool = False
    #: True when the dispatcher was told not to call the tool -- ``specunode replay`` without
    #: ``--dispatch``. The status still reads DISPATCHED because that is what the run *did*
    #: with the effect; this says the world never heard about it, and the rendering says so on
    #: the row rather than only in a banner somebody may not have kept.
    dry_run: bool = False

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.retire_seq, self.dispatch_index)

    @property
    def branch_label(self) -> str:
        return "br-??" if self.branch_ord < 0 else f"br-{self.branch_ord:02d}"


@dataclass(frozen=True, slots=True)
class Stall:
    """One branch that stopped guessing, and why.

    Section 7 types this ``tuple[int, Hazard]``, which cannot render Demo 3's own stall line
    -- ``stalls: 1 (free-text node "draft_reply")`` names the node. The node is recovered
    from the branch's ``branch_forked`` entry, so this stays a pure function of the journal.
    """

    step: int
    hazard: Hazard | None
    node_id: str
    #: The raw ``reason``/``hazard`` string when it names no member of the closed enum. Kept
    #: rather than dropped: a stall the ledger cannot classify is exactly the one worth
    #: showing an operator verbatim.
    raw: str = ""

    @property
    def slug(self) -> str:
        if self.hazard is None:
            return self.raw or "unclassified"
        name = self.hazard.name.lower().replace("_", "-")
        return name[: -len("-node")] if name.endswith("-node") else name


@dataclass(frozen=True, slots=True)
class ReadTally:
    """Read validation at retirement (rule E3).

    ``unwitnessed`` is a third verdict, not a flavour of not-fresh. Section 7's witness
    contract makes reporting an unwitnessed read as fresh a direct spec violation, and
    ``(fresh, total)`` as section 7 types it cannot express the distinction at all.
    """

    fresh: int = 0
    stale: int = 0
    unwitnessed: int = 0
    #: Re-probes that could not complete. A separate verdict again, and for the same reason:
    #: folding it into ``fresh`` would report an unchecked read as a checked one, and dropping
    #: it entirely removed it from both the numerator and the denominator of the rendered line.
    unreadable: int = 0
    total: int = 0

    @property
    def witnessed(self) -> int:
        """Reads that carried a witness, and so were *meant* to be validated.

        ``fresh + stale + unreadable``. It used to be ``fresh + stale``, which dropped a read
        whose re-probe raised from both the numerator and the denominator -- so a run where
        every probe errored rendered as ``0/0 fresh`` rather than as nothing having been
        checked. A denominator that shrinks when a check fails reports the failure as absence.
        """
        return self.fresh + self.stale + self.unreadable


@dataclass(frozen=True, slots=True)
class PolicyEventRow:
    step: int
    event: str
    reason: str


@dataclass(frozen=True, slots=True)
class JournalPosition:
    """The chain head and entry count a ledger binds itself to."""

    head: str
    entries: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Ledger:
    """A run's effects and what the speculation around them cost.

    Keyword-only: this is a twenty-odd field record, and two adjacent ``int`` counters
    transposed positionally would produce a plausible, wrong, signed artifact.
    """

    run_id: str
    #: Set iff this ledger was built from a replay store. ``verify-ledger`` resolves the
    #: journal by ``(run_id, replay_id)``; a replay runs under its origin's ``run_id``, so
    #: without this a replay ledger and its origin are indistinguishable by name while their
    #: entry counts, chain heads and therefore signatures necessarily differ.
    replay_id: str | None = None
    parent_run_id: str | None = None
    rows: tuple[LedgerRow, ...] = ()

    squashed_branches: int = 0
    discarded_effects: int = 0
    stalls: tuple[Stall, ...] = ()
    reads_validated: ReadTally = ReadTally()
    wasted_tokens: int = 0
    #: Reads that reached an upstream system without a durable decision behind them -- from a
    #: branch that was still a guess, or issued early for a turn not yet journaled -- plus
    #: witness re-fetches. The same definition as the runtime's own tally. Excluded from the
    #: equivalence relation and printed in every rendering: Hard Rule 9 is about the *effect*
    #: ledger, and speculative reads genuinely do reach upstream. That exclusion is honest
    #: only because the number is never hidden.
    speculative_reads_upstream: int = 0
    #: The share of those the read budget charged: reads made on forked guesses. This is what
    #: ``max_speculative_reads`` bounds, and the gate can close on it while the wider number
    #: above keeps growing, so the two are printed together.
    speculative_reads_charged: int = 0
    context_divergences: int = 0
    #: ``(checked, recorded)`` -- how many of the prompts a branch recorded were rebuilt and
    #: compared at retirement (Hard Rule 13).
    context_checks: tuple[int, int] = (0, 0)
    #: Reads issued while an ancestor's drain was in flight (correction C13). Reported, not
    #: blocked: blocking every descendant read until its ancestors' drains complete costs
    #: exactly the latency past-write speculation exists to buy, to close a race that can
    #: only arise from under-declared ``forward_keys``.
    reads_raced_drain: int = 0
    #: Prompt blocks a node supplied that the runtime cannot re-derive (correction C10).
    injected_blocks: int = 0
    alpha: float | None = None
    alpha_window: int | None = None
    #: What was actually graded, whether or not the window filled. ``alpha`` alone rendered
    #: ``n/a`` on every run shorter than the window -- which is most runs -- so a ledger said
    #: nothing about a rate the runtime had been measuring the whole time.
    alpha_hits: int = 0
    alpha_samples: int = 0
    #: The same counts per tier, tier 0 included, so the receipt can say which predictor the
    #: misses belong to. ``AlphaWindow`` kept this from the start and nothing ever reported it.
    alpha_by_tier: dict[int, tuple[int, int]] = field(default_factory=dict)
    #: Why speculation was switched off for the rest of the run, if it was.
    speculation_disabled_reason: str | None = None
    #: True when no break-even was measured for this workload, so the alpha gate is inactive.
    alpha_gate_unmeasured: bool = False
    #: Effects a branch sent out of program order (``_order_anomalies``).
    dispatch_order_anomalies: int = 0
    #: Effects whose staged and dispatched ``args_hash`` disagree.
    args_hash_mismatches: int = 0
    post_write_span: int = 0
    policy_events: tuple[PolicyEventRow, ...] = ()

    speculation: bool = False
    drafters: tuple[str, ...] = ()
    target_model: str = ""
    adapter: str = ""

    journal_entries: int = 0
    journal_head: str = ""
    #: ``run_finished`` is present. A precondition for comparing two ledgers, never a
    #: compared field -- and never part of the signed payload, because ``run_finished``
    #: carries the signature.
    terminal: bool = False
    #: ``""`` means unsigned, which is never conflated with forged.
    signature: str = ""

    @property
    def context_identity(
        self,
    ) -> Literal["enforced", "enforced(derivable)", "unenforced", "unchecked"]:
        """The Hard Rule 13 stamp, which never claims more than was checked.

        ``unenforced`` on the proxy path is not a weaker promise made quietly: it is the one
        path where the runtime cannot see the request at all, which is precisely why the
        stamp exists. ``unchecked`` is what a run gets when the check count is zero -- the
        failure correction C2 describes, where a sequential run stamps ``enforced`` having
        compared nothing.
        """
        if self.adapter == "mcp-proxy":
            return "unenforced"
        if self.context_checks[0] <= 0:
            return "unchecked"
        return "enforced(derivable)" if self.injected_blocks else "enforced"


# -- building -------------------------------------------------------------------------------


@dataclass
class _Build:
    """Scratch state for one pass over a run's entries."""

    run_id: str
    by_offset: dict[int, Entry] = field(default_factory=dict)
    staged: dict[str, Mapping[str, JsonValue]] = field(default_factory=dict)
    fork: dict[str, Mapping[str, JsonValue]] = field(default_factory=dict)
    fork_ord: dict[str, int] = field(default_factory=dict)
    resolutions: dict[str, list[Mapping[str, JsonValue]]] = field(default_factory=dict)
    first_resolution_offset: dict[str, int] = field(default_factory=dict)
    explicit_retire_seq: dict[str, int] = field(default_factory=dict)
    final_status: dict[str, str] = field(default_factory=dict)


def journal_head(entries: Iterable[Entry], run_id: str) -> JournalPosition:
    """The chain head and entry count a ledger binds itself to, excluding ``run_finished``.

    Correction C26. ``run_finished`` carries ``ledger_payload_hash`` and the signature, so
    including it would define the signed value in terms of an entry that does not exist until
    after the value is computed -- the journal-binding check would then fail on every genuine
    ledger, and the natural repair is to delete the check.
    """
    head = genesis_prev_hash(run_id)
    count = 0
    for entry in entries:
        if entry.kind == "run_finished":
            break
        head = entry_hash(
            run_id=entry.run_id,
            offset=entry.offset,
            kind=entry.kind,
            ts=entry.ts,
            payload_hash=entry.payload_hash,
            prev_hash=entry.prev_hash,
        )
        count += 1
    return JournalPosition(head=head, entries=count)


def build_ledger(journal: Journal, run_id: str) -> Ledger:
    """Render a run's ledger from its journal. Pure: the only input is the entries."""
    return build_ledger_from_entries(journal.read(run_id), run_id)


def build_ledger_from_entries(entries: Iterable[Entry], run_id: str) -> Ledger:
    """:func:`build_ledger` over an explicit entry sequence, for replay stores and tests."""
    ordered = sorted(entries, key=lambda entry: entry.offset)
    state = _Build(run_id=run_id)
    for entry in ordered:
        state.by_offset[entry.offset] = entry

    started: Mapping[str, JsonValue] = {}
    terminal = False
    dispatched: list[tuple[Entry, Mapping[str, JsonValue]]] = []
    dead_lettered: list[tuple[Entry, Mapping[str, JsonValue]]] = []
    discarded = 0
    reads = ReadTally()
    raced_from_results: set[str] = set()
    raced_from_validation = 0
    upstream = 0
    injected = 0
    context_checked = 0
    context_recorded = 0
    divergences = 0
    policy_events: list[PolicyEventRow] = []
    alpha: float | None = None
    alpha_hits = 0
    alpha_samples = 0
    alpha_by_tier: dict[int, tuple[int, int]] = {}
    alpha_gate_unmeasured = False
    reads_charged = 0
    disabled_reason: str | None = None

    for entry in ordered:
        payload = entry.payload
        kind = entry.kind
        if kind == "run_started":
            started = payload
        elif kind == "run_finished":
            terminal = True
        elif kind == "branch_forked":
            branch_id = _as_str(payload, "branch_id")
            state.fork.setdefault(branch_id, payload)
            if branch_id not in state.fork_ord:
                explicit = _as_int(payload, "fork_ord")
                state.fork_ord[branch_id] = explicit if explicit >= 0 else len(state.fork_ord)
        elif kind == "branch_resolved":
            _absorb_resolution(state, entry, payload)
            if _as_map(payload, "context_divergence") or _as_str(payload, "reason") == (
                "context_divergence"
            ):
                divergences += 1
            checks = _as_seq(payload, "context_checks")
            if len(checks) == 2:
                first, second = checks[0], checks[1]
                if isinstance(first, int) and not isinstance(first, bool):
                    context_checked += first
                if isinstance(second, int) and not isinstance(second, bool):
                    context_recorded += second
        elif kind == "effect_staged":
            state.staged[_as_str(payload, "effect_id")] = payload
        elif kind == "effect_dispatched":
            dispatched.append((entry, payload))
        elif kind == "effect_dead_lettered":
            dead_lettered.append((entry, payload))
        elif kind == "effect_discarded":
            discarded += max(0, _as_int(payload, "count", 0))
        elif kind == "read_validated":
            reads = ReadTally(
                fresh=reads.fresh + max(0, _as_int(payload, "fresh", 0)),
                stale=reads.stale + max(0, _as_int(payload, "stale", 0)),
                unwitnessed=reads.unwitnessed + max(0, _as_int(payload, "unwitnessed", 0)),
                unreadable=reads.unreadable + max(0, _as_int(payload, "unreadable", 0)),
                total=reads.total + max(0, _as_int(payload, "total", 0)),
            )
            raced_from_validation += max(0, _as_int(payload, "raced_drain", 0))
        elif kind == "tool_result":
            if _as_bool(payload, "raced_drain"):
                raced_from_results.add(_as_str(payload, "call_id") or str(entry.offset))
        elif kind == "model_request":
            # A turn served from the journal on resume repeats one already counted.
            if _as_str(payload, "role", "target") == "target" and "recorded_from" not in payload:
                index = _as_seq(payload, "injected_index")
                injected += len(index) if index else len(_as_seq(payload, "injected"))
        elif kind == "policy_event":
            event = _as_str(payload, "event")
            policy_events.append(
                PolicyEventRow(
                    step=_as_int(payload, "step"),
                    event=event,
                    reason=_as_str(payload, "reason"),
                )
            )
            reported = _as_float(payload, "alpha")
            if reported is not None:
                alpha = reported
            if event in ("alpha_observed", "speculation_disabled") and "hits" in payload:
                # The last observation wins: the window is a rolling one and the latest
                # event carries its current state. Only events that carry the whole state
                # count -- a closure entry without ``hits`` used to zero the hits.
                alpha_hits = max(0, _as_int(payload, "hits", 0))
                alpha_samples = max(0, _as_int(payload, "samples", 0))
                alpha_by_tier = _tiers_from(payload.get("by_tier"))
            if "speculative_reads_used" in payload:
                reads_charged = max(0, _as_int(payload, "speculative_reads_used", 0))
            if event == "speculation_disabled":
                disabled_reason = _as_str(payload, "reason") or "unknown"
            if event == "alpha_floor_unmeasured":
                alpha_gate_unmeasured = True

    # A second pass for tool_result, because "did this read's branch ever retire" is only
    # answerable once every branch_resolved entry has been seen.
    for entry in ordered:
        if entry.kind != "tool_result" or not _as_bool(entry.payload, "reached_upstream"):
            continue
        branch_id = _as_str(entry.payload, "branch_id")
        if (
            _as_bool(entry.payload, "witness_probe")
            or _as_bool(entry.payload, "speculative")
            or state.final_status.get(branch_id) not in ("retired", "confirmed")
        ):
            upstream += 1

    retire_seq = _retire_order(state, dispatched, dead_lettered)
    rows = _rows(state, dispatched, dead_lettered, retire_seq)
    position = journal_head(ordered, run_id)
    policy = _as_map(started, "policy")
    graph = _as_map(started, "graph")
    target = _as_map(started, "target")

    return Ledger(
        run_id=run_id,
        replay_id=_as_str(started, "replay_id") or None,
        parent_run_id=_as_str(started, "parent_run_id") or None,
        rows=rows,
        squashed_branches=sum(1 for status in state.final_status.values() if status == "squashed"),
        discarded_effects=discarded,
        stalls=_stalls(state),
        reads_validated=reads,
        wasted_tokens=_wasted_tokens(state),
        speculative_reads_upstream=upstream,
        speculative_reads_charged=reads_charged,
        speculation_disabled_reason=disabled_reason,
        context_divergences=divergences,
        context_checks=(context_checked, context_recorded),
        reads_raced_drain=len(raced_from_results) or raced_from_validation,
        injected_blocks=injected,
        alpha=alpha,
        alpha_window=_as_int(policy, "alpha_window", -1) if policy else None,
        alpha_hits=alpha_hits,
        alpha_samples=alpha_samples,
        alpha_by_tier=alpha_by_tier,
        alpha_gate_unmeasured=alpha_gate_unmeasured,
        dispatch_order_anomalies=_order_anomalies(rows),
        args_hash_mismatches=sum(1 for row in rows if row.args_mismatch),
        post_write_span=_post_write_span(state),
        policy_events=tuple(policy_events),
        speculation=_as_bool(policy, "speculation"),
        drafters=tuple(
            f"t{_as_int(drafter, 'tier', -1)}"
            for drafter in _as_seq(started, "drafters")
            if isinstance(drafter, Mapping)
        ),
        target_model=_as_str(target, "model"),
        adapter=_as_str(graph, "adapter"),
        journal_entries=position.entries,
        journal_head=position.head,
        terminal=terminal,
    )


def _absorb_resolution(state: _Build, entry: Entry, payload: Mapping[str, JsonValue]) -> None:
    branch_id = _as_str(payload, "branch_id")
    state.resolutions.setdefault(branch_id, []).append(payload)
    status = _as_str(payload, "status")
    # A branch is confirmed, then retired. The last word is the branch's fate; "confirmed"
    # surviving as the final status means the process died mid-drain, which is a different
    # thing from a completed retirement and is left visible as such.
    if status:
        state.final_status[branch_id] = status
    if status in ("confirmed", "retired") and branch_id not in state.first_resolution_offset:
        state.first_resolution_offset[branch_id] = entry.offset
    if status == "retired":
        recorded = _as_int(payload, "retire_seq")
        if recorded >= 0:
            state.explicit_retire_seq[branch_id] = recorded


def _retire_order(
    state: _Build,
    dispatched: Sequence[tuple[Entry, Mapping[str, JsonValue]]],
    dead_lettered: Sequence[tuple[Entry, Mapping[str, JsonValue]]],
) -> dict[str, int]:
    """Program-order rank of each branch's retirement.

    Derived from the offset of the entry that *confirmed* the branch rather than the one that
    retired it, because confirmation is durable before a drain may begin (Hard Rule 3) while
    the retirement entry is written after it -- so a run killed mid-drain still has a total
    order over exactly the branches that dispatched anything. Retirement is in program order,
    so ranking by confirmation offset and ranking by retirement agree whenever both exist.
    """
    first_seen: dict[str, int] = dict(state.first_resolution_offset)
    for entry, payload in (*dispatched, *dead_lettered):
        branch_id = _as_str(payload, "branch_id")
        if branch_id not in first_seen:
            first_seen[branch_id] = entry.offset
    derived = {
        branch_id: rank
        for rank, branch_id in enumerate(sorted(first_seen, key=lambda b: first_seen[b]))
    }
    # All-or-nothing: mixing a journaled ordinal with a derived one can interleave two
    # different orderings and produce a row order neither the runtime nor the journal claims.
    if state.explicit_retire_seq and set(derived) <= set(state.explicit_retire_seq):
        return dict(state.explicit_retire_seq)
    return derived


def _rows(
    state: _Build,
    dispatched: Sequence[tuple[Entry, Mapping[str, JsonValue]]],
    dead_lettered: Sequence[tuple[Entry, Mapping[str, JsonValue]]],
    retire_seq: Mapping[str, int],
) -> tuple[LedgerRow, ...]:
    rows: list[LedgerRow] = []
    compensated: set[str] = set()
    for entry, payload in dispatched:
        compensates = _as_str(payload, "compensation_for") or None
        if compensates:
            compensated.add(compensates)
        rows.append(_row(state, entry, payload, retire_seq, "DISPATCHED", compensates))
    # A dead letter whose effect later went out -- sent by a resume under the same key once it
    # was known never to have left, or resolved as landed by someone who checked the upstream --
    # is one effect with one outcome, the later one. Keyed by the dedupe key, not the effect id:
    # each resume stages the effect afresh under a new id, and the key is what makes it the
    # same effect.
    # And a resume that dead-letters it again does not make it two effects: the latest dead
    # letter for a key is the one that stands.
    sent = {_as_str(payload, "nkey") for _entry, payload in dispatched}
    latest: dict[str, tuple[Entry, Mapping[str, JsonValue]]] = {}
    for entry, payload in dead_lettered:
        latest[_as_str(payload, "nkey") or _as_str(payload, "effect_id")] = (entry, payload)
    for key, (entry, payload) in latest.items():
        if key in sent:
            continue
        rows.append(_row(state, entry, payload, retire_seq, "DEAD_LETTER", None))
    # A compensated effect's own row says so: the original reached the world and was later
    # undone, and a reader who sees only DISPATCHED there would be reading a world state that
    # no longer holds.
    rows = [
        replace(row, status="COMPENSATED") if row.effect_id in compensated else row for row in rows
    ]
    return tuple(sorted(rows, key=lambda row: (row.sort_key, row.effect_id)))


def _row(
    state: _Build,
    entry: Entry,
    payload: Mapping[str, JsonValue],
    retire_seq: Mapping[str, int],
    status: EffectStatus,
    compensates: str | None,
) -> LedgerRow:
    effect_id = _as_str(payload, "effect_id")
    staged = state.staged.get(effect_id, {})
    branch_id = _as_str(payload, "branch_id") or _as_str(staged, "branch_id")
    # Arguments are stored once, on effect_staged, and hashed on both entries. The join is on
    # effect_id and the digests are cross-checked, so an argument rewritten between staging
    # and dispatch has somewhere to surface instead of being rendered as though the staged
    # call were the one that left.
    staged_hash = _as_str(staged, "args_hash")
    dispatched_hash = _as_str(payload, "args_hash")
    args = _as_map(staged, "args")
    stage_index = _as_int(payload, "stage_index")
    if stage_index < 0:
        stage_index = _as_int(staged, "stage_index")
    dispatch_index = _as_int(payload, "dispatch_index")
    if dispatch_index < 0:
        # An effect_dead_lettered written before it recorded where it was tried. The drain
        # halts at the first dead letter, so its stage position stands in -- for display only:
        # ``_order_anomalies`` leaves dead letters out.
        dispatch_index = stage_index
    ack = payload.get("ack")
    return LedgerRow(
        effect_id=effect_id,
        call=ToolCall(
            name=_as_str(payload, "tool") or _as_str(staged, "tool"),
            args=args,
        ),
        key=_as_str(payload, "key") or _as_str(staged, "key"),
        nkey=_as_str(payload, "nkey") or _as_str(staged, "nkey"),
        node_id=_as_str(staged, "node_id"),
        step_index=_as_int(staged, "step"),
        branch_id=branch_id,
        branch_ord=state.fork_ord.get(branch_id, -1),
        stage_index=stage_index,
        dispatch_index=dispatch_index,
        retire_seq=retire_seq.get(branch_id, len(retire_seq)),
        authorised_by_step=_authorised_by_step(state, payload, branch_id),
        status=status,
        reason=_dead_letter_reason(payload) if status == "DEAD_LETTER" else None,
        ack=ack,
        ack_hash=None if ack is None else chash(ack),
        compensates=compensates,
        args_mismatch=bool(staged_hash and dispatched_hash and staged_hash != dispatched_hash),
        dry_run=_as_bool(payload, "dry_run"),
    )


def _order_anomalies(rows: Sequence[LedgerRow]) -> int:
    """Effects each branch sent out of program order: by position, then by stage.

    Program order, not stage order. A confirmed guess used to join its branch when its turn
    was journaled, before the writes the model emitted ahead of it were staged -- so its stage
    index was lower though its position was later -- and the drain, which sends by position,
    sent a correct run in exactly the right order while this reported it out of order.

    Only effects that were sent. A dead letter's place in the send order was not recorded until
    later, and the stage index that stood in for it collided with a sent effect's, so a correct
    run whose last effect failed was reported out of order.
    """
    anomalies = 0
    by_branch: dict[str, list[LedgerRow]] = {}
    for row in rows:
        if row.status in ("DISPATCHED", "COMPENSATED"):
            by_branch.setdefault(row.branch_id, []).append(row)
    for branch_rows in by_branch.values():
        program = sorted(branch_rows, key=lambda row: (row.step_index, row.stage_index))
        sent = sorted(branch_rows, key=lambda row: row.dispatch_index)
        anomalies += sum(
            1 for a, b in zip(program, sent, strict=True) if a.effect_id != b.effect_id
        )
    return anomalies


def _authorised_by_step(state: _Build, payload: Mapping[str, JsonValue], branch_id: str) -> int:
    """The ``step_index`` of the confirming model turn, however the journal recorded it.

    Three sources in decreasing directness: the field itself, the entry the dispatch names by
    offset, and the branch's own confirming entry. Never the offset -- see
    :class:`LedgerRow`.
    """
    explicit = _as_int(payload, "authorised_by_step")
    if explicit >= 0:
        return explicit
    for field_name in ("confirming_entry_offset", "authorised_by_offset"):
        offset = _as_int(payload, field_name)
        confirming = state.by_offset.get(offset) if offset >= 0 else None
        if confirming is not None:
            step = confirming.step
            if step is not None:
                return step
    for resolution in state.resolutions.get(branch_id, ()):
        if _as_str(resolution, "status") == "confirmed":
            decision_step = _as_int(resolution, "decision_step")
            return decision_step if decision_step >= 0 else _as_int(resolution, "step")
    return -1


def _dead_letter_reason(payload: Mapping[str, JsonValue]) -> str:
    explicit = _as_str(payload, "reason")
    if explicit:
        return explicit
    error = _as_map(payload, "last_error")
    return _as_str(error, "message") or _as_str(error, "type") or "unknown"


def _stalls(state: _Build) -> tuple[Stall, ...]:
    out: list[Stall] = []
    for branch_id, resolutions in state.resolutions.items():
        for payload in resolutions:
            if _as_str(payload, "status") != "stalled":
                continue
            raw = _as_str(payload, "hazard") or _as_str(payload, "reason")
            node_id = _as_str(payload, "node_id") or _as_str(
                state.fork.get(branch_id, {}), "node_id"
            )
            out.append(
                Stall(
                    step=_as_int(payload, "step"),
                    hazard=_HAZARD_BY_VALUE.get(raw) or _HAZARD_BY_NAME.get(raw),
                    node_id=node_id,
                    raw=raw,
                )
            )
    return tuple(sorted(out, key=lambda stall: (stall.step, stall.slug, stall.node_id)))


def _tiers_from(raw: object) -> dict[int, tuple[int, int]]:
    """Per-tier counts out of a payload, tolerating anything an edited journal might hold."""
    if not isinstance(raw, dict):
        return {}
    out: dict[int, tuple[int, int]] = {}
    for tier, counts in raw.items():
        if not isinstance(counts, dict) or not isinstance(tier, str):
            continue
        if not (tier.isascii() and tier.isdigit()):
            continue
        samples = max(0, _as_int(counts, "samples", 0))
        out[int(tier)] = (min(samples, max(0, _as_int(counts, "hits", 0))), samples)
    return out


def _wasted_tokens(state: _Build) -> int:
    """Tokens spent on branches that produced nothing (Hard Rule 10).

    Summed from ``branch_resolved`` rather than read off ``run_finished.counters``: the
    counters are the runtime's self-report, and a ledger that copied them would corroborate
    the bookkeeping with the bookkeeping. It also keeps the signed payload independent of
    ``run_finished``, which carries the signature.
    """
    total = 0
    for branch_id, status in state.final_status.items():
        if status not in ("squashed", "stalled"):
            continue
        for payload in state.resolutions.get(branch_id, ()):
            if _as_str(payload, "status") == status:
                total += max(0, _as_int(payload, "wasted_tokens", 0))
    return total


def _post_write_span(state: _Build) -> int:
    total = 0
    for branch_id, status in state.final_status.items():
        if status != "retired":
            continue
        for payload in state.resolutions.get(branch_id, ()):
            if _as_str(payload, "status") == "retired":
                total += max(0, _as_int(payload, "post_write_span", 0))
    return total


# -- the signed payload ---------------------------------------------------------------------


def _row_payload(row: LedgerRow) -> JsonValue:
    return {
        "effect_id": row.effect_id,
        "call": decision_payload(row.call),
        "key": row.key,
        "nkey": row.nkey,
        "node_id": row.node_id,
        "step_index": row.step_index,
        "branch_id": row.branch_id,
        "branch_ord": row.branch_ord,
        "stage_index": row.stage_index,
        "dispatch_index": row.dispatch_index,
        "retire_seq": row.retire_seq,
        "authorised_by_step": row.authorised_by_step,
        "status": row.status,
        "reason": row.reason,
        "ack_hash": row.ack_hash,
        "compensates": row.compensates,
        "args_mismatch": row.args_mismatch,
    }


def ledger_payload(ledger: Ledger) -> Mapping[str, JsonValue]:
    """The material a signature covers: every row in full, every aggregate, the run's binding.

    Excludes ``signature`` by construction, and excludes everything derived from
    ``run_finished`` -- which carries the signature, so a payload field taken from it would be
    defined in terms of a value that does not exist yet. ``ack`` is covered by ``ack_hash``
    rather than in full: an ack is the world's payload and can legitimately carry a megabyte.
    """
    return {
        "v": _PAYLOAD_VERSION,
        "run_id": ledger.run_id,
        "replay_id": ledger.replay_id,
        "parent_run_id": ledger.parent_run_id,
        "journal_head": ledger.journal_head,
        "journal_entries": ledger.journal_entries,
        "n": len(ledger.rows),
        "rows": [_row_payload(row) for row in ledger.rows],
        "squashed_branches": ledger.squashed_branches,
        "discarded_effects": ledger.discarded_effects,
        "stalls": [
            {"step": stall.step, "hazard": stall.slug, "node_id": stall.node_id}
            for stall in ledger.stalls
        ],
        "reads_validated": {
            "fresh": ledger.reads_validated.fresh,
            "stale": ledger.reads_validated.stale,
            "unwitnessed": ledger.reads_validated.unwitnessed,
            "total": ledger.reads_validated.total,
        },
        "wasted_tokens": ledger.wasted_tokens,
        "speculative_reads_upstream": ledger.speculative_reads_upstream,
        "speculative_reads_charged": ledger.speculative_reads_charged,
        "speculation_disabled_reason": ledger.speculation_disabled_reason,
        "context_divergences": ledger.context_divergences,
        "context_checks": list(ledger.context_checks),
        "context_identity": ledger.context_identity,
        "reads_raced_drain": ledger.reads_raced_drain,
        "injected_blocks": ledger.injected_blocks,
        "alpha": ledger.alpha,
        "alpha_window": ledger.alpha_window,
        "alpha_hits": ledger.alpha_hits,
        "alpha_samples": ledger.alpha_samples,
        "alpha_by_tier": {
            str(tier): {"hits": hits, "samples": samples}
            for tier, (hits, samples) in sorted(ledger.alpha_by_tier.items())
        },
        "alpha_gate_unmeasured": ledger.alpha_gate_unmeasured,
        "dispatch_order_anomalies": ledger.dispatch_order_anomalies,
        "args_hash_mismatches": ledger.args_hash_mismatches,
        "post_write_span": ledger.post_write_span,
        "policy_events": [
            {"step": event.step, "event": event.event, "reason": event.reason}
            for event in ledger.policy_events
        ],
        "speculation": ledger.speculation,
        "drafters": list(ledger.drafters),
        "target_model": ledger.target_model,
        "adapter": ledger.adapter,
    }


def signed_bytes(ledger: Ledger) -> bytes:
    """Exactly what Ed25519 signs and verifies."""
    return SIGN_DOMAIN + canonical(ledger_payload(ledger))


# -- keys -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SigningKey:
    """A local Ed25519 key. No passphrase, by design -- see the module docstring."""

    key_id: str
    private: Ed25519PrivateKey

    @property
    def public_bytes(self) -> bytes:
        return self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def key_id_for(public_bytes: bytes) -> str:
    """A short, stable name for a public key. Sixteen hex characters."""
    return blake2b(public_bytes, digest_size=8).hexdigest()


def _keystore(store: Path | str | None) -> Path:
    return Path(store) if store is not None else DEFAULT_KEYSTORE


def load_or_create_key(store: Path | str | None = None) -> SigningKey:
    """Load the local signing key, minting one on first use.

    ``$SPECUNODE_SIGNING_KEY`` overrides the location. The file is written ``0600`` with no
    passphrase: Hard Rule 11 says no accounts and no hosted components, and a passphrase
    prompt would break CI and every non-interactive run for a key that anyone with read
    access to the directory could copy anyway. The threat this addresses is a ledger edited
    in transit, not a machine an attacker already holds.
    """
    override = os.environ.get(_KEY_ENV)
    path = Path(override) if override else _keystore(store) / "signing_key.pem"
    if path.exists():
        loaded = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(loaded, Ed25519PrivateKey):
            raise ValueError(f"{path} does not hold an Ed25519 private key")
        private = loaded
    else:
        private = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            private.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        path.chmod(0o600)
    key = SigningKey(key_id=key_id_for(_public_raw(private)), private=private)
    _publish_public_key(_keystore(store), key)
    return key


def _public_raw(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


def _publish_public_key(store: Path, key: SigningKey) -> None:
    """Record the public half in the trust set, so later ledgers from this machine verify.

    Trust here is trust-on-first-use over a local directory, and the origin check is only as
    strong as that: it distinguishes "signed by a key this store has seen" from "signed by a
    key that signs nothing else here". It is not authentication.
    """
    directory = store / "keys"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key.key_id}.pub").write_text(_b64(key.public_bytes) + "\n", encoding="utf-8")


def trusted_key_ids(
    store: Path | str | None = None,
    *,
    journal: Journal | None = None,
    run_id: str | None = None,
    extra: Iterable[str] = (),
) -> frozenset[str]:
    """Key ids the origin check accepts: the local keystore, the journal, and explicit ones."""
    found = set(extra)
    directory = _keystore(store) / "keys"
    if directory.is_dir():
        found.update(path.stem for path in directory.glob("*.pub"))
    if journal is not None and run_id is not None:
        for entry in journal.read(run_id, kinds=("run_finished",)):
            key_id = _as_str(entry.payload, "signing_key_id")
            if key_id:
                found.add(key_id)
    return frozenset(found)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_ledger(ledger: Ledger, key: SigningKey) -> Ledger:
    """Return a copy carrying the signature envelope.

    The envelope embeds the **full public key**, not only its id. That is what makes the
    integrity and origin checks separable: a ledger signed with a key nobody has seen can
    still be verified mathematically, so ``verify-ledger`` can say "these rows are
    self-consistent but the key is a stranger" instead of collapsing both forgeries into
    "cannot verify".
    """
    signature = key.private.sign(signed_bytes(replace(ledger, signature="")))
    envelope = f"ed25519:{key.key_id}:{_b64(key.public_bytes)}:{_b64(signature)}"
    return replace(ledger, signature=envelope)


# -- verification ---------------------------------------------------------------------------

VerifyReason = Literal[
    "OK",
    "MALFORMED",
    "UNSIGNED",
    "INTEGRITY",
    "ORIGIN",
    "CHAIN_BROKEN",
    "JOURNAL_MISMATCH",
]

_EXIT_CODES: Final[Mapping[str, int]] = {
    "OK": 0,
    "INTEGRITY": 2,
    "ORIGIN": 3,
    "CHAIN_BROKEN": 4,
    "JOURNAL_MISMATCH": 4,
    "MALFORMED": 5,
    "UNSIGNED": 6,
}


@dataclass(frozen=True, slots=True)
class LedgerVerification:
    """What ``specunode verify-ledger`` learned, and how far it got."""

    ok: bool
    reason: VerifyReason
    #: ``integrity`` means the rows are not the bytes that were signed, or do not reproduce
    #: from the journal. ``origin`` means the rows are internally consistent and the key is a
    #: stranger. Spec task 5.4 requires both forgeries to be rejected *with the reason
    #: stated*, and they have different remedies.
    category: Literal["ok", "malformed", "unsigned", "integrity", "origin"]
    exit_code: int
    key_id: str | None = None
    journal: Literal["verified", "absent", "not_checked"] = "not_checked"
    detail: str | None = None


def _fail(
    reason: VerifyReason,
    category: Literal["malformed", "unsigned", "integrity", "origin"],
    detail: str,
    *,
    key_id: str | None = None,
    journal: Literal["verified", "absent", "not_checked"] = "not_checked",
) -> LedgerVerification:
    return LedgerVerification(
        ok=False,
        reason=reason,
        category=category,
        exit_code=_EXIT_CODES[reason],
        key_id=key_id,
        journal=journal,
        detail=detail,
    )


def verify_ledger(
    ledger: Ledger,
    *,
    journal: Journal | None = None,
    trusted: Iterable[str] | None = None,
    store: Path | str | None = None,
) -> LedgerVerification:
    """Four checks in a fixed order: parse, integrity, origin, journal.

    The order is the substance. If origin were checked first, a ledger that is *both* edited
    *and* signed by an unknown key would report ``ORIGIN``, and the operator would never learn
    the rows were altered -- which is the finding that actually matters. Integrity first means
    check 2 only ever runs on rows already shown to be the ones that were signed, so "unknown
    key" is a statement about provenance rather than a shrug.

    What a pass does not establish: that the effects happened in the world (only that the
    runtime recorded them), who ran it (there are no accounts and the key is unencrypted), or
    that this is the newest ledger for the run. Only check 3's journal binding ties a ledger
    to a run at all.
    """
    if not ledger.signature:
        return _fail("UNSIGNED", "unsigned", "the ledger carries no signature")

    parts = ledger.signature.split(":")
    if len(parts) != 4 or parts[0] != "ed25519":
        return _fail(
            "MALFORMED",
            "malformed",
            "signature is not ed25519:<key_id>:<public_key>:<signature>",
        )
    _, key_id, public_b64, signature_b64 = parts
    try:
        public_raw = _unb64(public_b64)
        signature = _unb64(signature_b64)
        public = Ed25519PublicKey.from_public_bytes(public_raw)
    except (ValueError, TypeError) as exc:
        return _fail("MALFORMED", "malformed", f"signature envelope does not decode: {exc}")
    if key_id_for(public_raw) != key_id:
        # An envelope whose key id does not name its own embedded key is not a forgery the
        # origin check could catch: it would be looked up under a name the key does not have.
        return _fail(
            "MALFORMED",
            "malformed",
            f"envelope key id {key_id} does not name its embedded public key",
        )

    try:
        public.verify(signature, signed_bytes(replace(ledger, signature="")))
    except InvalidSignature:
        return _fail(
            "INTEGRITY",
            "integrity",
            "the rows are not the bytes this signature covers; they were edited after signing",
            key_id=key_id,
        )

    known = frozenset(trusted) if trusted is not None else trusted_key_ids(store)
    if key_id not in known:
        return _fail(
            "ORIGIN",
            "origin",
            f"key {key_id} signs no other ledger in this store; the rows are internally "
            "consistent, so they were signed by a key this store has never seen",
            key_id=key_id,
        )

    if journal is None:
        return LedgerVerification(
            ok=True,
            reason="OK",
            category="ok",
            exit_code=0,
            key_id=key_id,
            journal="not_checked",
            detail="no journal supplied; the run binding was not checked",
        )

    chain = journal.verify_chain(ledger.run_id)
    if chain.reason == "empty":
        # Claim nothing more. A ledger for a run this store does not hold is not evidence of
        # forgery, and reporting JOURNAL_MISMATCH would make "I moved the file" look like an
        # attack.
        return LedgerVerification(
            ok=True,
            reason="OK",
            category="ok",
            exit_code=0,
            key_id=key_id,
            journal="absent",
            detail=f"no entries for run {ledger.run_id} in this store",
        )
    if not chain.ok:
        return _fail(
            "CHAIN_BROKEN",
            "integrity",
            f"journal chain breaks at offset {chain.first_bad_offset}: {chain.detail}",
            key_id=key_id,
            journal="verified",
        )

    rebuilt = build_ledger(journal, ledger.run_id)
    if canonical(ledger_payload(rebuilt)) != canonical(ledger_payload(ledger)):
        return _fail(
            "JOURNAL_MISMATCH",
            "integrity",
            "the rows do not reproduce from this run's journal",
            key_id=key_id,
            journal="verified",
        )
    return LedgerVerification(
        ok=True,
        reason="OK",
        category="ok",
        exit_code=0,
        key_id=key_id,
        journal="verified",
    )


# -- rendering ------------------------------------------------------------------------------


def _short(identifier: str, *, ellipsis: str, width: int = 6) -> str:
    return identifier if len(identifier) <= width else identifier[:width] + ellipsis


def _render_call(call: ToolCall, *, ellipsis: str) -> str:
    body = ", ".join(
        f"{name}:{canonical(call.args[name]).decode('utf-8')}" for name in sorted(call.args)
    )
    text = f"{call.name} {{{body}}}"
    if len(text) > _MAX_EFFECT_CELL:
        return text[: _MAX_EFFECT_CELL - len(ellipsis)] + ellipsis
    return text


def _render_status(row: LedgerRow, *, ellipsis: str) -> str:
    if row.status == "DEAD_LETTER":
        return f"DEAD_LETTER {row.reason or 'unknown'}"
    if row.dry_run:
        # On the row, not only in the command's banner. A rendering is something people paste
        # into a ticket, and one that reads DISPATCHED for an effect nobody sent is a lie that
        # travels.
        return f"{row.status} (dry run: not sent)"
    acks = 0 if row.ack is None else 1
    return f"{row.status} ack={acks}"


def _columns(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> Iterator[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    for source in (headers, *rows):
        cells = [
            cell if index == len(widths) - 1 else cell.ljust(widths[index])
            for index, cell in enumerate(source)
        ]
        yield ("  ".join(cells)).rstrip()


def render_ledger(
    ledger: Ledger,
    *,
    short_ids: bool = False,
    normalised: bool = False,
    ascii_only: bool = False,
    equivalence_digest: str | None = None,
) -> str:
    """Demo 3's table and summary. Pure, and golden-tested.

    The run id renders in full by default so a line can be pasted straight into
    ``specunode replay``; ``short_ids`` gives section 2's abbreviated form.

    ``normalised`` renders the same table with the run id, the branch column and every field
    the equivalence relation strips removed, so a demo can *show* two identical ledgers rather
    than assert that they are.

    ``equivalence_digest`` is passed in rather than computed here. The relation lives in
    ``verify/equivalence.py``, which reads ledgers; computing it here would make the two
    modules import each other, and the cycle would have to be broken by duplicating the
    normalisation -- two definitions of the relation Hard Rule 9 rests on.
    """
    ellipsis = "..." if ascii_only else _ELLIPSIS
    if normalised:
        return _render_normalised(ledger, ellipsis=ellipsis)

    run_label = _short(ledger.run_id, ellipsis=ellipsis) if short_ids else ledger.run_id
    header = (
        f"EFFECT LEDGER  run {run_label}  "
        f"speculation={'on' if ledger.speculation else 'off'}  "
        f"drafters={'+'.join(ledger.drafters) or 'none'}  "
        f"target={ledger.target_model or 'unknown'}"
    )
    if ledger.replay_id is not None:
        replay = _short(ledger.replay_id, ellipsis=ellipsis) if short_ids else ledger.replay_id
        header += f"  replay {replay}"
    lines = [header]

    if ledger.rows:
        table = [
            [
                str(index),
                _render_call(row.call, ellipsis=ellipsis),
                row.key[:_KEY_PREFIX] + ellipsis,
                row.branch_label,
                f"step {row.authorised_by_step} "
                f"({'compensated' if row.status == 'COMPENSATED' else 'retired'})",
                _render_status(row, ellipsis=ellipsis),
            ]
            for index, row in enumerate(ledger.rows, start=1)
        ]
        width = max(len(cells[0]) for cells in table)
        for cells in table:
            cells[0] = cells[0].rjust(width)
        lines.extend(
            _columns(
                table, ["#".rjust(width), "effect", "key", "branch", "decided-by(step)", "status"]
            )
        )
    else:
        lines.append("(no effects reached the world)")

    lines.extend(_summary(ledger, ellipsis=ellipsis, equivalence_digest=equivalence_digest))
    return "\n".join(lines) + "\n"


def _summary(ledger: Ledger, *, ellipsis: str, equivalence_digest: str | None) -> list[str]:
    """The summary lines, all of them, every time.

    Printed even when zero. An omitted line reads as "not measured", and the distance between
    "no effects were discarded" and "nobody counted" is the whole difference between a receipt
    and a reassurance.
    """
    stall_detail = ""
    if ledger.stalls:
        shown = ", ".join(f'{stall.slug} node "{stall.node_id}"' for stall in ledger.stalls[:3])
        stall_detail = f" ({shown})"
    tally = ledger.reads_validated
    window = "-" if ledger.alpha_window is None or ledger.alpha_window < 0 else ledger.alpha_window
    if ledger.alpha is not None:
        alpha = f"{ledger.alpha:.2f}"
    elif ledger.alpha_samples:
        # Judged by the gate as "not enough evidence yet"; reported by the receipt as exactly
        # what it was. "n/a" here read as "nothing was measured" and that was never true.
        alpha = f"unjudged ({ledger.alpha_hits}/{ledger.alpha_samples} graded)"
    else:
        alpha = "n/a"
    if ledger.alpha_by_tier:
        # Which predictor the graded guesses belong to. Tier 0 is one by construction and is
        # excluded from the gate, so it appears here only when it was exercised at all.
        alpha += "  by tier: " + ", ".join(
            f"T{tier} {hits}/{samples}"
            for tier, (hits, samples) in sorted(ledger.alpha_by_tier.items())
        )
    gate = "  [gate inactive: unmeasured]" if ledger.alpha_gate_unmeasured else ""
    order = (
        "in program order"
        if ledger.dispatch_order_anomalies == 0
        else f"{ledger.dispatch_order_anomalies} effect(s) dispatched out of program order"
    )
    lines = [
        f"squashed branches: {ledger.squashed_branches}   "
        f"staged effects discarded: {ledger.discarded_effects}   "
        f"stalls: {len(ledger.stalls)}{stall_detail}",
        f"reads validated at retirement: {tally.fresh}/{tally.witnessed} fresh, "
        f"{tally.unwitnessed} unwitnessed, {tally.unreadable} unreadable   "
        f"speculative reads upstream: {ledger.speculative_reads_upstream} "
        f"({ledger.speculative_reads_charged} charged to the read budget)   "
        f"reads racing a drain: {ledger.reads_raced_drain}",
        f"wasted tokens: {ledger.wasted_tokens:,}   alpha (window {window}): {alpha}{gate}   "
        f"context divergences: {ledger.context_divergences}   "
        f"injected blocks: {ledger.injected_blocks}",
        f"dispatch order: {order}   args-hash mismatches: {ledger.args_hash_mismatches}",
    ]
    # Observations already feed the alpha line above; the rest are decisions, and the one
    # that switched speculation off is never allowed to scroll out of the first three.
    decisions = [e for e in ledger.policy_events if e.event != "alpha_observed"]
    closures = [e for e in decisions if e.event == "speculation_disabled"]
    others = [e for e in decisions if e.event != "speculation_disabled"]
    for event in closures + others[: max(0, 3 - len(closures))]:
        lines.append(f"policy: {event.event} - {event.reason}")
    lines.append(f"equivalence digest: {equivalence_digest or '(not computed)'}")
    signature = "(unsigned)"
    if ledger.signature:
        parts = ledger.signature.split(":")
        signature = f"ed25519:{parts[1]}" if len(parts) == 4 else "(malformed)"
    # "chain ok" is not printed here: this function never walked the chain, and a stamp for a
    # property nobody checked is worse than no stamp. verify_ledger is what checks it.
    head = ledger.journal_head[:_KEY_PREFIX] + ellipsis if ledger.journal_head else "none"
    lines.append(
        f"signature: {signature}   journal: {ledger.journal_entries} entries, head {head}   "
        f"context_identity: {ledger.context_identity}"
    )
    if not ledger.terminal:
        lines.append("run: incomplete - no run_finished entry in this journal")
    return lines


def _render_normalised(ledger: Ledger, *, ellipsis: str) -> str:
    """Exactly the fields the equivalence relation keeps, in the order it compares them."""
    lines = [f"EFFECT LEDGER (normalised)  n={len(ledger.rows)}"]
    if not ledger.rows:
        lines.append("(no effects reached the world)")
        return "\n".join(lines) + "\n"
    table = [
        [
            str(index),
            _render_call(row.call, ellipsis=ellipsis),
            f"{row.node_id}@{row.step_index}",
            f"auth {row.authorised_by_step}",
            row.status,
        ]
        for index, row in enumerate(ledger.rows, start=1)
    ]
    width = max(len(cells[0]) for cells in table)
    for cells in table:
        cells[0] = cells[0].rjust(width)
    lines.extend(_columns(table, ["#".rjust(width), "effect", "node@step", "decided-by", "status"]))
    return "\n".join(lines) + "\n"
