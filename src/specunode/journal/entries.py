"""Journal entry kinds, the hash chain, and what each payload must carry.

Fifteen kinds, frozen. Extension happens by adding optional payload fields, never by adding
a kind and never by adding a column -- a reader that meets an unknown kind cannot know
whether ignoring it is safe, so it must not have to decide.

The required-field table below is small but load-bearing. A ``model_request`` missing its
``request_hash`` makes :class:`~specunode.journal.replay.ReplayDivergence` undetectable; an
``effect_dispatched`` missing ``authorised_by_offset`` makes the ledger unable to say which
model decision authorised an effect, which is the one question the ledger exists to answer.
Those are silent failures at exactly the moments that matter, so they are caught at append.

Two integers appear throughout and are never conflated:

``step``
    the per-run decision index -- Hard Rule 8's ``step_index``, section 7's ``fork_step``.
    Recovered on resume as the maximum over the run's entries, which is what makes an
    idempotency key re-derived after a crash equal the one derived before it.
``offset``
    position in the journal. Only fields ending in ``_offset`` hold one.

A speculative run and a sequential run journal different numbers of entries, so offsets can
never align the two. ``step`` can, and that is why the equivalence and context-equivalence
tests are written against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import blake2b
from typing import Final

from specunode.canonical import JsonValue, canonical, chash

__all__ = [
    "ENTRY_KINDS",
    "PAYLOAD_VERSION",
    "REQUIRED_FIELDS",
    "Entry",
    "EntryKind",
    "InvalidEntry",
    "entry_hash",
    "genesis_prev_hash",
    "validate_payload",
]

PAYLOAD_VERSION: Final = 1

#: Domain separator, so a journal hash can never be confused with an idempotency key or a
#: ledger signature preimage even if the same bytes were fed to all three.
_GENESIS_DOMAIN: Final = b"specunode.journal.v1\x00"

EntryKind = str

ENTRY_KINDS: Final[tuple[str, ...]] = (
    "run_started",
    "model_request",
    "model_response",
    "tool_request",
    "tool_result",
    "branch_forked",
    "branch_resolved",
    "effect_staged",
    "effect_dispatched",
    "effect_dead_lettered",
    "effect_discarded",
    "state_delta_applied",
    "read_validated",
    "policy_event",
    "run_finished",
)

#: The fields without which a later phase cannot do its job. Not a full schema: payloads
#: carry more than this, and optional fields are how the format grows.
REQUIRED_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "run_started": frozenset({"mode", "config_hash", "policy", "registry_hash", "target"}),
    # request_hash is Hard Rule 13's prompt identity and ReplayModel's divergence check.
    "model_request": frozenset({"step", "branch_id", "role", "request_hash", "model"}),
    # decision is what the gate resolves against; text carries the FreeText preimage.
    "model_response": frozenset({"step", "branch_id", "request_hash", "decision", "speculative"}),
    # program_order is the ordering key Rule 13 assembles results by -- never completion order.
    "tool_request": frozenset(
        {"step", "branch_id", "call_id", "tool", "args_hash", "effect", "mode", "program_order"}
    ),
    "tool_result": frozenset({"step", "branch_id", "call_id", "ok", "reached_upstream"}),
    "branch_forked": frozenset({"branch_id", "lineage", "fork_step", "predicted_hash", "tier"}),
    "branch_resolved": frozenset({"branch_id", "step", "status"}),
    # key_inputs lets normalise_for_equivalence re-derive the key from journaled facts alone.
    "effect_staged": frozenset(
        {"effect_id", "branch_id", "step", "tool", "args_hash", "key", "nkey", "stage_index"}
    ),
    # dispatch_index records the order effects actually LEFT, which is not the same fact as
    # stage_index. A ledger ordered by stage_index would put a reversed drain back into stage
    # order and quietly pass the equivalence test's "dispatch order reversed" planted bug.
    "effect_dispatched": frozenset(
        {
            "effect_id",
            "branch_id",
            "nkey",
            "stage_index",
            "dispatch_index",
            "authorised_by_offset",
            "deduped",
        }
    ),
    "effect_dead_lettered": frozenset({"effect_id", "branch_id", "nkey", "attempts", "last_error"}),
    "effect_discarded": frozenset({"branch_id", "count", "reason"}),
    "state_delta_applied": frozenset({"branch_id", "step", "patch", "result_state_hash"}),
    "read_validated": frozenset({"branch_id", "step", "fresh", "stale", "unwitnessed", "total"}),
    "policy_event": frozenset({"event", "reason"}),
    "run_finished": frozenset({"ok", "status", "counters"}),
}

assert set(REQUIRED_FIELDS) == set(ENTRY_KINDS), "every entry kind needs a required-field set"


class InvalidEntry(ValueError):
    """A payload cannot be journaled as written."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One row of the journal."""

    run_id: str
    offset: int
    kind: str
    payload: Mapping[str, JsonValue]
    payload_hash: str
    prev_hash: str
    ts: str
    #: The bytes actually stored, kept so verification can check that the stored text is
    #: itself canonical. Re-encoding the parsed payload and comparing it to itself is a
    #: check that can never fail, which is worse than no check at all.
    payload_json: str = ""

    @property
    def step(self) -> int | None:
        value = self.payload.get("step")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def branch_id(self) -> str | None:
        value = self.payload.get("branch_id")
        return value if isinstance(value, str) else None


def genesis_prev_hash(run_id: str) -> str:
    """``prev_hash`` of a run's first entry.

    A domain-separated constant rather than NULL or the empty string, so that "this is the
    start of run R" is itself a claim the chain makes, and an entry cannot be moved to the
    head of a different run without detection.
    """
    return blake2b(_GENESIS_DOMAIN + run_id.encode("utf-8"), digest_size=32).hexdigest()


def entry_hash(
    *, run_id: str, offset: int, kind: str, ts: str, payload_hash: str, prev_hash: str
) -> str:
    """The link value the next entry stores as its ``prev_hash``.

    Covers every stored column but ``payload_json`` (whose content is covered transitively by
    ``payload_hash``). Including ``kind`` and ``offset`` is what stops an entry being retyped
    or reordered without breaking the chain. It is deliberately not a column: it is a pure
    function of the stored ones, so the table keeps exactly the columns the spec names and
    the chain is still verifiable from them.
    """
    return chash(
        {
            "run_id": run_id,
            "offset": offset,
            "kind": kind,
            "ts": ts,
            "payload_hash": payload_hash,
            "prev_hash": prev_hash,
        }
    )


def validate_payload(kind: str, payload: Mapping[str, JsonValue]) -> None:
    """Check a payload before it is written. Raises :class:`InvalidEntry`."""
    if kind not in REQUIRED_FIELDS:
        raise InvalidEntry(
            f"unknown journal entry kind {kind!r}; the fifteen kinds are frozen and extension "
            "happens through optional payload fields"
        )
    version = payload.get("v")
    if version != PAYLOAD_VERSION:
        raise InvalidEntry(
            f"{kind} payload must carry v={PAYLOAD_VERSION}, got {version!r}; a reader that "
            "cannot tell which version it is looking at cannot tell what is missing"
        )
    for reserved in ("run_id", "offset", "kind", "ts"):
        if reserved in payload:
            raise InvalidEntry(
                f"{kind} payload must not repeat the column {reserved!r}; two copies of one "
                "fact can disagree, and the column is the one the chain covers"
            )
    missing = REQUIRED_FIELDS[kind] - set(payload)
    if missing:
        raise InvalidEntry(f"{kind} payload is missing required field(s): {sorted(missing)}")
    try:
        canonical(payload)
    except ValueError as exc:
        raise InvalidEntry(f"{kind} payload has no canonical form: {exc}") from exc
