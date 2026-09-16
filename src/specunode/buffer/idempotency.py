"""Idempotency keys: one preimage, three derivations (Hard Rule 8).

    key = blake2b(run_id | branch_lineage | node_id | step_index | tool_name | canonical(args))

The spec's formula, with the framing made explicit: the preimage is a canonical JSON *object*,
never a concatenation, because ``("ab", "c")`` and ``("a", "bc")`` concatenate to the same
bytes and two different effects must never share a key.

Three values come out of that one preimage, and confusing them is the subtle failure this
module exists to prevent:

``key`` -- lineage-bearing, internal
    Indexes the store buffer, names a :class:`StagedEffect`, and appears in the journal and the
    ledger. The lineage is what keeps two siblings' staged rows apart *before* resolution:
    they occupy the same ``step_index`` and the same ``node_id`` by construction, so without it
    their entries alias and discarding the loser would drop the winner's row.

``nkey`` -- lineage-free, run-scoped
    The dedupe table's primary key, and the idempotency token handed to the tool adapter. It
    must be lineage-free because the same logical effect legitimately arrives down different
    paths: a stalled branch's effect is discarded and re-staged by the sequential re-execution
    under a new branch id, and a resume re-mints branch ids entirely. Keyed on ``key``, every
    one of those would re-dispatch -- a second charge on the same card.

``ekey`` -- lineage-free *and* run-free, comparison only
    Used by :func:`~specunode.verify.equivalence.normalise_for_equivalence` and nowhere else.
    It carries a different domain prefix so that it cannot be mistaken for a dispatch key even
    if it were passed somewhere one was expected.

Lineage contributes no uniqueness among *dispatched* effects, because at most one branch per
program position ever retires. That is exactly why stripping it is sound for dedupe and for
equivalence, and exactly why it must stay in ``key``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import blake2b
from typing import NewType

from specunode.canonical import JsonValue, canonical

__all__ = [
    "EQUIV_DOMAIN",
    "KEY_DOMAIN",
    "DedupeKey",
    "EquivalenceKey",
    "IdempotencyKey",
    "dedupe_key",
    "equivalence_key",
    "idempotency_key",
    "key_preimage",
]

#: Domain separators, so a key can never collide with a journal entry hash or a ledger
#: signature preimage even when the same bytes are fed to all three.
KEY_DOMAIN = b"specunode/idempotency/v1\x00"
EQUIV_DOMAIN = b"specunode/equiv-key/v1\x00"

IdempotencyKey = NewType("IdempotencyKey", str)
DedupeKey = NewType("DedupeKey", str)
EquivalenceKey = NewType("EquivalenceKey", str)


def key_preimage(
    *,
    run_id: str,
    lineage: Sequence[str],
    node_id: str,
    step_index: int,
    tool_name: str,
    args: Mapping[str, JsonValue],
) -> bytes:
    """The canonical bytes every key derivation hashes.

    Public, because the property tests and ``docs/replay.md`` assert on these exact bytes: a
    key nobody can reproduce by hand is a key nobody can audit.
    """
    return canonical(
        {
            "run_id": run_id,
            "lineage": list(lineage),
            "node_id": node_id,
            "step_index": step_index,
            "tool": tool_name,
            "args": args,
        }
    )


def _digest(domain: bytes, preimage: bytes) -> str:
    return blake2b(domain + preimage, digest_size=32).hexdigest()


def idempotency_key(
    *,
    run_id: str,
    lineage: Sequence[str],
    node_id: str,
    step_index: int,
    tool_name: str,
    args: Mapping[str, JsonValue],
) -> IdempotencyKey:
    """Hard Rule 8's key, verbatim. Internal: indexes the store buffer, never leaves."""
    return IdempotencyKey(
        _digest(
            KEY_DOMAIN,
            key_preimage(
                run_id=run_id,
                lineage=lineage,
                node_id=node_id,
                step_index=step_index,
                tool_name=tool_name,
                args=args,
            ),
        )
    )


def dedupe_key(
    *,
    run_id: str,
    node_id: str,
    step_index: int,
    tool_name: str,
    args: Mapping[str, JsonValue],
) -> DedupeKey:
    """The value the dedupe table and the tool adapter see. Lineage-free by construction."""
    return DedupeKey(
        _digest(
            KEY_DOMAIN,
            key_preimage(
                run_id=run_id,
                lineage=(),
                node_id=node_id,
                step_index=step_index,
                tool_name=tool_name,
                args=args,
            ),
        )
    )


def equivalence_key(
    *, node_id: str, step_index: int, tool_name: str, args: Mapping[str, JsonValue]
) -> EquivalenceKey:
    """Comparison only (Hard Rule 9). Run-free, lineage-free, and never dispatchable."""
    return EquivalenceKey(
        _digest(
            EQUIV_DOMAIN,
            key_preimage(
                run_id="",
                lineage=(),
                node_id=node_id,
                step_index=step_index,
                tool_name=tool_name,
                args=args,
            ),
        )
    )
