"""ULIDs for runs, branches, steps and effects.

A ULID is 128 bits -- a 48-bit millisecond timestamp followed by 80 bits of randomness --
rendered as 26 Crockford base32 characters. Two properties earn it its place here over a
UUID4: the text form sorts lexicographically in creation order, which makes a journal or a
ledger readable without a join, and the timestamp is recoverable for diagnostics.

Identifiers minted here are *fresh*, so they are never a source of truth that replay depends
on. Anything replay must reproduce (idempotency keys, step indices) is derived from journaled
content, not from an id; see :mod:`specunode.buffer.idempotency`.
"""

from __future__ import annotations

import os
import threading
import time

__all__ = ["MAX_TIMESTAMP_MS", "new_ulid", "ulid_from", "ulid_timestamp_ms"]

# Crockford base32: no I, L, O or U, so a ULID cannot be misread or accidentally profane.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {char: index for index, char in enumerate(_ALPHABET)}
_ULID_LENGTH = 26
_RANDOM_BITS = 80
_RANDOM_BYTES = _RANDOM_BITS // 8
_MAX_RANDOM = (1 << _RANDOM_BITS) - 1
MAX_TIMESTAMP_MS = (1 << 48) - 1

_lock = threading.Lock()
_last_ms = -1
_last_random = 0


def _encode(value: int) -> str:
    chars = [""] * _ULID_LENGTH
    for position in range(_ULID_LENGTH - 1, -1, -1):
        chars[position] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def ulid_from(timestamp_ms: int, randomness: bytes) -> str:
    """Build a ULID from an explicit timestamp and 10 bytes of randomness.

    Deterministic, so tests and fixtures can pin ids without patching the clock.
    """
    if not 0 <= timestamp_ms <= MAX_TIMESTAMP_MS:
        raise ValueError(f"timestamp_ms out of range for a ULID: {timestamp_ms}")
    if len(randomness) != _RANDOM_BYTES:
        raise ValueError(f"randomness must be {_RANDOM_BYTES} bytes, got {len(randomness)}")
    return _encode((timestamp_ms << _RANDOM_BITS) | int.from_bytes(randomness, "big"))


def new_ulid() -> str:
    """Mint a fresh ULID, strictly increasing within this process.

    Within one millisecond the randomness is incremented rather than redrawn, so ids minted
    in a tight loop still sort in creation order -- which matters when a branch forks several
    effects in the same tick and the ledger has to render them in stage order.
    """
    global _last_ms, _last_random
    now_ms = time.time_ns() // 1_000_000
    with _lock:
        if now_ms == _last_ms:
            if _last_random >= _MAX_RANDOM:
                # 2^80 ids in one millisecond is not reachable; step the clock rather than
                # wrap and break monotonicity.
                now_ms += 1
                _last_ms = now_ms
                _last_random = int.from_bytes(os.urandom(_RANDOM_BYTES), "big")
            else:
                _last_random += 1
        else:
            if now_ms < _last_ms:
                # The wall clock went backwards (NTP step). Keep monotonicity by holding the
                # last millisecond and stepping randomness instead.
                now_ms = _last_ms
                _last_random += 1
            else:
                _last_ms = now_ms
                _last_random = int.from_bytes(os.urandom(_RANDOM_BYTES), "big")
        value = (now_ms << _RANDOM_BITS) | (_last_random & _MAX_RANDOM)
    return _encode(value)


def ulid_timestamp_ms(ulid: str) -> int:
    """Recover the millisecond timestamp embedded in a ULID."""
    if len(ulid) != _ULID_LENGTH:
        raise ValueError(f"not a ULID: {ulid!r}")
    value = 0
    for char in ulid.upper():
        try:
            value = (value << 5) | _DECODE[char]
        except KeyError as exc:
            raise ValueError(f"not a ULID, bad character {char!r}: {ulid!r}") from exc
    return value >> _RANDOM_BITS
