"""Canonical form and content hashing.

Hard Rule 4 (branch resolution is exact canonical equality) and Hard Rule 8 (idempotency
keys are deterministic and stable) both rest on this module. Two values that a tool would
treat as the same must produce the same bytes on every machine, interpreter version and run;
two values that differ in any way a tool would notice must not.

The canonical form is JSON with:

* object keys sorted by Unicode code point, *after* NFC normalisation
* no insignificant whitespace
* all strings NFC-normalised
* floats rendered by :func:`repr` (Python's shortest round-tripping representation)
* ``-0.0`` folded to ``0.0``
* NaN and the infinities rejected -- they are not JSON, and they are not decisions

``canonical`` is a fixed point of a JSON round trip: for every value it accepts,
``canonical(json.loads(canonical(v))) == canonical(v)``.

Two traps this module exists to close:

* Python's ``==`` says ``1 == 1.0`` and ``True == 1``. JSON does not, and neither does a
  billing API being handed ``{"amount": 1}`` versus ``{"amount": 1.0}``. Resolution compares
  canonical bytes, never ``==``; see :func:`specunode.core.decision.decisions_equal`.
* ``"e\\u0301"`` (NFD) and ``"\\u00e9"`` (NFC) are the same text to a user and different
  bytes to a hash. They are normalised to one form here, and two keys that collide *because*
  of that normalisation are rejected rather than silently merged.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from hashlib import blake2b
from typing import TypeAlias

__all__ = [
    "CanonicalError",
    "JsonValue",
    "canonical",
    "chash",
    "chash_bytes",
    "from_canonical",
]

JsonValue: TypeAlias = (
    "bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue] | None"
)

#: Guard against unbounded recursion, including self-referential containers, which would
#: otherwise surface as a bare ``RecursionError`` from deep inside the encoder.
MAX_DEPTH = 100


class CanonicalError(ValueError):
    """A value has no canonical form.

    Raised for non-JSON types, NaN, the infinities, non-string object keys, keys that
    collide only after NFC normalisation, strings containing unpaired surrogates, and
    structures nested past :data:`MAX_DEPTH`.
    """


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _encode_str(text: str) -> str:
    normalised = _norm(text)
    try:
        normalised.encode("utf-8")
    except UnicodeEncodeError as exc:  # unpaired surrogate
        raise CanonicalError(
            f"string is not encodable as UTF-8 and has no canonical form: {exc}"
        ) from exc
    # ensure_ascii=False keeps non-ASCII as raw UTF-8: shorter, still deterministic, and
    # readable in the journal. Control characters are still escaped by the encoder.
    return json.dumps(normalised, ensure_ascii=False)


def _encode_float(value: float) -> str:
    if math.isnan(value):
        raise CanonicalError("NaN has no canonical form")
    if math.isinf(value):
        raise CanonicalError(f"{value!r} has no canonical form")
    if value == 0.0:
        # True for both 0.0 and -0.0; folding them is what the spec asks for, and it keeps
        # the form a JSON round-trip fixed point.
        return "0.0"
    return repr(value)


def _encode(value: object, depth: int, out: list[str]) -> None:
    if depth > MAX_DEPTH:
        raise CanonicalError(f"value nests deeper than MAX_DEPTH={MAX_DEPTH} (or contains a cycle)")

    if value is None:
        out.append("null")
        return
    # bool before int: bool is a subclass of int, and `True` is not `1` in JSON.
    if value is True:
        out.append("true")
        return
    if value is False:
        out.append("false")
        return
    if isinstance(value, int):
        out.append(str(value))
        return
    if isinstance(value, float):
        out.append(_encode_float(value))
        return
    if isinstance(value, str):
        out.append(_encode_str(value))
        return
    if isinstance(value, Mapping):
        items: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise CanonicalError(
                    f"object keys must be strings, got {type(raw_key).__name__}: {raw_key!r}"
                )
            key = _norm(raw_key)
            if key in items:
                raise CanonicalError(
                    f"keys {raw_key!r} and an earlier key collide after NFC normalisation; "
                    "refusing to silently drop one"
                )
            items[key] = raw_value
        out.append("{")
        for index, key in enumerate(sorted(items)):
            if index:
                out.append(",")
            out.append(_encode_str(key))
            out.append(":")
            _encode(items[key], depth + 1, out)
        out.append("}")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise CanonicalError(
            "bytes have no JSON canonical form; encode them explicitly (e.g. base64) first"
        )
    if isinstance(value, Sequence):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _encode(item, depth + 1, out)
        out.append("]")
        return
    raise CanonicalError(f"{type(value).__name__} is not a JSON value: {value!r}")


def canonical(obj: JsonValue) -> bytes:
    """Return the canonical UTF-8 bytes of a JSON value.

    Raises :class:`CanonicalError` for anything that has no canonical form. Never returns
    a partial result: the whole value is encoded or the call raises.
    """
    out: list[str] = []
    _encode(obj, 0, out)
    return "".join(out).encode("utf-8")


def chash(obj: JsonValue) -> str:
    """blake2b-256 hex digest of :func:`canonical` bytes (Hard Rule 8)."""
    return chash_bytes(canonical(obj))


def chash_bytes(data: bytes) -> str:
    """blake2b-256 hex digest of bytes that are already canonical."""
    return blake2b(data, digest_size=32).hexdigest()


def from_canonical(data: bytes) -> JsonValue:
    """Parse canonical bytes back into a JSON value.

    The inverse of :func:`canonical` up to the normalisations it performs: a value that
    survived ``canonical`` re-encodes to identical bytes after this round trip.
    """
    parsed: JsonValue = json.loads(data.decode("utf-8"))
    return parsed
