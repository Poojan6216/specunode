"""What a node decides, and the only relation that resolves a branch.

A *decision* is the unit a speculation is made about. The target model's real decision at
step *i* is the ground truth that retires or squashes every branch forked on a prediction
for step *i* (Hard Rule 4).

Four kinds, and the fourth is a barrier:

``ToolCall``
    a tool name plus canonicalised arguments
``Route``
    a conditional-edge label
``Structured``
    a schema id plus a JSON value
``FreeText``
    prose. It cannot be predicted token-for-token, so it cannot be confirmed by equality,
    so nothing speculates on it. :func:`decisions_equal` returns ``False`` for every pair of
    ``FreeText`` values *including two copies of the same text* -- that asymmetry is the
    point, and it is what makes a free-text node a speculation barrier rather than a
    speculation that always succeeds.

**``==`` is not the resolution relation.** The dataclasses keep structural equality so they
behave in sets and dicts, but Python says ``1 == 1.0`` and ``True == 1`` while a payment API
handed ``{"amount": 1}`` versus ``{"amount": 1.0}`` does not. Resolution compares canonical
bytes. Always call :func:`decisions_equal`; never ``a == b``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from specunode.canonical import JsonValue, canonical, chash, chash_bytes

__all__ = [
    "Decision",
    "DecisionKind",
    "FreeText",
    "Route",
    "Structured",
    "ToolCall",
    "decision_key",
    "decision_payload",
    "decisions_equal",
    "from_payload",
    "is_barrier",
]

DecisionKind: TypeAlias = Literal["tool_call", "route", "structured", "free_text"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request to run ``name`` with ``args``."""

    name: str
    args: Mapping[str, JsonValue]

    kind: DecisionKind = "tool_call"


@dataclass(frozen=True, slots=True)
class Route:
    """A conditional-edge label chosen by the graph."""

    label: str

    kind: DecisionKind = "route"


@dataclass(frozen=True, slots=True)
class Structured:
    """A schema-tagged structured output."""

    schema_id: str
    value: JsonValue

    kind: DecisionKind = "structured"


@dataclass(frozen=True, slots=True)
class FreeText:
    """Prose. A speculation barrier: it never resolves equal to anything, ever.

    Only the hash lives here; the text itself lives in the journal, so that a decision
    object stays small and cheap to compare while the run stays fully replayable.
    """

    content_hash: str

    kind: DecisionKind = "free_text"

    @staticmethod
    def of(text: str) -> FreeText:
        """Hash ``text`` (NFC-normalised) into a :class:`FreeText`."""
        return FreeText(chash_bytes(unicodedata.normalize("NFC", text).encode("utf-8")))


Decision: TypeAlias = ToolCall | Route | Structured | FreeText


def decision_payload(decision: Decision) -> JsonValue:
    """The JSON value that stands for ``decision`` in the journal and in a hash.

    ``kind`` is part of the payload so that decisions of different kinds can never collide
    into the same canonical bytes.
    """
    match decision:
        case ToolCall(name=name, args=args):
            return {"kind": "tool_call", "name": name, "args": args}
        case Route(label=label):
            return {"kind": "route", "label": label}
        case Structured(schema_id=schema_id, value=value):
            return {"kind": "structured", "schema_id": schema_id, "value": value}
        case FreeText(content_hash=content_hash):
            return {"kind": "free_text", "content_hash": content_hash}


def from_payload(payload: Mapping[str, JsonValue]) -> Decision:
    """Rebuild a decision from :func:`decision_payload` output, for replay and the ledger."""
    kind = payload.get("kind")
    match kind:
        case "tool_call":
            name = payload["name"]
            args = payload["args"]
            if not isinstance(name, str) or not isinstance(args, Mapping):
                raise ValueError(f"malformed tool_call decision payload: {payload!r}")
            return ToolCall(name=name, args=args)
        case "route":
            label = payload["label"]
            if not isinstance(label, str):
                raise ValueError(f"malformed route decision payload: {payload!r}")
            return Route(label=label)
        case "structured":
            schema_id = payload["schema_id"]
            if not isinstance(schema_id, str):
                raise ValueError(f"malformed structured decision payload: {payload!r}")
            return Structured(schema_id=schema_id, value=payload["value"])
        case "free_text":
            content_hash = payload["content_hash"]
            if not isinstance(content_hash, str):
                raise ValueError(f"malformed free_text decision payload: {payload!r}")
            return FreeText(content_hash=content_hash)
        case _:
            raise ValueError(f"unknown decision kind {kind!r}")


def decision_key(decision: Decision) -> str:
    """A stable content hash, used to dedupe candidate predictions by canonical form."""
    return chash(decision_payload(decision))


def is_barrier(decision: Decision) -> bool:
    """True iff nothing may speculate on this decision (Hard Rule 4)."""
    return isinstance(decision, FreeText)


def decisions_equal(predicted: Decision, actual: Decision) -> bool:
    """Exact canonical equality -- the *only* branch-resolution relation (Hard Rule 4).

    Never semantic similarity, never fuzzy argument matching, never an LLM judge. Two
    :class:`FreeText` values never compare equal, including two copies of the same text,
    because free text is a barrier rather than a speculation that trivially succeeds.
    """
    if is_barrier(predicted) or is_barrier(actual):
        return False
    if predicted.kind != actual.kind:
        return False
    return canonical(decision_payload(predicted)) == canonical(decision_payload(actual))
