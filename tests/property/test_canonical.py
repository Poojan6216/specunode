"""Property tests for the canonical form (spec task 0.2).

Everything downstream -- branch resolution (Hard Rule 4), idempotency keys (Hard Rule 8),
the hash chain in the journal, prompt identity (Hard Rule 13) -- is only as sound as these
properties. The spec asks for 2,000 examples per property; that is what runs here.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specunode.canonical import (
    MAX_DEPTH,
    CanonicalError,
    canonical,
    chash,
    from_canonical,
)
from specunode.core.decision import (
    FreeText,
    Route,
    Structured,
    ToolCall,
    decision_key,
    decision_payload,
    decisions_equal,
    from_payload,
)

EXAMPLES = settings(max_examples=2000, deadline=None)

# Floats restricted to the finite ones: NaN and the infinities have no canonical form by
# design, and are tested separately for rejection rather than mixed in here.
finite_floats = st.floats(allow_nan=False, allow_infinity=False)

json_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63),
    finite_floats,
    st.text(max_size=40),
)

json_values = st.recursive(
    json_leaves,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(max_size=12), children, max_size=6),
    ),
    max_leaves=25,
)


@given(json_values)
@EXAMPLES
def test_json_round_trip_is_a_fixed_point(value: object) -> None:
    """canonical(json.loads(canonical(v))) == canonical(v).

    Without this, a value that went through the journal (which stores JSON text) would hash
    differently from the value that went in, and every key derived from it on resume would
    miss the dedupe table.
    """
    once = canonical(value)
    assert canonical(from_canonical(once)) == once


#: Matches one JSON string literal, so structural whitespace can be inspected without
#: tripping over whitespace that legitimately lives *inside* a string value.
_STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')


@given(json_values)
@EXAMPLES
def test_canonical_is_valid_compact_json(value: object) -> None:
    text = canonical(value).decode("utf-8")
    json.loads(text)  # parses
    structure = _STRING_LITERAL.sub("", text)
    assert not any(char.isspace() for char in structure), (
        f"insignificant whitespace in canonical output: {text!r}"
    )


@given(st.dictionaries(st.text(max_size=8), json_leaves, min_size=1, max_size=8))
@EXAMPLES
def test_key_order_never_changes_the_hash(mapping: dict[str, object]) -> None:
    """A dict built in a different insertion order is the same JSON value."""
    reordered = dict(reversed(list(mapping.items())))
    assert chash(mapping) == chash(reordered)


@given(st.text(max_size=40))
@EXAMPLES
def test_nfd_and_nfc_hash_equal(text: str) -> None:
    """The same text in two Unicode normal forms is the same text."""
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    assert chash({"k": nfc}) == chash({"k": nfd})
    assert chash({nfc: 1}) == chash({nfd: 1})


@given(st.text(max_size=20))
@EXAMPLES
def test_whitespace_in_the_source_text_is_irrelevant(text: str) -> None:
    """Parsing pretty-printed JSON and compact JSON yields the same canonical bytes."""
    value = {"a": text, "b": [1, 2]}
    pretty = json.loads(json.dumps(value, indent=4))
    compact = json.loads(json.dumps(value, separators=(",", ":")))
    assert canonical(pretty) == canonical(compact)


def test_negative_zero_folds_to_zero() -> None:
    assert chash({"a": -0.0}) == chash({"a": 0.0})
    assert canonical(-0.0) == b"0.0"


@given(finite_floats)
@EXAMPLES
def test_every_finite_float_survives_the_round_trip(value: float) -> None:
    parsed = json.loads(canonical(value))
    assert math.isclose(parsed, value, rel_tol=0, abs_tol=0) or parsed == value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinities_are_rejected(bad: float) -> None:
    with pytest.raises(CanonicalError):
        canonical(bad)
    with pytest.raises(CanonicalError):
        canonical({"a": [bad]})


def test_int_and_float_are_different_values() -> None:
    """Python says 1 == 1.0. A billing API does not. Hard Rule 4 follows JSON, not Python."""
    assert canonical({"amount": 1}) != canonical({"amount": 1.0})
    assert canonical({"flag": True}) != canonical({"flag": 1})


def test_keys_colliding_only_after_normalisation_are_rejected() -> None:
    """Silently dropping one of two keys would let two different calls share a hash."""
    with pytest.raises(CanonicalError, match="collide after NFC"):
        canonical({"é": 1, "é": 2})


def test_non_json_types_are_rejected() -> None:
    for bad in [b"bytes", {1: "int key"}, object(), {"k": object()}, 1 + 2j]:
        with pytest.raises(CanonicalError):
            canonical(bad)  # type: ignore[arg-type]


def test_depth_limit_and_cycles_raise_cleanly() -> None:
    deep: object = 1
    for _ in range(MAX_DEPTH + 5):
        deep = [deep]
    with pytest.raises(CanonicalError, match="MAX_DEPTH"):
        canonical(deep)

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(CanonicalError, match="MAX_DEPTH"):
        canonical(cycle)


# --------------------------------------------------------------------------------------
# Decisions and the resolution relation (Hard Rule 4)
# --------------------------------------------------------------------------------------

decisions = st.one_of(
    st.builds(ToolCall, st.text(max_size=12), st.dictionaries(st.text(max_size=8), json_leaves)),
    st.builds(Route, st.text(max_size=12)),
    st.builds(Structured, st.text(max_size=12), json_values),
    st.builds(FreeText.of, st.text(max_size=40)),
)


@given(decisions)
@EXAMPLES
def test_resolution_is_reflexive_except_for_free_text(decision: object) -> None:
    """for all d, resolve(d, d) is CONFIRMED except FreeText."""
    same = from_payload(decision_payload(decision))  # type: ignore[arg-type]
    if isinstance(decision, FreeText):
        assert not decisions_equal(decision, same)
        assert not decisions_equal(decision, decision)
    else:
        assert decisions_equal(decision, same)  # type: ignore[arg-type]


@given(decisions, decisions)
@EXAMPLES
def test_different_decisions_never_resolve_equal(a: object, b: object) -> None:
    """for all d != e, SQUASHED. Equality is decided on canonical bytes, not Python ==."""
    if decision_key(a) != decision_key(b):  # type: ignore[arg-type]
        assert not decisions_equal(a, b)  # type: ignore[arg-type]


@given(decisions)
@EXAMPLES
def test_decision_payload_round_trips(decision: object) -> None:
    rebuilt = from_payload(decision_payload(decision))  # type: ignore[arg-type]
    assert decision_key(rebuilt) == decision_key(decision)  # type: ignore[arg-type]


@given(st.text(max_size=40))
@EXAMPLES
def test_two_free_texts_never_resolve_equal(text: str) -> None:
    """A free-text node is a barrier, not a speculation that always succeeds."""
    assert not decisions_equal(FreeText.of(text), FreeText.of(text))


def test_a_tool_call_never_resolves_against_another_kind() -> None:
    assert not decisions_equal(ToolCall("x", {}), Route("x"))
    assert not decisions_equal(Route("x"), Structured("x", None))
