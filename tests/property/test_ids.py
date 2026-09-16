"""ULID properties (spec section 5: ULID for runs, branches, steps, effects)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specunode.ids import MAX_TIMESTAMP_MS, new_ulid, ulid_from, ulid_timestamp_ms

EXAMPLES = settings(max_examples=2000, deadline=None)


@given(
    st.integers(min_value=0, max_value=MAX_TIMESTAMP_MS),
    st.binary(min_size=10, max_size=10),
)
@EXAMPLES
def test_timestamp_round_trips(timestamp_ms: int, randomness: bytes) -> None:
    assert ulid_timestamp_ms(ulid_from(timestamp_ms, randomness)) == timestamp_ms


@given(st.integers(min_value=0, max_value=MAX_TIMESTAMP_MS), st.binary(min_size=10, max_size=10))
@EXAMPLES
def test_shape_is_26_crockford_characters(timestamp_ms: int, randomness: bytes) -> None:
    value = ulid_from(timestamp_ms, randomness)
    assert len(value) == 26
    assert set(value) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_minted_ulids_sort_in_creation_order() -> None:
    """Needed so effects staged in the same millisecond still render in stage order."""
    minted = [new_ulid() for _ in range(5000)]
    assert minted == sorted(minted)
    assert len(set(minted)) == len(minted)


def test_timestamps_earlier_than_later_ones_sort_first() -> None:
    early = ulid_from(1, b"\xff" * 10)
    late = ulid_from(2, b"\x00" * 10)
    assert early < late


@pytest.mark.parametrize("bad", ["", "short", "I" * 26, "x" * 27])
def test_malformed_ulids_raise(bad: str) -> None:
    with pytest.raises(ValueError):
        ulid_timestamp_ms(bad)


def test_out_of_range_inputs_raise() -> None:
    with pytest.raises(ValueError):
        ulid_from(MAX_TIMESTAMP_MS + 1, b"\x00" * 10)
    with pytest.raises(ValueError):
        ulid_from(0, b"\x00" * 9)
