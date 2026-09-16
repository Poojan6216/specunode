"""The journal (spec task 0.3, Hard Rule 5)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specunode.canonical import JsonValue, canonical, chash
from specunode.journal.entries import (
    ENTRY_KINDS,
    PAYLOAD_VERSION,
    REQUIRED_FIELDS,
    InvalidEntry,
    entry_hash,
    genesis_prev_hash,
    validate_payload,
)
from specunode.journal.journal import Journal, JournalConcurrencyError

RUN = "01RUNAAAAAAAAAAAAAAAAAAAAA"


def payload_for(entry_kind: str, **extra: JsonValue) -> dict[str, JsonValue]:
    """A minimal payload that satisfies the kind's required fields."""
    filler: dict[str, JsonValue] = {"v": PAYLOAD_VERSION}
    for field in sorted(REQUIRED_FIELDS[entry_kind]):
        filler[field] = {"counters": 0} if field == "counters" else 0
    for field in ("branch_id", "mode", "status", "kind", "tool", "reason", "event", "role"):
        if field in REQUIRED_FIELDS[entry_kind]:
            filler[field] = "x"
    for field in ("ok", "deduped", "speculative", "reached_upstream"):
        if field in REQUIRED_FIELDS[entry_kind]:
            filler[field] = True
    for field in ("lineage", "patch"):
        if field in REQUIRED_FIELDS[entry_kind]:
            filler[field] = []
    for field in ("policy", "target", "decision", "last_error", "counters"):
        if field in REQUIRED_FIELDS[entry_kind]:
            filler[field] = {}
    for field in sorted(REQUIRED_FIELDS[entry_kind]):
        if field.endswith("_hash") or field in {"config_hash", "registry_hash", "key", "nkey"}:
            filler[field] = "h" * 8
    filler.update(extra)
    return filler


def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal.db")


# -- shape ------------------------------------------------------------------------------------


def test_there_are_exactly_fifteen_entry_kinds() -> None:
    assert len(ENTRY_KINDS) == 15
    assert len(set(ENTRY_KINDS)) == 15


def test_offsets_are_dense_from_zero(tmp_path: Path) -> None:
    j = journal(tmp_path)
    offsets = [j.append(RUN, "policy_event", payload_for("policy_event")) for _ in range(20)]
    assert offsets == list(range(20))


def test_two_runs_have_independent_offsets_and_chains(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for _ in range(3):
        j.append("run-a", "policy_event", payload_for("policy_event"))
        j.append("run-b", "policy_event", payload_for("policy_event"))
    assert [e.offset for e in j.read("run-a")] == [0, 1, 2]
    assert [e.offset for e in j.read("run-b")] == [0, 1, 2]
    assert j.verify_chain("run-a").ok and j.verify_chain("run-b").ok


def test_entries_are_stored_canonically(tmp_path: Path) -> None:
    j = journal(tmp_path)
    j.append(RUN, "policy_event", payload_for("policy_event", zzz=1, aaa=2))
    entry = next(iter(j.read(RUN)))
    assert entry.payload_json.encode("utf-8") == canonical(entry.payload)
    assert entry.payload_hash == chash(entry.payload)


def test_read_can_filter_by_kind_and_start_offset(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for index in range(6):
        kind = "policy_event" if index % 2 else "read_validated"
        j.append(RUN, kind, payload_for(kind))
    assert [e.offset for e in j.read(RUN, kinds=["policy_event"])] == [1, 3, 5]
    assert [e.offset for e in j.read(RUN, after=3)] == [4, 5]
    assert [e.offset for e in j.read(RUN)] == [0, 1, 2, 3, 4, 5], (
        "the default must not skip entry 0"
    )


def test_read_pages_through_more_entries_than_one_chunk(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for _ in range(50):
        j.append(RUN, "policy_event", payload_for("policy_event"))
    assert len(list(j.read(RUN, chunk=7))) == 50


def test_max_step_recovers_the_counter_a_resume_needs(tmp_path: Path) -> None:
    """Hard Rule 8: a key re-derived after a crash must equal the one derived before it."""
    j = journal(tmp_path)
    assert j.max_step(RUN) == -1
    for step in (0, 1, 2, 7, 5):
        j.append(RUN, "branch_resolved", payload_for("branch_resolved", step=step))
    assert j.max_step(RUN) == 7


# -- validation --------------------------------------------------------------------------------


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidEntry, match="unknown journal entry kind"):
        journal(tmp_path).append(RUN, "invented_kind", {"v": 1})


def test_a_payload_missing_a_required_field_is_refused() -> None:
    """A model_request with no request_hash makes ReplayDivergence undetectable."""
    payload = payload_for("model_request")
    del payload["request_hash"]
    with pytest.raises(InvalidEntry, match="request_hash"):
        validate_payload("model_request", payload)


def test_a_payload_without_a_version_is_refused() -> None:
    with pytest.raises(InvalidEntry, match="v=1"):
        validate_payload("policy_event", {"event": "x", "reason": "y"})


@pytest.mark.parametrize("column", ["run_id", "offset", "kind", "ts"])
def test_a_payload_may_not_repeat_a_column(column: str) -> None:
    with pytest.raises(InvalidEntry, match="must not repeat"):
        validate_payload("policy_event", payload_for("policy_event", **{column: "dup"}))


def test_a_payload_with_no_canonical_form_is_refused() -> None:
    with pytest.raises(InvalidEntry, match="canonical"):
        validate_payload("policy_event", payload_for("policy_event", bad=float("nan")))


# -- the hash chain ------------------------------------------------------------------------------


def test_the_genesis_hash_is_per_run(tmp_path: Path) -> None:
    assert genesis_prev_hash("a") != genesis_prev_hash("b")
    j = journal(tmp_path)
    j.append(RUN, "policy_event", payload_for("policy_event"))
    assert next(iter(j.read(RUN))).prev_hash == genesis_prev_hash(RUN)


def test_a_clean_chain_verifies(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for _ in range(100):
        j.append(RUN, "policy_event", payload_for("policy_event"))
    result = j.verify_chain(RUN)
    assert result.ok and result.entries == 100 and result.reason == "ok"


def test_an_empty_run_verifies_as_empty(tmp_path: Path) -> None:
    assert journal(tmp_path).verify_chain("nothing-here").reason == "empty"


def _tamper(path: Path, sql: str, params: tuple[object, ...] = ()) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def test_an_edited_payload_breaks_the_chain(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for _ in range(5):
        j.append(RUN, "policy_event", payload_for("policy_event"))
    edited = json.dumps({"v": 1, "event": "tampered", "reason": "x"}, separators=(",", ":"))
    _tamper(
        tmp_path / "journal.db", 'UPDATE entries SET payload_json=? WHERE "offset"=2', (edited,)
    )
    result = j.verify_chain(RUN)
    assert not result.ok and result.first_bad_offset == 2 and result.reason == "payload_mismatch"


def test_a_retyped_entry_breaks_the_chain(tmp_path: Path) -> None:
    """kind is inside entry_hash, so relabelling an entry cannot go unnoticed."""
    j = journal(tmp_path)
    for _ in range(5):
        j.append(RUN, "policy_event", payload_for("policy_event"))
    _tamper(
        tmp_path / "journal.db", 'UPDATE entries SET kind=? WHERE "offset"=1', ("read_validated",)
    )
    result = j.verify_chain(RUN)
    assert not result.ok and result.reason == "chain_break" and result.first_bad_offset == 2


def test_a_rewritten_timestamp_breaks_the_chain(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for _ in range(4):
        j.append(RUN, "policy_event", payload_for("policy_event"))
    _tamper(
        tmp_path / "journal.db",
        'UPDATE entries SET ts=? WHERE "offset"=0',
        ("1999-01-01T00:00:00.0Z",),
    )
    assert journal(tmp_path).verify_chain(RUN).reason == "chain_break"


def test_a_deleted_entry_shows_as_a_gap(tmp_path: Path) -> None:
    j = journal(tmp_path)
    for _ in range(5):
        j.append(RUN, "policy_event", payload_for("policy_event"))
    _tamper(tmp_path / "journal.db", 'DELETE FROM entries WHERE "offset"=2')
    result = j.verify_chain(RUN)
    assert not result.ok and result.reason == "gap" and result.first_bad_offset == 3


def test_non_canonical_stored_text_is_caught_even_though_its_hash_matches(tmp_path: Path) -> None:
    """A hand-edited journal can be valid JSON that is not the canonical encoding."""
    j = journal(tmp_path)
    j.append(RUN, "policy_event", payload_for("policy_event"))
    entry = next(iter(j.read(RUN)))
    pretty = json.dumps(dict(entry.payload), indent=2)  # same content, different bytes
    _tamper(
        tmp_path / "journal.db",
        'UPDATE entries SET payload_json=? WHERE "offset"=0',
        (pretty,),
    )
    result = j.verify_chain(RUN)
    assert not result.ok and result.reason == "non_canonical_payload"


def test_an_offset_collision_raises_rather_than_reordering(tmp_path: Path) -> None:
    j = journal(tmp_path)
    j.append(RUN, "policy_event", payload_for("policy_event"))
    _tamper(
        tmp_path / "journal.db",
        'INSERT INTO entries (run_id,"offset",kind,payload_json,payload_hash,prev_hash,ts) '
        "VALUES (?,?,?,?,?,?,?)",
        (RUN, 1, "policy_event", "{}", "h", "p", "t"),
    )
    with pytest.raises(JournalConcurrencyError):
        j.append(RUN, "policy_event", payload_for("policy_event"))


@given(st.lists(st.sampled_from(ENTRY_KINDS), min_size=1, max_size=40))
@settings(max_examples=60, deadline=None)
def test_any_sequence_of_kinds_chains(
    tmp_path_factory: pytest.TempPathFactory, kinds: list[str]
) -> None:
    path = tmp_path_factory.mktemp("chain") / "journal.db"
    j = Journal(path)
    for kind in kinds:
        j.append(RUN, kind, payload_for(kind))
    result = j.verify_chain(RUN)
    assert result.ok and result.entries == len(kinds)


def test_entry_hash_covers_every_stored_column() -> None:
    base = {
        "run_id": "r",
        "offset": 3,
        "kind": "policy_event",
        "ts": "t",
        "payload_hash": "ph",
        "prev_hash": "pv",
    }
    baseline = entry_hash(**base)  # type: ignore[arg-type]
    for field in base:
        changed = {**base, field: ("z" if isinstance(base[field], str) else 99)}
        got = entry_hash(**changed)  # type: ignore[arg-type]
        assert got != baseline, f"{field} is not covered by entry_hash"
