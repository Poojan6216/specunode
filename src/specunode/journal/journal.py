"""The journal: every model output and tool result, durable before the runtime acts on it.

Hard Rule 5. Replay never calls a model; it reads this. Resume never re-asks; it reads this.
The ledger is rendered from this. So the properties that matter are unglamorous ones:

**One entry is one commit is one fsync.** Not batched. ~100 fsyncs on a 30-step run costs
tens of milliseconds against multi-second model turns, and batching would open a crash window
in exchange for nothing. Because the unit of commit is exactly one row, "the last committed
transaction" and "the last entry" are the same thing, which is what makes the entry at the
moment of a crash either wholly present or wholly absent -- never half-written.

**Ordering is by ``offset``, never by ``ts``.** No query in this codebase contains
``ORDER BY ts``. Timestamps are for humans; two entries written in the same microsecond, or
written either side of an NTP step, still have a total order because the writer assigns it.

**One writer per file, per process.** SQLite in WAL mode admits one writer at a time. Rather
than scatter ``SQLITE_BUSY`` retries through the runtime, a process-global registry gives each
database file exactly one connection and one single-threaded executor; twenty concurrent runs
sharing a journal queue on it in FIFO order and cannot collide (spec task 5.3). Cross-process
contention -- a resume, the CLI, the MCP proxy, a chaos subprocess -- is what ``busy_timeout``
is for, and with single-row transactions the write lock is held for well under a millisecond.

**The fsync never runs on the event loop.** :meth:`Journal.append` blocks, and is the
signature section 7 specifies; every caller inside the runtime uses
:meth:`Journal.append_async`, which awaits the executor instead of stalling every other
branch for the duration of a disk flush.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from specunode.canonical import JsonValue, canonical, chash
from specunode.journal.entries import (
    ENTRY_KINDS,
    Entry,
    entry_hash,
    genesis_prev_hash,
    validate_payload,
)

__all__ = [
    "ChainVerification",
    "Journal",
    "JournalBusy",
    "JournalConcurrencyError",
    "JournalConfigError",
    "JournalError",
    "JournalWriteError",
    "close_all_writers",
]

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_INSERT_ENTRY = (
    'INSERT INTO entries (run_id, "offset", kind, payload_json, payload_hash, prev_hash, ts) '
    "VALUES (:run_id, :offset, :kind, :payload_json, :payload_hash, :prev_hash, :ts)"
)
_SELECT_HEAD = (
    'SELECT "offset", kind, payload_hash, prev_hash, ts FROM entries '
    'WHERE run_id = :run_id ORDER BY "offset" DESC LIMIT 1'
)
_SELECT_CHUNK = (
    'SELECT run_id, "offset", kind, payload_json, payload_hash, prev_hash, ts FROM entries '
    'WHERE run_id = :run_id AND "offset" > :after ORDER BY "offset" ASC LIMIT :limit'
)
_SELECT_CHUNK_KINDS = (
    'SELECT run_id, "offset", kind, payload_json, payload_hash, prev_hash, ts FROM entries '
    'WHERE run_id = :run_id AND "offset" > :after AND kind IN ({placeholders}) '
    'ORDER BY "offset" ASC LIMIT :limit'
)


class JournalError(RuntimeError):
    """Base class for journal failures."""


class JournalConfigError(JournalError):
    """The database cannot provide the durability this design rests on."""


class JournalWriteError(JournalError):
    """An append failed. The entry is either wholly absent or wholly present."""


class JournalConcurrencyError(JournalError):
    """Another writer claimed this offset. Never retried silently."""


class JournalBusy(JournalError):
    """A cross-process lock could not be acquired within the busy timeout."""


@dataclass(frozen=True, slots=True)
class ChainHead:
    offset: int
    entry_hash: str


@dataclass(frozen=True, slots=True)
class ChainVerification:
    """The result of walking a run's hash chain."""

    ok: bool
    run_id: str
    entries: int
    first_bad_offset: int | None = None
    reason: Literal[
        "ok",
        "empty",
        "gap",
        "chain_break",
        "payload_mismatch",
        "non_canonical_payload",
        "unknown_kind",
    ] = "ok"
    detail: str | None = None


class Backend(Protocol):
    """The two-method surface the journal needs from a database."""

    def execute(
        self, sql: str, params: Mapping[str, JsonValue] | None = None
    ) -> sqlite3.Cursor: ...
    def begin_immediate(self) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


def _load_schema_statements() -> list[str]:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    return [statement for statement in text.split("\n;\n") if statement.strip()]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class _SqliteBackend:
    """SQLite in WAL mode, autocommit, one statement per transaction."""

    def __init__(self, path: Path, *, fullfsync: bool = False, read_only: bool = False) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None is autocommit on the 3.11 floor: each statement is its own
        # implicit transaction, so the WAL commit frame is written and fsynced before
        # execute() returns. (3.12+ spells this autocommit=True; 3.11 is the floor.)
        self._conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False, timeout=5.0
        )
        self._conn.row_factory = sqlite3.Row
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise JournalConfigError(
                f"{path} could not be opened in WAL mode (got {mode!r}). WAL is unavailable on "
                "some network filesystems, and without it one entry is not one durable commit."
            )
        self._conn.execute("PRAGMA synchronous=FULL")
        level = self._conn.execute("PRAGMA synchronous").fetchone()[0]
        if int(level) != 2:
            raise JournalConfigError(
                f"{path}: synchronous is {level}, not FULL(2); append() would return before "
                "the entry was durable, which is the one thing Hard Rule 5 forbids"
            )
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=OFF")
        if fullfsync:
            # On macOS a plain fsync does not flush the drive's own write cache.
            self._conn.execute("PRAGMA fullfsync=ON")
        if read_only:
            self._conn.execute("PRAGMA query_only=ON")
        else:
            for statement in _load_schema_statements():
                self._conn.execute(statement)

    def execute(self, sql: str, params: Mapping[str, JsonValue] | None = None) -> sqlite3.Cursor:
        return self._conn.execute(sql, dict(params) if params else {})

    def begin_immediate(self) -> None:
        # IMMEDIATE, never deferred: a deferred read-then-write upgrade raises
        # SQLITE_BUSY_SNAPSHOT, which busy_timeout does not retry.
        self._conn.execute("BEGIN IMMEDIATE")

    def commit(self) -> None:
        self._conn.execute("COMMIT")

    def rollback(self) -> None:
        self._conn.execute("ROLLBACK")

    def close(self) -> None:
        self._conn.close()


class _JournalWriter:
    """One connection and one thread per database file, shared by every run in the process."""

    def __init__(self, path: Path, *, fullfsync: bool = False) -> None:
        self.path = path
        self._backend = _SqliteBackend(path, fullfsync=fullfsync)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="specunode-journal")
        self._head: dict[str, ChainHead] = {}
        self._max_step: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- runs on the writer thread ---------------------------------------------------------

    def _seed_head(self, run_id: str) -> ChainHead | None:
        row = self._backend.execute(_SELECT_HEAD, {"run_id": run_id}).fetchone()
        if row is None:
            return None
        return ChainHead(
            offset=int(row["offset"]),
            entry_hash=entry_hash(
                run_id=run_id,
                offset=int(row["offset"]),
                kind=str(row["kind"]),
                ts=str(row["ts"]),
                payload_hash=str(row["payload_hash"]),
                prev_hash=str(row["prev_hash"]),
            ),
        )

    def _append(self, run_id: str, kind: str, payload_json: str, payload_hash: str) -> int:
        head = self._head.get(run_id)
        if head is None:
            head = self._seed_head(run_id)
        offset = 0 if head is None else head.offset + 1
        prev = genesis_prev_hash(run_id) if head is None else head.entry_hash
        ts = _utc_now()
        try:
            self._backend.execute(
                _INSERT_ENTRY,
                {
                    "run_id": run_id,
                    "offset": offset,
                    "kind": kind,
                    "payload_json": payload_json,
                    "payload_hash": payload_hash,
                    "prev_hash": prev,
                    "ts": ts,
                },
            )
        except sqlite3.IntegrityError as exc:
            # Someone else owns this offset. Never silently re-seed and retry: that would
            # reorder two entries whose order is the thing the chain exists to fix.
            self._head.pop(run_id, None)
            raise JournalConcurrencyError(
                f"offset {offset} of run {run_id} is already taken; another writer holds this "
                f"journal ({self.path})"
            ) from exc
        except sqlite3.OperationalError as exc:
            self._head.pop(run_id, None)
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise JournalBusy(f"{self.path}: {exc}") from exc
            raise JournalWriteError(f"{self.path}: {exc}") from exc
        except Exception as exc:  # the head cache must be poisoned whatever failed
            self._head.pop(run_id, None)
            raise JournalWriteError(f"{self.path}: {exc}") from exc

        # Only now, after the commit returned, does the in-memory head advance.
        self._head[run_id] = ChainHead(
            offset=offset,
            entry_hash=entry_hash(
                run_id=run_id,
                offset=offset,
                kind=kind,
                ts=ts,
                payload_hash=payload_hash,
                prev_hash=prev,
            ),
        )
        return offset

    # -- called from any thread --------------------------------------------------------------

    def submit_append(self, run_id: str, kind: str, payload_json: str, payload_hash: str) -> int:
        future = self._executor.submit(self._append, run_id, kind, payload_json, payload_hash)
        return future.result()

    async def submit_append_async(
        self, run_id: str, kind: str, payload_json: str, payload_hash: str
    ) -> int:
        future = self._executor.submit(self._append, run_id, kind, payload_json, payload_hash)
        return await asyncio.wrap_future(future)

    def run(self, fn: object, *args: object) -> object:
        return self._executor.submit(fn, *args).result()  # type: ignore[arg-type]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._backend.close()


_writers: dict[str, _JournalWriter] = {}
_writers_lock = threading.Lock()


def _writer_for(path: Path, *, fullfsync: bool = False) -> _JournalWriter:
    key = os.path.realpath(path)
    with _writers_lock:
        writer = _writers.get(key)
        if writer is None:
            writer = _JournalWriter(Path(key), fullfsync=fullfsync)
            _writers[key] = writer
        return writer


def close_all_writers() -> None:
    """Close every open journal writer. Tests and CLI shutdown call this."""
    with _writers_lock:
        for writer in _writers.values():
            writer.close()
        _writers.clear()


class Journal:
    """Append-only, hash-chained, durable before return."""

    def __init__(self, path: Path | str, *, fullfsync: bool = False) -> None:
        self.path = Path(path)
        self._writer = _writer_for(self.path, fullfsync=fullfsync)

    # -- writing -----------------------------------------------------------------------------

    def _prepare(self, kind: str, payload: Mapping[str, JsonValue]) -> tuple[str, str]:
        """Validate and canonicalise off the writer thread."""
        validate_payload(kind, payload)
        return canonical(payload).decode("utf-8"), chash(payload)

    def append(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        """Append one entry. Returns its offset, and only after it is durable (Hard Rule 5).

        Blocking. Inside the runtime use :meth:`append_async`, so that one branch's fsync
        does not stall every other branch on the event loop.
        """
        payload_json, payload_hash = self._prepare(kind, payload)
        return self._writer.submit_append(run_id, kind, payload_json, payload_hash)

    async def append_async(self, run_id: str, kind: str, payload: Mapping[str, JsonValue]) -> int:
        payload_json, payload_hash = self._prepare(kind, payload)
        return await self._writer.submit_append_async(run_id, kind, payload_json, payload_hash)

    # -- reading -----------------------------------------------------------------------------

    def _reader(self) -> _SqliteBackend:
        # A separate read-only connection, never the writer's. query_only rather than a
        # mode=ro URI, because a read-only URI connection fails when the -shm file has to be
        # created.
        return _SqliteBackend(self.path, read_only=True)

    def read(
        self, run_id: str, after: int = -1, kinds: Sequence[str] | None = None, chunk: int = 512
    ) -> Iterator[Entry]:
        """Iterate a run's entries in offset order, returning those with ``offset > after``.

        A cursor, not a snapshot: each chunk is a fresh short query, so iteration sees entries
        appended while it runs and holds no long read transaction. The journal is append-only,
        so a cursor can gain entries but never lose or change one.

        ``after`` defaults to ``-1``, not to section 7's ``0``. Offsets are dense from zero, so
        an exclusive ``after=0`` would silently skip the first entry of every run -- which for
        a replay means starting one entry late, and for ``run_started`` in particular means
        losing the config and tool registry the replay is supposed to check against. Logged as
        a signature change in the Progress Log.
        """
        backend = self._reader()
        try:
            last = after
            while True:
                if kinds:
                    placeholders = ", ".join(f":k{i}" for i in range(len(kinds)))
                    sql = _SELECT_CHUNK_KINDS.format(placeholders=placeholders)
                    params: dict[str, JsonValue] = {f"k{i}": k for i, k in enumerate(kinds)}
                else:
                    sql, params = _SELECT_CHUNK, {}
                params.update({"run_id": run_id, "after": last, "limit": chunk})
                rows = backend.execute(sql, params).fetchall()
                if not rows:
                    return
                for row in rows:
                    last = int(row["offset"])
                    yield Entry(
                        run_id=str(row["run_id"]),
                        offset=last,
                        kind=str(row["kind"]),
                        payload=json.loads(row["payload_json"]),
                        payload_hash=str(row["payload_hash"]),
                        prev_hash=str(row["prev_hash"]),
                        ts=str(row["ts"]),
                        payload_json=str(row["payload_json"]),
                    )
                if len(rows) < chunk:
                    return
        finally:
            backend.close()

    def last_offset(self, run_id: str) -> int | None:
        backend = self._reader()
        try:
            row = backend.execute(_SELECT_HEAD, {"run_id": run_id}).fetchone()
            return None if row is None else int(row["offset"])
        finally:
            backend.close()

    def max_step(self, run_id: str) -> int:
        """The highest ``step`` any entry recorded, or -1.

        This is how the step counter survives a crash (Hard Rule 8): an idempotency key
        re-derived after a resume must equal the one derived before it, and the counter it
        depends on therefore cannot start again from zero.
        """
        highest = -1
        for entry in self.read(run_id):
            step = entry.step
            if step is not None and step > highest:
                highest = step
        return highest

    def runs(self) -> list[str]:
        backend = self._reader()
        try:
            rows = backend.execute("SELECT DISTINCT run_id FROM entries", {}).fetchall()
            return sorted(str(row["run_id"]) for row in rows)
        finally:
            backend.close()

    # -- verification --------------------------------------------------------------------------

    def verify_chain(self, run_id: str) -> ChainVerification:
        """Walk the chain, stopping at the first break and saying where it is."""
        expected_prev = genesis_prev_hash(run_id)
        count = 0
        for entry in self.read(run_id):
            if entry.offset != count:
                return ChainVerification(
                    False,
                    run_id,
                    count,
                    entry.offset,
                    "gap",
                    f"expected offset {count}, found {entry.offset}",
                )
            recomputed = chash(entry.payload)
            if recomputed != entry.payload_hash:
                return ChainVerification(
                    False,
                    run_id,
                    count,
                    entry.offset,
                    "payload_mismatch",
                    f"payload hashes to {recomputed}, row says {entry.payload_hash}",
                )
            if entry.kind not in ENTRY_KINDS:
                return ChainVerification(
                    False, run_id, count, entry.offset, "unknown_kind", entry.kind
                )
            # The bytes on disk, against the canonical encoding of what they parse to.
            # This is what catches a hand-edited journal whose text is valid JSON but not
            # canonical -- the hash would still match, because the hash is over the parse.
            if entry.payload_json.encode("utf-8") != canonical(entry.payload):
                return ChainVerification(
                    False,
                    run_id,
                    count,
                    entry.offset,
                    "non_canonical_payload",
                    "stored payload text is not the canonical encoding of its own content",
                )
            if entry.prev_hash != expected_prev:
                return ChainVerification(
                    False,
                    run_id,
                    count,
                    entry.offset,
                    "chain_break",
                    f"prev_hash {entry.prev_hash} does not follow offset {entry.offset - 1}",
                )
            expected_prev = entry_hash(
                run_id=entry.run_id,
                offset=entry.offset,
                kind=entry.kind,
                ts=entry.ts,
                payload_hash=entry.payload_hash,
                prev_hash=entry.prev_hash,
            )
            count += 1
        if count == 0:
            return ChainVerification(True, run_id, 0, None, "empty")
        return ChainVerification(True, run_id, count, None, "ok")

    def close(self) -> None:
        """Journals share a process-global writer; closing one closes none of the others."""
