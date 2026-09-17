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
import contextlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

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
    "Claim",
    "DispatchClaim",
    "Journal",
    "JournalBusy",
    "JournalConcurrencyError",
    "JournalConfigError",
    "JournalError",
    "JournalWriteError",
    "PendingClaim",
    "close_all_writers",
]

_T = TypeVar("_T")

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
_SELECT_CLAIM = (
    "SELECT nkey, idem_key, effect_id, branch_id, tool, status, last_outcome, attempt, "
    "ack_json, entry_offset FROM effect_dispatch WHERE run_id = :run_id AND nkey = :nkey"
)
# ON CONFLICT DO NOTHING is spelled identically on SQLite >= 3.24 and on Postgres, which is
# why it is the only conflict construct this codebase uses.
_INSERT_CLAIM = (
    "INSERT INTO effect_dispatch (run_id, nkey, idem_key, effect_id, branch_id, tool, status, "
    "last_outcome, attempt, claimed_at) VALUES (:run_id, :nkey, :idem_key, :effect_id, "
    ":branch_id, :tool, 'in_flight', 'unknown', :attempt, :claimed_at) "
    "ON CONFLICT (run_id, nkey) DO NOTHING"
)
_UPDATE_NOT_SENT = (
    "UPDATE effect_dispatch SET last_outcome = 'not_sent', attempt = :attempt "
    "WHERE run_id = :run_id AND nkey = :nkey"
)
_UPDATE_SETTLED = (
    "UPDATE effect_dispatch SET status = :status, last_outcome = 'settled', "
    "ack_json = :ack_json, entry_offset = :entry_offset, attempt = :attempt, "
    "settled_at = :settled_at WHERE run_id = :run_id AND nkey = :nkey"
)
_SELECT_UNRESOLVED = (
    "SELECT nkey, idem_key, effect_id, branch_id, tool, status, last_outcome, attempt "
    "FROM effect_dispatch WHERE run_id = :run_id AND status = 'in_flight'"
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


class Claim(Enum):
    """What a dispatch claim found."""

    #: Nobody else has this key. We own the send.
    OWNED = "owned"
    #: Already acked. Reuse the recorded ack and send nothing.
    ALREADY_DISPATCHED = "already_dispatched"
    #: A previous attempt demonstrably never left the process, or was dead-lettered and a
    #: resume is retrying it. Sending again is not a duplicate.
    RETRY_SAFE = "retry_safe"
    #: A previous attempt may or may not have taken effect upstream. The two-generals
    #: boundary: the caller decides using the tool's declared idempotency, and if it has not
    #: declared, the honest answer is to dead-letter and let a human look.
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class DispatchClaim:
    outcome: Claim
    attempt: int = 0
    ack: JsonValue = None

    @property
    def may_send(self) -> bool:
        return self.outcome in (Claim.OWNED, Claim.RETRY_SAFE)


@dataclass(frozen=True, slots=True)
class PendingClaim:
    run_id: str
    nkey: str
    idem_key: str
    effect_id: str
    branch_id: str
    tool: str
    attempt: int = 1

    def as_params(self, claimed_at: str) -> dict[str, JsonValue]:
        return {
            "run_id": self.run_id,
            "nkey": self.nkey,
            "idem_key": self.idem_key,
            "effect_id": self.effect_id,
            "branch_id": self.branch_id,
            "tool": self.tool,
            "attempt": self.attempt,
            "claimed_at": claimed_at,
        }


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


#: SQLite takes ``:name`` parameters and psycopg 3 takes ``%(name)s``. The DML is written once
#: with the former and rewritten once at import time for the latter, so there is one copy of
#: every statement and the two backends cannot drift apart in what they execute.
_PARAM = re.compile(r"(?<![:\w]):([a-z_][a-z0-9_]*)")


def _to_postgres(sql: str) -> str:
    return _PARAM.sub(r"%(\1)s", sql)


class _PostgresBackend:
    """Postgres 16, sharing ``schema.sql`` verbatim.

    Autocommit, so one statement is one commit is one flush, matching the SQLite path's
    discipline. ``synchronous_commit`` is set explicitly per session because a server
    configured ``off`` would silently discard the durability the whole design rests on, and the
    failure would look like a journal that loses its last few entries under load.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise JournalConfigError(
                "the postgres journal needs the optional extra: pip install 'specunode[postgres]'"
            ) from exc
        from psycopg.rows import dict_row

        # Every query in this module indexes its result by column name (``row["offset"]``,
        # ``row["kind"]``). psycopg returns tuples by default, so without this the backend
        # raises ``TypeError`` on the first row it reads -- which is where it would have failed
        # had anything ever actually connected it.
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        self._conn.execute("SET synchronous_commit = on")
        self._bootstrap(psycopg)

    def _bootstrap(self, psycopg: Any) -> None:
        """Load ``schema.sql``, tolerating another connection loading it at the same moment.

        Postgres's ``CREATE TABLE/INDEX IF NOT EXISTS`` is **not** race-safe: two connections
        running it concurrently collide in the system catalogues and one gets a
        ``UniqueViolation`` from ``pg_type`` or ``pg_class``. And this runs on *every*
        connection -- the writer opens one and every read opens another -- so two processes
        starting against a fresh database mostly failed, with a raw driver exception rather
        than a :class:`JournalError`. SQLite never showed it because its write lock serialises
        the same DDL.

        A lost race means somebody else created the object, which is the outcome this wanted.
        Anything else is re-raised.
        """
        for statement in _load_schema_statements():
            try:
                self._conn.execute(statement)
            except psycopg.errors.UniqueViolation:
                # Another connection created it between our IF NOT EXISTS check and our insert
                # into the catalogue. Idempotent by intent, so this is success.
                self._conn.execute("ROLLBACK")
            except psycopg.errors.DuplicateTable:
                self._conn.execute("ROLLBACK")
            except psycopg.errors.DuplicateObject:
                self._conn.execute("ROLLBACK")

    def execute(self, sql: str, params: Mapping[str, JsonValue] | None = None) -> Any:
        return self._conn.execute(_to_postgres(sql), dict(params) if params else {})

    def begin_immediate(self) -> None:
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.execute("COMMIT")

    def rollback(self) -> None:
        self._conn.execute("ROLLBACK")

    def close(self) -> None:
        self._conn.close()


#: Schemes that mean "this is a Postgres DSN, not a filesystem path".
_POSTGRES_SCHEMES = ("postgresql://", "postgres://")


def is_postgres_dsn(location: Path | str) -> bool:
    """True when this names a Postgres server rather than a file.

    Needed because ``Journal`` takes one argument for both. It used to coerce whatever it was
    given with ``Path(...)``, so a DSN became a relative filename and opened a perfectly good
    SQLite database -- including in the test named "the same journal on Postgres", which passed
    without ever opening a connection.
    """
    return isinstance(location, str) and location.startswith(_POSTGRES_SCHEMES)


def _backend_for(
    location: Path | str, *, fullfsync: bool = False, read_only: bool = False
) -> _SqliteBackend | _PostgresBackend:
    if is_postgres_dsn(location):
        # Postgres has no separate read-only handle here: a second connection to the same
        # server is a second connection, and ``query_only`` is a SQLite concept.
        return _PostgresBackend(str(location))
    return _SqliteBackend(Path(location), fullfsync=fullfsync, read_only=read_only)


def _integrity_errors() -> tuple[type[BaseException], ...]:
    """Every "a unique constraint said no" class, across the backends that are wired.

    ``sqlite3.IntegrityError`` alone was enough while only SQLite could ever run. psycopg
    raises its own, and catching only SQLite's turns a concurrent-offset collision -- the case
    the hash chain exists to refuse -- into an unhandled exception that escapes as a generic
    write failure, losing the diagnosis.
    """
    errors: list[type[BaseException]] = [sqlite3.IntegrityError]
    try:  # pragma: no cover - depends on the extras installed
        import psycopg

        errors.append(psycopg.errors.IntegrityError)
    except ImportError:
        pass
    return tuple(errors)


def _operational_errors() -> tuple[type[BaseException], ...]:
    """Every "the database could not do that right now" class."""
    errors: list[type[BaseException]] = [sqlite3.OperationalError]
    try:  # pragma: no cover - depends on the extras installed
        import psycopg

        errors.append(psycopg.errors.OperationalError)
    except ImportError:
        pass
    return tuple(errors)


#: Resolved once at import; the set of installed drivers does not change at run time.
_INTEGRITY_ERRORS = _integrity_errors()
_OPERATIONAL_ERRORS = _operational_errors()


class _JournalWriter:
    """One connection and one thread per database file, shared by every run in the process."""

    def __init__(self, path: Path | str, *, fullfsync: bool = False) -> None:
        self.path = path
        self._backend = _backend_for(path, fullfsync=fullfsync)
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
        except _INTEGRITY_ERRORS as exc:
            # Someone else owns this offset. Never silently re-seed and retry: that would
            # reorder two entries whose order is the thing the chain exists to fix.
            self._head.pop(run_id, None)
            raise JournalConcurrencyError(
                f"offset {offset} of run {run_id} is already taken; another writer holds this "
                f"journal ({self.path})"
            ) from exc
        except _OPERATIONAL_ERRORS as exc:
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

    def submit(self, fn: Callable[[], _T]) -> _T:
        return self._executor.submit(fn).result()

    async def submit_async(self, fn: Callable[[], _T]) -> _T:
        return await asyncio.wrap_future(self._executor.submit(fn))

    # -- dispatch deduplication (Hard Rule 8), all on the writer thread ---------------------

    def _claim(self, claim: PendingClaim) -> DispatchClaim:
        params: dict[str, JsonValue] = {"run_id": claim.run_id, "nkey": claim.nkey}
        row = self._backend.execute(_SELECT_CLAIM, params).fetchone()
        if row is None:
            try:
                self._backend.execute(_INSERT_CLAIM, claim.as_params(_utc_now()))
            except _INTEGRITY_ERRORS as exc:
                # The nkey was free, so this is the other unique index: one effect id already
                # has a dispatch row under a different key. That means the same effect was
                # keyed two ways, which would let it be dispatched twice.
                raise JournalWriteError(
                    f"effect {claim.effect_id!r} already has a dispatch row under a different "
                    f"key; one effect is one idempotency key ({exc})"
                ) from exc
            probe = self._backend.execute(_SELECT_CLAIM, params).fetchone()
            if probe is not None and str(probe["effect_id"]) == claim.effect_id:
                return DispatchClaim(Claim.OWNED, attempt=claim.attempt)
            row = probe
        if row is None:  # pragma: no cover - the insert both failed and left nothing
            raise JournalWriteError(f"dispatch claim for {claim.nkey} neither inserted nor found")

        status, outcome = str(row["status"]), str(row["last_outcome"])
        attempt = int(row["attempt"])
        if status == "dispatched":
            ack = row["ack_json"]
            return DispatchClaim(
                Claim.ALREADY_DISPATCHED,
                attempt=attempt,
                ack=json.loads(ack) if ack else None,
            )
        if status == "dead_letter":
            # A resume must retry a dead-lettered effect: the run finished ok=False and the
            # operator's remedy is to heal the upstream and resume (task 1.5).
            return DispatchClaim(Claim.RETRY_SAFE, attempt=attempt)
        if outcome == "not_sent":
            # The request demonstrably never left this process, so resending is not a
            # duplicate. This downgrade is what keeps crash window W3 narrow.
            return DispatchClaim(Claim.RETRY_SAFE, attempt=attempt)
        # in_flight with an unknown outcome: the request may or may not have taken effect.
        # This is the two-generals boundary and nothing removes it; the caller decides using
        # the tool's declared idempotency.
        return DispatchClaim(Claim.AMBIGUOUS, attempt=attempt)

    def _mark_not_sent(self, run_id: str, nkey: str, attempt: int) -> None:
        self._backend.execute(
            _UPDATE_NOT_SENT, {"run_id": run_id, "nkey": nkey, "attempt": attempt}
        )

    def _settle(
        self,
        run_id: str,
        nkey: str,
        status: str,
        ack: JsonValue,
        attempt: int,
        kind: str,
        payload: Mapping[str, JsonValue],
    ) -> int:
        """Record the outcome and journal it in ONE transaction, one fsync.

        Splitting them would leave the entry durable and the dedupe row not, or the reverse.
        A crash in between makes a resume re-send a charge that already went through, and
        the kill/resume test fails at whichever of its points lands in that microsecond --
        that is, it fails flakily, and a flaky safety test gets quarantined.
        """
        payload_json, payload_hash = canonical(payload).decode("utf-8"), chash(payload)
        self._backend.begin_immediate()
        try:
            head = self._head.get(run_id) or self._seed_head(run_id)
            offset = 0 if head is None else head.offset + 1
            prev = genesis_prev_hash(run_id) if head is None else head.entry_hash
            ts = _utc_now()
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
            self._backend.execute(
                _UPDATE_SETTLED,
                {
                    "run_id": run_id,
                    "nkey": nkey,
                    "status": status,
                    "ack_json": None if ack is None else canonical(ack).decode("utf-8"),
                    "entry_offset": offset,
                    "attempt": attempt,
                    "settled_at": ts,
                },
            )
            self._backend.commit()
        except Exception:
            self._backend.rollback()
            self._head.pop(run_id, None)
            raise
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

    def _unresolved(self, run_id: str) -> list[Mapping[str, JsonValue]]:
        rows = self._backend.execute(_SELECT_UNRESOLVED, {"run_id": run_id}).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._backend.close()


_writers: dict[str, _JournalWriter] = {}
_writers_lock = threading.Lock()


def evict_writer(path: Path | str) -> None:
    """Drop a cached writer, so the next :class:`Journal` opens a fresh connection.

    ``_writers`` caches one writer per location for the life of the process, and
    ``_PostgresBackend`` holds a single connection with no reconnect. After the server drops it
    -- a restart, a failover, a pooler's idle timeout, ``pg_terminate_backend`` -- every later
    append failed forever, and constructing a brand-new ``Journal(dsn)`` handed back the same
    dead writer. SQLite has no equivalent failure mode, so this only became reachable when the
    Postgres backend was wired.
    """
    key = str(path) if is_postgres_dsn(path) else os.path.realpath(path)
    with _writers_lock:
        writer = _writers.pop(key, None)
    if writer is not None:
        with contextlib.suppress(Exception):
            writer.close()


def _writer_for(path: Path | str, *, fullfsync: bool = False) -> _JournalWriter:
    # A DSN is its own key. ``realpath`` on one produces a nonsense relative path, and two
    # different DSNs could collapse onto the same entry.
    key = str(path) if is_postgres_dsn(path) else os.path.realpath(path)
    with _writers_lock:
        writer = _writers.get(key)
        if writer is None:
            writer = _JournalWriter(key if is_postgres_dsn(key) else Path(key), fullfsync=fullfsync)
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
        #: Kept as given. A Postgres DSN is not a path, and coercing it to one is how the
        #: Postgres backend came to be unreachable while its test reported success.
        self.location: Path | str = str(path) if is_postgres_dsn(path) else Path(path)
        self.path = Path(path) if not is_postgres_dsn(path) else Path(str(path))
        self._writer = _writer_for(self.location, fullfsync=fullfsync)

    @property
    def is_postgres(self) -> bool:
        return is_postgres_dsn(self.location)

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

    def _reader(self) -> _SqliteBackend | _PostgresBackend:
        # A separate read-only connection, never the writer's. query_only rather than a
        # mode=ro URI, because a read-only URI connection fails when the -shm file has to be
        # created.
        return _backend_for(self.location, read_only=True)

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

    def max_step(self, run_id: str, *, branches: frozenset[str] | None = None) -> int:
        """The highest ``step`` recorded, or -1.

        ``branches`` restricts the scan, and a resume must pass the set of branches that
        reached RETIRED. Counting every entry would include steps consumed by branches whose
        work was thrown away, so the counter would resume higher than the committed program
        position, every key derived after the resume would differ from the one derived before
        it, and the kill/resume comparison would fail intermittently -- at exactly the kill
        points that landed after a squash.
        """
        highest = -1
        for entry in self.read(run_id):
            owned = entry.branch_id is None or branches is None or entry.branch_id in branches
            if not owned:
                continue
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

    # -- dispatch deduplication (Hard Rule 8) --------------------------------------------

    async def claim_dispatch(self, claim: PendingClaim) -> DispatchClaim:
        """Stake a claim on sending this effect, durably, *before* the tool is called.

        The claim row is the intent record, which is why no sixteenth journal entry kind is
        invented for it: a send intent is neither a model output nor a tool result, so Hard
        Rule 5 does not reach it.
        """
        return await self._writer.submit_async(lambda: self._writer._claim(claim))

    async def mark_not_sent(self, run_id: str, nkey: str, attempt: int) -> None:
        """Record that an attempt failed before anything left the process."""
        await self._writer.submit_async(lambda: self._writer._mark_not_sent(run_id, nkey, attempt))

    async def settle_dispatch(
        self,
        *,
        run_id: str,
        nkey: str,
        status: Literal["dispatched", "dead_letter"],
        ack: JsonValue,
        attempt: int,
        kind: Literal["effect_dispatched", "effect_dead_lettered"],
        payload: Mapping[str, JsonValue],
    ) -> int:
        """Record the outcome and journal it atomically. Returns the entry offset."""
        validate_payload(kind, payload)
        return await self._writer.submit_async(
            lambda: self._writer._settle(run_id, nkey, status, ack, attempt, kind, payload)
        )

    def unresolved_dispatches(self, run_id: str) -> list[Mapping[str, JsonValue]]:
        """Claims still in flight. A resume's reconciliation list."""
        return self._writer.submit(lambda: self._writer._unresolved(run_id))

    def close(self) -> None:
        """Journals share a process-global writer; closing one closes none of the others."""
