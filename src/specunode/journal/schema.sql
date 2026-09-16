-- SpecuNode journal schema. One file, loaded unmodified on SQLite and on Postgres 16.
--
-- Portability is why this file looks plainer than it could. Only TEXT, BIGINT, PRIMARY KEY,
-- and CREATE {TABLE,INDEX} IF NOT EXISTS appear. Deliberately absent: SERIAL, AUTOINCREMENT,
-- GENERATED AS IDENTITY, JSON/JSONB, BLOB/BYTEA, BOOLEAN, ENUM, TIMESTAMP, DEFAULT, CHECK,
-- triggers, WITHOUT ROWID, STRICT, COLLATE and every PRAGMA -- each of them either does not
-- exist on one side or means something different on the two. Offsets are assigned in Python
-- (see journal.py) rather than by the database, so the two backends cannot drift.
--
-- "offset" is a reserved word in both dialects and is double-quoted in every statement here
-- and in every query in the codebase.
--
-- Statements are separated by a line containing only a semicolon, so the loader can split
-- them without a SQL parser and split them identically for both backends.

CREATE TABLE IF NOT EXISTS journal_meta (
    k TEXT NOT NULL PRIMARY KEY,
    v TEXT NOT NULL
)
;

-- The append-only log. One row per entry, one entry per commit, one commit per fsync.
-- Hash-chained per run: prev_hash of entry n is entry_hash(entry n-1), where entry_hash is a
-- pure function of the other six columns, so the chain covers kind, offset, run_id and ts as
-- well as the payload and cannot be reordered or retyped undetected.
CREATE TABLE IF NOT EXISTS entries (
    run_id TEXT NOT NULL,
    "offset" BIGINT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    ts TEXT NOT NULL,
    PRIMARY KEY (run_id, "offset")
)
;

CREATE INDEX IF NOT EXISTS entries_run_kind_offset ON entries (run_id, kind, "offset")
;

-- Dispatch deduplication (Hard Rule 8). Keyed on nkey -- the Rule 8 key derived with an
-- EMPTY lineage -- and not on the lineage-bearing key. The reason is that the same logical
-- effect legitimately arrives down different branch paths: a stall discards the staged
-- effect and the sequential re-execution re-stages it under a different branch id, and a
-- resume re-mints branch ids entirely. Keyed on the lineage-bearing key, every one of those
-- would re-dispatch. nkey is also the idempotency token handed to the tool adapter, so the
-- upstream's own deduplication sees a value that is stable across all of it.
--
-- last_outcome distinguishes "the request never left this process" from "we do not know",
-- which is what lets a crash mid-dispatch be resolved without either double-charging a card
-- or dead-lettering an effect that plainly never went out.
CREATE TABLE IF NOT EXISTS effect_dispatch (
    run_id TEXT NOT NULL,
    nkey TEXT NOT NULL,
    idem_key TEXT NOT NULL,
    effect_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    status TEXT NOT NULL,
    last_outcome TEXT NOT NULL,
    attempt BIGINT NOT NULL,
    ack_json TEXT,
    entry_offset BIGINT,
    claimed_at TEXT NOT NULL,
    settled_at TEXT,
    PRIMARY KEY (run_id, nkey)
)
;

CREATE INDEX IF NOT EXISTS effect_dispatch_run_status ON effect_dispatch (run_id, status)
;

CREATE UNIQUE INDEX IF NOT EXISTS effect_dispatch_effect ON effect_dispatch (run_id, effect_id)
;
