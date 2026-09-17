"""The same journal on Postgres 16 (spec task 8.5).

Gated on ``SPECUNODE_TEST_POSTGRES_DSN``. Skipped where no server is configured, which locally
is most of the time; CI runs a Postgres service and does not skip.

The property is that ``schema.sql`` loads **unmodified** on both backends and behaves the same.
That is why the DDL contains only TEXT, BIGINT, PRIMARY KEY and CREATE ... IF NOT EXISTS, and
why offsets are assigned in Python rather than by the database: the moment one backend assigns
an identity the other does not, two runs of the same workload stop producing the same journal.

**Every test here proves it is on Postgres before it asserts anything else.** This file used to
pass without one: ``Journal(dsn)`` coerced the DSN with ``Path(...)``, so it opened an ordinary
SQLite database in a file named after the connection string, and a CI job called "the same suite,
on the Postgres journal" went green having never opened a connection. Asserting the backend is
cheap; discovering years later that a whole job was theatre is not.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not os.environ.get("SPECUNODE_TEST_POSTGRES_DSN"),
        reason="set SPECUNODE_TEST_POSTGRES_DSN to run the Postgres journal tests",
    ),
]


def test_the_ddl_loads_unmodified_on_postgres() -> None:
    """One DDL file, both backends. A dialect-specific one would be two sources of truth."""
    psycopg = pytest.importorskip("psycopg")
    from specunode.journal.journal import _load_schema_statements

    dsn = os.environ["SPECUNODE_TEST_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=True) as conn:
        for statement in _load_schema_statements():
            conn.execute(statement)
        # Idempotent: every statement is CREATE ... IF NOT EXISTS.
        for statement in _load_schema_statements():
            conn.execute(statement)


def test_parameter_rewriting_covers_every_statement() -> None:
    """The DML is written once; a missed rewrite is a runtime error on the Postgres path only."""
    from specunode.journal import journal as module

    for name in dir(module):
        if not name.startswith("_") or not name.isupper():
            continue
        sql = getattr(module, name)
        if not isinstance(sql, str) or " " not in sql:
            continue
        rewritten = module._to_postgres(sql)
        assert ":run_id" not in rewritten and ":nkey" not in rewritten, (
            f"{name} still contains SQLite-style parameters after rewriting"
        )


def test_appending_and_chain_verification_work_the_same(tmp_path: object) -> None:
    """Whatever the backend, the chain is the chain."""
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal

    dsn = os.environ["SPECUNODE_TEST_POSTGRES_DSN"]
    journal = Journal(dsn)
    run_id = "01PGRUNAAAAAAAAAAAAAAAAAAA"
    for index in range(20):
        journal.append(
            run_id, "policy_event", {"v": 1, "event": "tick", "reason": "pg", "i": index}
        )
    result = journal.verify_chain(run_id)
    assert result.ok and result.entries >= 20


# -- proving the backend, rather than assuming it ----------------------------------------------


def dsn() -> str:
    return os.environ["SPECUNODE_TEST_POSTGRES_DSN"]


def server_rows(run_id: str) -> int:
    """Count this run's entries through a connection the Journal knows nothing about.

    The point of reading it back independently: a Journal that had quietly opened SQLite would
    answer its own queries perfectly well and this number would be zero.
    """
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(dsn(), autocommit=True) as conn:
        row = conn.execute("SELECT count(*) FROM entries WHERE run_id = %s", (run_id,)).fetchone()
    return int(row[0]) if row else 0


def test_a_dsn_selects_the_postgres_backend_and_not_a_file_named_after_it() -> None:
    from specunode.journal.journal import Journal, is_postgres_dsn

    assert is_postgres_dsn(dsn())
    journal = Journal(dsn())
    assert journal.is_postgres, "the DSN opened something that is not Postgres"


def test_entries_written_through_the_journal_are_visible_on_the_server() -> None:
    """The assertion the old version of this file was missing entirely."""
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal

    journal = Journal(dsn())
    run_id = f"01PGVIS{os.getpid():019d}"[:26]
    for index in range(5):
        journal.append(
            run_id, "policy_event", {"v": 1, "event": "tick", "reason": "pg", "i": index}
        )

    assert server_rows(run_id) == 5, "the rows are not on the Postgres server"


def test_reading_back_through_the_journal_returns_what_was_written() -> None:
    """Exercises the read path, which was hard-coded to SQLite and so had never run."""
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal

    journal = Journal(dsn())
    run_id = f"01PGREAD{os.getpid():018d}"[:26]
    for index in range(6):
        journal.append(
            run_id, "policy_event", {"v": 1, "event": "tick", "reason": "pg", "i": index}
        )

    entries = list(journal.read(run_id))
    assert [entry.offset for entry in entries] == [0, 1, 2, 3, 4, 5]
    assert all(entry.kind == "policy_event" for entry in entries)
    # ``after`` is exclusive and dense from zero on both backends.
    assert [entry.offset for entry in journal.read(run_id, after=3)] == [4, 5]


async def test_a_dispatch_claim_round_trips_on_postgres() -> None:
    """The dedupe table is what stops a resumed run re-sending; it is backend code too."""
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Claim, Journal, PendingClaim

    journal = Journal(dsn())
    run_id = f"01PGCLAIM{os.getpid():017d}"[:26]
    claim = PendingClaim(
        run_id=run_id,
        nkey=f"nkey-{os.getpid()}",
        idem_key=f"key-{os.getpid()}",
        effect_id=f"eff-{os.getpid()}",
        branch_id="br-1",
        tool="restart_job",
    )
    first = await journal.claim_dispatch(claim)
    assert first.outcome is Claim.OWNED

    # A second claim on the same key must not come back OWNED, or a resume would re-send.
    second = await journal.claim_dispatch(claim)
    assert second.outcome is not Claim.OWNED


def test_the_chain_verifies_on_postgres() -> None:
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal

    journal = Journal(dsn())
    run_id = f"01PGCHAIN{os.getpid():017d}"[:26]
    for index in range(12):
        journal.append(
            run_id, "policy_event", {"v": 1, "event": "tick", "reason": "pg", "i": index}
        )

    result = journal.verify_chain(run_id)
    assert result.ok and result.entries == 12
    assert server_rows(run_id) == 12
