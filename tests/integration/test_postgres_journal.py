"""The same journal on Postgres 16 (spec task 8.5).

Gated on ``SPECUNODE_TEST_POSTGRES_DSN``. Skipped where no server is configured, which locally
is most of the time; CI runs a Postgres service and does not skip.

The property is that ``schema.sql`` loads **unmodified** on both backends and behaves the same.
That is why the DDL contains only TEXT, BIGINT, PRIMARY KEY and CREATE ... IF NOT EXISTS, and
why offsets are assigned in Python rather than by the database: the moment one backend assigns
an identity the other does not, two runs of the same workload stop producing the same journal.
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
