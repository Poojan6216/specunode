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
import threading

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


def test_concurrent_bootstrap_of_a_fresh_database_does_not_race() -> None:
    """Postgres's ``CREATE ... IF NOT EXISTS`` is not race-safe, and this runs on every connect.

    Two connections executing the same DDL concurrently collide in the system catalogues and
    one gets a ``UniqueViolation`` from ``pg_type`` or ``pg_class`` -- a raw driver exception,
    not a ``JournalError``. Both the writer and every reader open a connection, so a handful of
    processes starting against a not-yet-bootstrapped database mostly failed. SQLite never
    showed it, because its write lock serialises the same DDL.

    Measured before the fix with eight concurrent processes over three rounds: 21 of 24 opens
    failed. Threads are used here rather than processes because the failure is in the server's
    catalogue, not in the client, so it reproduces either way and this keeps the test cheap.
    """
    psycopg = pytest.importorskip("psycopg")
    from specunode.journal.journal import _PostgresBackend

    with psycopg.connect(dsn(), autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS entries, effect_dispatch, journal_meta CASCADE")

    errors: list[str] = []
    backends: list[object] = []

    def bootstrap() -> None:
        try:
            backends.append(_PostgresBackend(dsn()))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=bootstrap) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for backend in backends:
        backend.close()  # type: ignore[attr-defined]

    assert errors == [], f"concurrent bootstrap raced: {errors[:3]}"


def test_a_dead_writer_can_be_evicted_and_replaced() -> None:
    """``_writers`` caches one writer per DSN for the life of the process, with no reconnect.

    After the server drops the connection -- a restart, a failover, a pooler's idle timeout --
    every later append failed forever, and constructing a brand-new ``Journal(dsn)`` handed back
    the same dead writer. SQLite has no equivalent failure mode, so this only became reachable
    when the Postgres backend was wired.
    """
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal, evict_writer

    run_id = f"01PGEVICT{os.getpid():017d}"[:26]
    first = Journal(dsn())
    first.append(run_id, "policy_event", {"v": 1, "event": "tick", "reason": "pg"})
    before = server_rows(run_id)

    evict_writer(dsn())

    second = Journal(dsn())
    second.append(run_id, "policy_event", {"v": 1, "event": "tick", "reason": "pg"})
    assert server_rows(run_id) == before + 1, "the replacement writer did not reach the server"
    assert second.verify_chain(run_id).ok, "the chain did not survive the writer being replaced"


def test_two_runs_whose_old_lock_keys_collide_are_driven_at_once() -> None:
    """The run lock's key was ``hashtext``, 32 bits, and ``specunode-run:run-37193`` and
    ``specunode-run:run-81426`` hash alike: two unrelated runs could not be driven at once.
    Found by the eleventh review."""
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal

    tag = f"{os.getpid()}"
    journal = Journal(dsn())
    with journal.hold_run(f"run-37193-{tag}"), journal.hold_run(f"run-81426-{tag}"):
        pass
    # The pair itself, as the review found it.
    with journal.hold_run("run-37193"), journal.hold_run("run-81426"):
        pass


def test_evicting_the_writer_does_not_release_a_held_run() -> None:
    """The lock was taken on the writer's connection, and ``evict_writer`` -- the cure for a
    dead one -- closed it, releasing every run's lock with the runs still going."""
    pytest.importorskip("psycopg")
    from specunode.journal.journal import Journal, RunBusy, evict_writer

    run_id = f"01PGHELD{os.getpid():018d}"[:26]
    with Journal(dsn()).hold_run(run_id):
        evict_writer(dsn())
        with pytest.raises(RunBusy, match="another process"), Journal(dsn())._run_lock(run_id):
            pass


async def test_a_run_whose_lock_went_with_its_connection_sends_nothing_more() -> None:
    """The lock lives as long as its connection: a restart, a failover or
    ``pg_terminate_backend`` ends both while the process drives on, and another could then take
    the run up. The next claim is refused instead of made."""
    psycopg = pytest.importorskip("psycopg")
    from specunode.journal import journal as module
    from specunode.journal.journal import Journal, PendingClaim, RunBusy

    run_id = f"01PGLOST{os.getpid():018d}"[:26]
    journal = Journal(dsn())
    with journal.hold_run(run_id):
        held = module._run_lock_connections[(journal._lock_location(), run_id)]
        with psycopg.connect(dsn(), autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (held.info.backend_pid,))
        with pytest.raises(RunBusy, match="lost its lock"):
            await journal.claim_dispatch(
                PendingClaim(
                    run_id=run_id,
                    nkey="n-1",
                    idem_key="k-1",
                    effect_id="e-1",
                    branch_id="b-1",
                    tool="charge_card",
                )
            )
    assert not journal.unresolved_dispatches(run_id), "a claim was staked after the lock was lost"


def test_the_cli_reads_a_postgres_journal() -> None:
    """``--journal`` was a ``Path``, which folds ``postgresql://`` into ``postgresql:/``: the
    CLI opened a SQLite file in a folder named ``postgresql:``. Found by the eleventh review."""
    pytest.importorskip("psycopg")
    from typer.testing import CliRunner

    from specunode.cli import app
    from specunode.journal.journal import Journal

    run_id = f"01PGCLI{os.getpid():019d}"[:26]
    Journal(dsn()).append(run_id, "policy_event", {"v": 1, "event": "tick", "reason": "cli"})
    result = CliRunner().invoke(app, ["runs", "--journal", dsn()])
    assert result.exit_code == 0, result.output
    assert run_id in result.output.split()


def test_a_runtime_given_a_dsn_opens_postgres() -> None:
    pytest.importorskip("psycopg")
    import specunode
    from specunode.core.decision import FreeText

    @specunode.node()
    async def done(session: specunode.RunSession) -> specunode.Decision:
        return FreeText.of("done")

    runtime = specunode.Runtime(specunode.graph([done], lambda state: None), journal=dsn())
    assert runtime._journal().is_postgres
