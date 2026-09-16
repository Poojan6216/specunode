"""The ``specunode`` command-line interface.

Every subcommand here answers a question about a run that already happened, or continues one
that did not finish. None of them decides anything: the journal decides, and these read it.

``replay`` deserves a note. It re-runs a journaled run against :class:`ReplayModel`, which
serves the recorded responses and refuses the moment the run would ask the model something the
journal does not record. By default it dispatches nothing -- a replay that re-sent every effect
would charge every card again -- and re-dispatching is available only behind an explicit flag
that names what it is for.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from specunode import __version__
from specunode.config import DEFAULT_CONFIG_NAME
from specunode.journal.journal import Journal
from specunode.journal.ledger import (
    build_ledger,
    load_or_create_key,
    render_ledger,
    sign_ledger,
    verify_ledger,
)
from specunode.journal.replay import recover

app = typer.Typer(
    name="specunode",
    help="Speculative execution for agent graphs, with a store buffer.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_JOURNAL = Path("./.specunode/journal.db")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"specunode {__version__}")
        raise typer.Exit(0)


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """SpecuNode CLI."""


@app.command()
def init(
    directory: Path = typer.Argument(Path(), help="Where to write the config and state."),
) -> None:
    """Write ``specunode.yaml`` and create ``.specunode/``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".specunode").mkdir(exist_ok=True)
    target = directory / DEFAULT_CONFIG_NAME
    if target.exists():
        typer.echo(f"{target} already exists; leaving it alone")
    else:
        example = Path(__file__).resolve().parents[2] / "specunode.yaml.example"
        if example.is_file():
            target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        else:  # pragma: no cover - installed wheels carry the example in the sdist only
            target.write_text("schema_version: 1\n", encoding="utf-8")
        typer.echo(f"wrote {target}")
    typer.echo(f"created {directory / '.specunode'}")


@app.command()
def runs(
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
) -> None:
    """List the runs this journal holds."""
    for run_id in Journal(journal).runs():
        typer.echo(run_id)


@app.command()
def ledger(
    run_id: str = typer.Argument(..., help="The run to render."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
    short: bool = typer.Option(False, "--short", help="Abbreviate ids."),
    normalised: bool = typer.Option(
        False, "--normalised", help="Render only what the equivalence relation compares."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the rows as JSON."),
) -> None:
    """Print a run's effect ledger: what reached the world, and what authorised it."""
    built = build_ledger(Journal(journal), run_id)
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "effect_id": row.effect_id,
                        "tool": row.call.name,
                        "args": dict(row.call.args),
                        "key": row.nkey,
                        "branch": row.branch_id,
                        "authorised_by_step": row.authorised_by_step,
                        "status": row.status,
                    }
                    for row in built.rows
                ],
                indent=2,
            )
        )
        return
    typer.echo(render_ledger(built, short_ids=short, normalised=normalised))


@app.command("verify-ledger")
def verify_ledger_command(
    run_id: str = typer.Argument(..., help="The run to verify."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
    keystore: Path = typer.Option(Path("./.specunode/keys"), "--keystore"),
) -> None:
    """Check a ledger's signature and its journal chain, and say which failed.

    An edited ledger and a ledger signed by an unknown key are different problems with
    different remedies, so they are reported as different reasons rather than one refusal.
    """
    store = Journal(journal)
    built = build_ledger(store, run_id)
    result = verify_ledger(built, journal=store, store=keystore)
    typer.echo(f"{result.category}: {result.detail or result.reason}")
    if result.key_id:
        typer.echo(f"key: {result.key_id}")
    typer.echo(f"journal: {result.journal}")
    raise typer.Exit(result.exit_code)


@app.command("sign-ledger")
def sign_ledger_command(
    run_id: str = typer.Argument(..., help="The run to sign."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
    keystore: Path = typer.Option(Path("./.specunode/keys"), "--keystore"),
) -> None:
    """Sign a run's ledger with the local key, creating one on first use."""
    store = Journal(journal)
    key = load_or_create_key(keystore)
    signed = sign_ledger(build_ledger(store, run_id), key)
    typer.echo(f"signed with {key.key_id}")
    typer.echo(signed.signature)


@app.command()
def verify(
    run_id: str = typer.Argument(..., help="The run whose chain to walk."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
) -> None:
    """Walk a run's hash chain and report the first break, if any."""
    result = Journal(journal).verify_chain(run_id)
    if result.ok:
        typer.echo(f"ok: {result.entries} entries, chain verified")
        return
    typer.echo(f"{result.reason} at offset {result.first_bad_offset}: {result.detail or ''}")
    raise typer.Exit(1)


@app.command()
def status(
    run_id: str = typer.Argument(..., help="The run to inspect."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
) -> None:
    """Say what a run left behind, and what a resume would build on."""
    recovery = recover(Journal(journal), run_id)
    typer.echo(f"run {run_id}")
    typer.echo(f"  finished: {recovery.finished}")
    typer.echo(f"  entries through offset: {recovery.last_offset}")
    typer.echo(f"  retired branches: {len(recovery.retired_branches)}")
    typer.echo(f"  confirmed but not retired: {list(recovery.confirmed_not_retired)}")
    typer.echo(f"  dispatch claims still in flight: {len(recovery.unresolved_dispatches)}")
    typer.echo(f"  step index to continue above: {recovery.step_index}")
    typer.echo(f"  resumable: {recovery.resumable}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
