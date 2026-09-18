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
from dataclasses import replace
from importlib import resources
from pathlib import Path

import typer

from specunode import __version__
from specunode.config import DEFAULT_CONFIG_NAME, Config, ConfigError, load_config
from specunode.journal.journal import Journal
from specunode.journal.ledger import (
    build_ledger,
    load_or_create_key,
    render_ledger,
    sign_ledger,
    verify_ledger,
)
from specunode.journal.replay import ReplayDivergence, ReplayModel, recover
from specunode.runner import RunnerError, build_graph, build_target

app = typer.Typer(
    name="specunode",
    help="Speculative execution for agent graphs, with a store buffer.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_JOURNAL = Path("./.specunode/journal.db")


def _load_or_exit(config: Path | None) -> Config:
    """Load a config, reporting a missing or unreadable one rather than raising a traceback.

    An explicit ``--config`` that does not exist is exit 2 with a message. Without the option,
    the search is ``./specunode.yaml`` then ``$XDG_CONFIG_HOME/specunode/config.yaml``, and
    finding nothing means built-in defaults, as it always has.
    """
    if config is not None and not config.is_file():
        typer.echo(f"no config at {config}", err=True)
        raise typer.Exit(2)
    try:
        return load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


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
        # Read from inside the package, not from the repo root. It used to resolve
        # ``parents[2]``, which is the checkout only when running from source: from an
        # installed wheel that is ``lib/python3.11/``, the file was absent, and the fallback
        # wrote ``schema_version: 1`` and nothing else -- no ``graph:``, no ``target:``, no
        # ``tools:``. Every user who installed the package and ran ``init`` got an 18-byte
        # config that cannot drive anything. The example was listed in neither the wheel nor
        # the sdist include lists, so the fallback was the only path that ever ran for them.
        example = resources.files("specunode").joinpath("specunode.yaml.example")
        target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
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


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="The run to continue."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
    config: Path = typer.Option(None, "--config", help=f"Defaults to ./{DEFAULT_CONFIG_NAME}."),
) -> None:
    """Continue a run that was interrupted, without re-sending what already went out.

    Nothing is replayed and nothing is re-decided. Committed state is rebuilt from the branches
    the journal records as retired, the step counter continues above the position they consumed,
    and the graph is driven on from there. An effect that was acked before the crash is claimed
    and skipped; one whose request demonstrably never left is re-sent; one that may or may not
    have taken effect is dead-lettered unless its tool declared a repeat harmless.

    This needs a live target: the turns the journal does not already hold have to be asked for.
    """
    import asyncio

    from specunode.buffer.dispatcher import Dispatcher
    from specunode.buffer.store_buffer import StoreBuffer
    from specunode.core.model import JournaledModel
    from specunode.core.scheduler import Scheduler

    loaded = _load_or_exit(config)
    try:
        adapter, registry = build_graph(loaded)
        target = build_target(loaded)
    except RunnerError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    book = Journal(journal)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=book,
        buffer=StoreBuffer(journal=book, run_id=""),
        dispatcher=Dispatcher(registry=registry),
        target=JournaledModel(target, book, provider=loaded.target.provider),
        policy=loaded.to_policy(),
        reducers=loaded.state.reducers,
    )
    result = asyncio.run(scheduler.resume(run_id))
    typer.echo(render_ledger(result.ledger))
    if not result.ok:
        typer.echo(f"run did not complete: {result.error}", err=True)
        raise typer.Exit(1)


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="The run to replay."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
    config: Path = typer.Option(None, "--config", help=f"Defaults to ./{DEFAULT_CONFIG_NAME}."),
    speculation: str = typer.Option("on", "--speculation", help="on|off."),
    dispatch: bool = typer.Option(
        False,
        "--dispatch",
        help="Actually send effects. Off by default: a replay that re-sent every effect "
        "would charge every card again.",
    ),
) -> None:
    """Re-run a journaled run against its own recorded model output.

    Refuses at the first turn whose request does not match the journal's, naming the step and
    the fields that differ, rather than continuing down a trajectory the recorded run never
    took. Dispatches nothing unless ``--dispatch`` says otherwise.
    """
    import asyncio

    from specunode.buffer.dispatcher import Dispatcher
    from specunode.buffer.store_buffer import StoreBuffer
    from specunode.core.scheduler import Scheduler
    from specunode.ids import new_ulid

    if speculation not in {"on", "off"}:
        typer.echo(f"--speculation takes 'on' or 'off', not {speculation!r}", err=True)
        raise typer.Exit(2)

    loaded = _load_or_exit(config)
    try:
        adapter, registry = build_graph(loaded)
    except RunnerError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    source = Journal(journal)
    recovery = recover(source, run_id)
    # A replay re-drives the run from its *beginning*, so it starts from the inputs the
    # journal recorded rather than from the state the run ended in. Starting from the end
    # state routes straight to the graph's terminal node and replays nothing, which looks
    # like a clean replay and checks not one thing.
    started = next(iter(source.read(run_id, kinds=["run_started"])), None)
    if started is None:
        typer.echo(f"run {run_id} has no run_started entry; there is nothing to replay", err=True)
        raise typer.Exit(2)
    inputs = started.payload.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    # A fresh journal: replaying into the one being read would interleave a new run's entries
    # with the record it is checking against, and the record is the only evidence there is.
    into = Journal(journal.parent / f"replay-{run_id}.db")
    policy = loaded.to_policy()
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=into,
        buffer=StoreBuffer(journal=into, run_id=""),
        dispatcher=Dispatcher(registry=registry, dry_run=not dispatch),
        target=ReplayModel(
            journal=source, run_id=run_id, retired_branches=recovery.retired_branches
        ),
        policy=replace(policy, speculation=speculation == "on"),
        reducers=loaded.state.reducers,
    )
    try:
        result = asyncio.run(scheduler.run(new_ulid(), dict(inputs)))
    except ReplayDivergence as divergence:
        typer.echo(str(divergence), err=True)
        raise typer.Exit(1) from divergence
    if not dispatch:
        typer.echo("(dry run: no effect was sent; pass --dispatch to send them)")
    typer.echo(render_ledger(result.ledger))
    if not result.ok:
        typer.echo(f"replay did not complete: {result.error}", err=True)
        raise typer.Exit(1)


def signature_path(journal: Path, run_id: str, explicit: Path | None = None) -> Path:
    """Where a run's ledger signature lives: beside the journal, never inside it.

    Inside is impossible, not merely untidy. The signed payload covers ``journal_head`` and
    ``journal_entries``, so appending the signature to the journal it signs changes the material
    it was computed over and the signature stops verifying against a freshly built ledger.

    This is why ``verify-ledger`` could never verify anything: ``sign-ledger`` echoed the
    envelope to the terminal and wrote it nowhere, ``build_ledger`` never assigns ``signature``,
    and so verification returned ``unsigned`` before running any of its four checks.
    """
    if explicit is not None:
        return explicit
    return journal.parent / "ledgers" / f"{run_id}.sig"


@app.command("verify-ledger")
def verify_ledger_command(
    run_id: str = typer.Argument(..., help="The run to verify."),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="Journal database."),
    keystore: Path = typer.Option(Path("./.specunode/keys"), "--keystore"),
    signature: Path = typer.Option(
        None, "--signature", help="Signature file. Defaults to <journal dir>/ledgers/<run>.sig."
    ),
) -> None:
    """Check a ledger's signature and its journal chain, and say which failed.

    An edited ledger and a ledger signed by an unknown key are different problems with
    different remedies, so they are reported as different reasons rather than one refusal.
    """
    from dataclasses import replace as _replace

    store = Journal(journal)
    built = build_ledger(store, run_id)
    envelope_path = signature_path(journal, run_id, signature)
    if envelope_path.is_file():
        built = _replace(built, signature=envelope_path.read_text(encoding="utf-8").strip())
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
    out: Path = typer.Option(
        None, "--out", help="Where to write it. Defaults to <journal dir>/ledgers/<run>.sig."
    ),
) -> None:
    """Sign a run's ledger with the local key, creating one on first use.

    The envelope is **written to a file**, not only printed. It used to be echoed and stored
    nowhere, so ``verify-ledger`` rebuilt an unsigned ledger and reported ``unsigned`` on every
    run that had been signed.
    """
    store = Journal(journal)
    key = load_or_create_key(keystore)
    signed = sign_ledger(build_ledger(store, run_id), key)
    destination = signature_path(journal, run_id, out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(signed.signature + "\n", encoding="utf-8")
    typer.echo(f"signed with {key.key_id}")
    typer.echo(f"wrote {destination}")
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


@app.command("mcp-proxy")
def mcp_proxy(
    upstream: str = typer.Option(
        ..., "--upstream", help="Command that starts the upstream server."
    ),
    config: Path = typer.Option(None, "--config", help=f"Defaults to ./{DEFAULT_CONFIG_NAME}."),
    deadline: float = typer.Option(
        300.0,
        "--deadline",
        help=(
            "Seconds a blocking write waits for a decision before giving up. It is then "
            "still held and still unsent, and the client is told so."
        ),
    ),
    handles: bool = typer.Option(
        False,
        "--handles",
        help=(
            "Return a staged-write handle instead of blocking. Only safe for a client that "
            "understands one and will not put it in a prompt."
        ),
    ),
) -> None:
    """Proxy an MCP server, holding its writes until the model's decision confirms them.

    Reads are forwarded immediately. Everything else is held. Because the proxy cannot see the
    model, the decision arrives out of band -- from a client that reports it, or from
    a second call to the proxy's own ``specunode.retire`` tool. That needs a client which can
    issue one while a write is outstanding; a single-threaded client blocked on the write
    cannot, and its call gives up after the proxy's decision deadline with the write still
    held and unsent.
    """
    import asyncio
    import shlex

    from specunode.core.effects import ToolRegistry
    from specunode.integrations.mcp_proxy import ClientMode, ProxyState, serve

    # An explicit path that does not exist is an error, not a reason to load something else.
    # This used to fall back to ``find_config()`` -- ./specunode.yaml, then
    # $XDG_CONFIG_HOME/specunode/config.yaml -- so a typo in the path silently proxied with a
    # different override table and said nothing about which file it had read.
    if config is not None and not config.is_file():
        typer.echo(f"no config at {config}", err=True)
        raise typer.Exit(2)
    loaded = load_config(config)
    registry = ToolRegistry()
    for _name, spec in loaded.tool_overrides().items():
        registry.register(spec)

    state = ProxyState(
        registry=registry,
        mode=ClientMode.HANDLES if handles else ClientMode.BLOCKING,
        decision_deadline_s=deadline,
    )
    typer.echo(
        f"proxying {upstream!r} in {state.mode.value} mode; "
        f"context identity is {state.context_identity} over MCP",
        err=True,
    )
    asyncio.run(serve(shlex.split(upstream), state))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
