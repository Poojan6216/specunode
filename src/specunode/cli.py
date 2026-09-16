"""The ``specunode`` command-line interface."""

from __future__ import annotations

import typer

from specunode import __version__

app = typer.Typer(
    name="specunode",
    help="Speculative execution for agent graphs, with a store buffer.",
    no_args_is_help=True,
    add_completion=False,
)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
