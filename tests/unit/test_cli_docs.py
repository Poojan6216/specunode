"""Every subcommand the CLI registers is documented, and nothing documented is imaginary.

Seven of ten commands had no mention anywhere in the docs. A reference written once and never
checked drifts the moment a flag is renamed, so the reference is held to the source: every
registered command name and every option literal must appear in ``docs/cli.md``, and every
command the doc describes must exist.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "src" / "specunode" / "cli.py"
DOC = REPO / "docs" / "cli.md"

_HEADING = re.compile(r"^### `specunode ([a-z-]+)", re.MULTILINE)


def registered_commands() -> list[str]:
    """Asked of Typer, not of the source text.

    A regex over ``@app.command(...)`` missed every other registration form -- a ``name=``
    keyword, single quotes, an extra argument, ``async def`` -- and the only guard was a
    minimum count, so an eleventh command registered any of those ways would have been
    undocumented and unnoticed. The app object knows what it registered.
    """
    from specunode.cli import app

    return [
        command.name or (command.callback.__name__.replace("_", "-") if command.callback else "")
        for command in app.registered_commands
    ]


def option_flags() -> list[str]:
    """Every long option Typer will accept, per command, from the built Click command."""
    import typer

    from specunode.cli import app

    built = typer.main.get_command(app)
    flags: set[str] = set()
    for name in built.commands:  # type: ignore[attr-defined]
        for param in built.commands[name].params:  # type: ignore[attr-defined]
            flags.update(opt for opt in param.opts if opt.startswith("--"))
    for param in built.params:
        flags.update(opt for opt in param.opts if opt.startswith("--"))
    return sorted(flags)


def test_every_registered_command_is_documented() -> None:
    names = registered_commands()
    assert len(names) >= 10, names
    documented = _HEADING.findall(DOC.read_text(encoding="utf-8"))
    for name in names:
        assert name in documented, f"`specunode {name}` is registered but not in docs/cli.md"


def test_nothing_documented_is_imaginary() -> None:
    documented = _HEADING.findall(DOC.read_text(encoding="utf-8"))
    assert set(documented) == set(registered_commands())


def test_every_option_flag_is_documented() -> None:
    doc = DOC.read_text(encoding="utf-8")
    flags = option_flags()
    assert {"--dispatch", "--journal", "--version"} <= set(flags), flags
    for flag in flags:
        assert f"`{flag}" in doc, f"{flag} is an option of the CLI and absent from docs/cli.md"


def test_every_verify_ledger_exit_code_is_documented() -> None:
    """The doc used to say "the exit code is the category's", which is not a number.

    There are six, and two different failures share one of them, so a reader cannot derive the
    table from the category names.
    """
    from specunode.journal.ledger import _EXIT_CODES

    section = DOC.read_text(encoding="utf-8").split("### `specunode verify-ledger")[1]
    section = section.split("### ")[0]
    for code in sorted(set(_EXIT_CODES.values())):
        assert f"{code} " in section, f"exit code {code} is not in the verify-ledger section"


def test_the_config_search_order_is_documented_where_it_is_used() -> None:
    """A command that silently loads a different file than the one named is a trap.

    ``find_config`` falls back to ``$XDG_CONFIG_HOME``; the reference described only
    ``./specunode.yaml``, and ``mcp-proxy`` fell back to that search even when ``--config``
    named a file that did not exist.
    """
    doc = DOC.read_text(encoding="utf-8")
    assert "XDG_CONFIG_HOME" in doc
    assert "exit 2" in doc and "never a silent fallback" in doc
