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

_COMMAND = re.compile(r'@app\.command\((?:"([a-z-]+)")?\)\s*\ndef ([a-z_]+)\(')
_HEADING = re.compile(r"^### `specunode ([a-z-]+)", re.MULTILINE)
_FLAG = re.compile(r'"(--[a-z][a-z-]*)"')


def registered_commands() -> list[str]:
    source = CLI.read_text(encoding="utf-8")
    return [name or fn.replace("_", "-") for name, fn in _COMMAND.findall(source)]


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
    flags = sorted(set(_FLAG.findall(CLI.read_text(encoding="utf-8"))))
    assert "--dispatch" in flags, "the regex stopped finding flags"
    for flag in flags:
        assert f"`{flag}" in doc, f"{flag} is an option in cli.py and absent from docs/cli.md"
