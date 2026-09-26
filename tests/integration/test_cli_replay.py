"""``specunode replay``, end to end (spec task 2.6).

The command existed in the CLI module's own docstring long before it existed in the CLI, which
is the kind of gap a test catches and a reading does not. What is asserted here is the pair of
behaviours the spec names: it refuses when the run would ask the model a different question,
naming the step and what differs; and it sends nothing unless told to.

The tests are synchronous. The CLI calls ``asyncio.run`` -- it is a command-line program, not a
coroutine -- and that cannot nest inside the loop ``pytest-asyncio`` opens for an ``async def``
test. Running the command the way a person runs it is the point.

What reached the world is read from the world's durable log rather than from a ``World`` object,
because the CLI builds its own through a ``module:attribute`` reference and there is no shared
object to inspect. See ``_cli_graph`` for why that indirection is stateless.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.integration._cli_graph import (
    SYSTEM_ENV,
    TURN,
    WORLD_ENV,
    SystemGraph,
    registry_for,
    world_for,
)
from typer.testing import CliRunner

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.cli import app
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn

GRAPH_REFERENCE = "tests.integration._cli_graph:build"


@pytest.fixture
def scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A directory holding the journal, the world log and the config."""
    monkeypatch.setenv(WORLD_ENV, str(tmp_path / "world.jsonl"))
    monkeypatch.delenv(SYSTEM_ENV, raising=False)
    (tmp_path / "specunode.yaml").write_text(
        f'schema_version: 1\ngraph: "{GRAPH_REFERENCE}"\n', encoding="utf-8"
    )
    yield tmp_path


def mutations(directory: Path) -> list[str]:
    log = directory / "world.jsonl"
    if not log.exists():
        return []
    return [
        str(json.loads(line)["tool"])
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind") == "mutation"
    ]


async def _record_async(directory: Path) -> str:
    world = world_for(directory / "world.jsonl")
    registry = registry_for(world)
    journal = Journal(directory / "journal.db")
    scheduler = Scheduler(
        graph=SystemGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(turns=[tool_turn(*TURN, turn=0)]), journal, provider="scripted"
        ),
        policy=Policy(speculation=True),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error
    world.close()
    return run_id


def record(directory: Path) -> str:
    """Produce a real journal, and a world log with one mutation in it, for the CLI to replay."""
    run_id = asyncio.run(_record_async(directory))
    assert mutations(directory) == ["restart_job"], "the recorded run changed nothing"
    (directory / "world.jsonl").unlink()  # the replay starts from an untouched world
    return run_id


def invoke(scene: Path, run_id: str, *extra: str) -> object:
    return CliRunner().invoke(
        app,
        [
            "replay",
            run_id,
            "--journal",
            str(scene / "journal.db"),
            "--config",
            str(scene / "specunode.yaml"),
            *extra,
        ],
    )


def test_replay_sends_nothing_by_default(scene: Path) -> None:
    """A replay that re-sent every effect would charge every card again."""
    run_id = record(scene)
    result = invoke(scene, run_id)

    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert "dry run" in result.output  # type: ignore[attr-defined]
    assert mutations(scene) == [], "a dry-run replay reached the world"
    # The ledger says so on the row, not only in the banner: a rendering is something people
    # paste into a ticket, and one that reads DISPATCHED for an unsent effect is a lie that
    # travels further than the banner does.
    assert "(dry run: not sent)" in result.output  # type: ignore[attr-defined]


def test_replay_dispatches_when_told_to(scene: Path) -> None:
    run_id = record(scene)
    result = invoke(scene, run_id, "--dispatch")

    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert mutations(scene) == ["restart_job"]


def test_replay_refuses_a_changed_prompt_and_names_the_step(
    scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the command: refuse, rather than re-run a trajectory nobody recorded."""
    run_id = record(scene)
    monkeypatch.setenv(SYSTEM_ENV, "You are a cautious operator.")

    result = invoke(scene, run_id)

    assert result.exit_code == 1, result.output  # type: ignore[attr-defined]
    assert "diverged at step" in result.output  # type: ignore[attr-defined]
    assert "cautious operator" in result.output  # type: ignore[attr-defined]
    assert mutations(scene) == [], "a refused replay still reached the world"


def test_replay_with_speculation_off_also_completes(scene: Path) -> None:
    run_id = record(scene)
    result = invoke(scene, run_id, "--speculation", "off")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]


def test_replay_refuses_a_bad_speculation_flag(scene: Path) -> None:
    run_id = record(scene)
    result = invoke(scene, run_id, "--speculation", "maybe")
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "'on' or 'off'" in result.output  # type: ignore[attr-defined]


def test_replay_without_a_graph_in_the_config_says_what_to_add(scene: Path) -> None:
    run_id = record(scene)
    bare = scene / "bare.yaml"
    bare.write_text("schema_version: 1\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["replay", run_id, "--journal", str(scene / "journal.db"), "--config", str(bare)],
    )
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "graph:" in result.output  # type: ignore[attr-defined]


def test_the_graph_reference_the_tests_use_is_the_one_the_config_names() -> None:
    """Otherwise a rename leaves every test above exercising a stale module quietly."""
    from specunode.runner import load_reference

    assert callable(load_reference(GRAPH_REFERENCE))


def test_init_writes_a_config_that_actually_parses(tmp_path: Path) -> None:
    """``specunode init`` from an installed wheel used to write 18 unusable bytes.

    The template was located at ``Path(__file__).parents[2]``, which is the checkout only when
    running from source; from an installed wheel that is ``lib/python3.11/``, the file was
    absent, and the ``pragma: no cover`` fallback wrote ``schema_version: 1`` and nothing else.
    The example was listed in neither the wheel nor the sdist include lists, so for every user
    who installed the package that fallback was the *only* path that ever ran.

    It lives inside the package now. This asserts the result parses and carries the keys that
    make it usable, rather than asserting a byte count.
    """
    result = CliRunner().invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output

    written = tmp_path / "specunode.yaml"
    assert written.is_file()
    text = written.read_text(encoding="utf-8")
    assert len(text) > 500, f"init wrote {len(text)} bytes; the fallback stub is back"

    from specunode.config import load_config

    config = load_config(written)
    assert config.schema_version == 1
    # The keys an operator needs in front of them. `graph` in particular is what `resume` and
    # `replay` refuse without, so a template that omits it teaches the wrong shape.
    for key in ("graph:", "target:", "journal:", "policy:", "tools:"):
        assert key in text, f"the template no longer documents {key!r}"


@pytest.mark.parametrize("command", ["replay", "resume", "mcp-proxy"])
def test_a_config_path_that_does_not_exist_is_refused_rather_than_replaced(
    tmp_path: Path, command: str
) -> None:
    """A typo in ``--config`` used to load a different file, or raise a traceback.

    ``mcp-proxy`` fell back to the search order -- ``./specunode.yaml``, then
    ``$XDG_CONFIG_HOME/specunode/config.yaml`` -- so it proxied with whatever override table
    that found and said nothing about which file it had read. ``resume`` and ``replay`` raised
    ``FileNotFoundError`` at the user. Both are now exit 2 naming the path.
    """
    absent = tmp_path / "nowhere.yaml"
    args = ["--config", str(absent)]
    if command == "mcp-proxy":
        args = ["mcp-proxy", "--upstream", "true", *args]
    else:
        args = [command, "01RUNTHATDOESNOTEXISTAAAAA", *args]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert str(absent) in result.output


def test_the_cli_reads_the_journal_the_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config's ``journal`` section was read by nothing: every command opened
    ``./.specunode/journal.db`` whatever it said. ``--journal`` still wins."""
    elsewhere = tmp_path / "books" / "journal.db"
    elsewhere.parent.mkdir()
    Journal(elsewhere).append("01CONFIGURED", "policy_event", {"v": 1, "event": "t", "reason": "r"})
    (tmp_path / "specunode.yaml").write_text(
        f"schema_version: 1\njournal:\n  path: {elsewhere}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    listed = CliRunner().invoke(app, ["runs"])
    assert listed.exit_code == 0, listed.output
    assert listed.output.split() == ["01CONFIGURED"]
    other = tmp_path / "other.db"
    Journal(other).append("01FLAGGED", "policy_event", {"v": 1, "event": "t", "reason": "r"})
    flagged = CliRunner().invoke(app, ["runs", "--journal", str(other)])
    assert flagged.output.split() == ["01FLAGGED"]


def test_resuming_an_unknown_run_is_a_message_not_a_traceback(scene: Path) -> None:
    """The refusal escaped as a traceback and exit 1 -- which says a run did not complete.
    Found by the twelfth review."""
    Journal(scene / "journal.db")  # a journal, without this run in it
    result = CliRunner().invoke(
        app,
        [
            "resume",
            "01NOSUCHRUNAAAAAAAAAAAAAAA",
            "--journal",
            str(scene / "journal.db"),
            "--config",
            str(scene / "specunode.yaml"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "no entries" in result.output
    assert isinstance(result.exception, SystemExit), result.exception


def test_a_journal_path_in_a_config_is_read_from_the_configs_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative to wherever the command ran, a config read from elsewhere named a journal
    nobody had written; and ``~`` was taken literally, making a folder named ``~``. Every
    command takes ``--config`` now, so all of them can read the journal ``resume`` used.
    Found by the twelfth review."""
    project = tmp_path / "project"
    (project / "books").mkdir(parents=True)
    Journal(project / "books" / "j.db").append(
        "01RELATIVE", "policy_event", {"v": 1, "event": "e", "reason": "r"}
    )
    (project / "specunode.yaml").write_text(
        "schema_version: 1\njournal:\n  path: books/j.db\n", encoding="utf-8"
    )
    home = tmp_path / "home"
    (home / "journals").mkdir(parents=True)
    Journal(home / "journals" / "h.db").append(
        "01TILDE", "policy_event", {"v": 1, "event": "e", "reason": "r"}
    )
    (tmp_path / "tilde.yaml").write_text(
        "schema_version: 1\njournal:\n  path: ~/journals/h.db\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)  # not the project: the path is the config's, not the shell's

    relative = CliRunner().invoke(app, ["runs", "--config", str(project / "specunode.yaml")])
    assert relative.output.split() == ["01RELATIVE"], relative.output
    tilde = CliRunner().invoke(app, ["runs", "--config", str(tmp_path / "tilde.yaml")])
    assert tilde.output.split() == ["01TILDE"], tilde.output
    assert not (tmp_path / "~").exists()


def test_a_config_that_does_not_load_says_how_to_name_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "specunode.yaml").write_text("schema_version: 1\nbogus_key: 1\n", encoding="utf-8")
    event = {"v": 1, "event": "e", "reason": "r"}
    Journal(tmp_path / "j.db").append("01NAMED", "policy_event", event)
    monkeypatch.chdir(tmp_path)
    refused = CliRunner().invoke(app, ["runs"])
    assert refused.exit_code == 2 and "--journal" in refused.output, refused.output
    named = CliRunner().invoke(app, ["runs", "--journal", str(tmp_path / "j.db")])
    assert named.exit_code == 0 and named.output.split() == ["01NAMED"], named.output


def test_a_config_that_names_no_journal_leaves_it_where_runs_write_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user-level config with no ``journal`` section sent every command to an empty journal
    beside it, created as it was read, while ``Runtime`` writes ``./.specunode/journal.db``.
    Only a path the config names is taken from the config's folder. Found by the thirteenth
    review."""
    xdg = tmp_path / "xdg"
    (xdg / "specunode").mkdir(parents=True)
    (xdg / "specunode" / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    work = tmp_path / "work"
    (work / ".specunode").mkdir(parents=True)
    event = {"v": 1, "event": "e", "reason": "r"}
    Journal(work / ".specunode" / "journal.db").append("01HERE", "policy_event", event)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.chdir(work)
    listed = CliRunner().invoke(app, ["runs"])
    assert listed.output.split() == ["01HERE"], listed.output
    assert not (xdg / "specunode" / ".specunode").exists()


@pytest.mark.parametrize("command", ["verify", "status", "ledger"])
def test_a_read_only_command_refuses_a_run_or_journal_that_is_not_there(
    tmp_path: Path, command: str
) -> None:
    """A mistyped run id was reported on as though it existed -- ``verify`` said the chain of
    0 entries verified -- and a mistyped journal path was created, empty, to report on. Found
    by the fifteenth review."""
    missing = tmp_path / "nowhere.db"
    absent = CliRunner().invoke(app, [command, "01TYPO", "--journal", str(missing)])
    assert absent.exit_code == 2 and "no journal at" in absent.output, absent.output
    assert not missing.exists(), "reading a journal created it"
    real = tmp_path / "journal.db"
    Journal(real).append("01REAL", "policy_event", {"v": 1, "event": "e", "reason": "r"})
    typo = CliRunner().invoke(app, [command, "01TYPO", "--journal", str(real)])
    assert typo.exit_code == 2 and "no entries" in typo.output, typo.output
