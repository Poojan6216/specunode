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
