"""Kill a run part-way through, resume it, and compare (spec task 2.5).

The claim is that a resumed run reaches the same place as one that was never interrupted, and
does not re-send what already went out. Both halves need a world that survives the kill, which
is why ``World`` keeps an append-only log: a resumed process that could not see the dead one's
effects could not avoid repeating them, and the test would pass while the card was charged
twice.

**What "the same" means, stated honestly.** Dispatch is at-least-once. A crash in the window
between a request leaving the process and its ack being recorded is the two-generals case, and
the resumed run may deliver that one effect again -- the world log then has one more *delivery*
than the clean run. So the comparison is over logical effects, keyed by idempotency key, and
duplicate deliveries are counted and reported rather than normalised away. A tool that declared
itself idempotent absorbs them; one that did not is dead-lettered instead of guessed at, which
is why a duplicated charge cannot happen silently in either direction.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger
from specunode.verify.equivalence import normalise_for_equivalence

HELPER = Path(__file__).with_name("_kill_agent.py")


def run_agent(
    directory: Path, run_id: str, delay_ms: float, resuming: bool = False
) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, str(HELPER), str(directory), run_id, str(delay_ms)]
    if resuming:
        args.append("resume")
    return subprocess.run(args, capture_output=True, text=True, timeout=180)


def work_ms_of(result: subprocess.CompletedProcess[str]) -> float:
    """How long the run itself took, excluding interpreter startup and imports."""
    for token in result.stdout.split():
        if token.startswith("work_ms="):
            return float(token.split("=", 1)[1])
    raise AssertionError(f"helper did not report its work duration: {result.stdout!r}")


def world_log(directory: Path) -> list[dict[str, object]]:
    path = directory / "world.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a torn final line: the process died mid-append
    return events


def logical_effects(directory: Path) -> list[tuple[str, str]]:
    """Distinct effects that reached the world, in order, keyed by idempotency key."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for event in world_log(directory):
        if event.get("kind") != "mutation":
            continue
        key = str(event.get("effect_key", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append((str(event.get("tool")), str(event.get("args_hash"))))
    return out


def duplicate_deliveries(directory: Path) -> int:
    keys = [
        str(e.get("effect_key"))
        for e in world_log(directory)
        if e.get("kind") == "mutation" and e.get("effect_key")
    ]
    return len(keys) - len(set(keys))


def clean_run(tmp_path: Path) -> tuple[Path, str, list[tuple[str, str]], bytes]:
    directory = tmp_path / "clean"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = "01CLEANRUNAAAAAAAAAAAAAAAA"
    assert run_agent(directory, run_id, -1).returncode == 0
    ledger = build_ledger(Journal(directory / "journal.db"), run_id)
    return directory, run_id, logical_effects(directory), normalise_for_equivalence(ledger)


def test_an_uninterrupted_run_is_the_reference(tmp_path: Path) -> None:
    _directory, _run_id, effects, normalised = clean_run(tmp_path)
    assert [tool for tool, _ in effects] == ["charge_card", "send_receipt"]
    assert normalised


@pytest.mark.slow
def test_killing_and_resuming_never_duplicates_and_never_invents(tmp_path: Path) -> None:
    """Spec task 2.5's Verify, at fifteen kill points, with one clause stated honestly.

    The spec asks that the resumed run's effects equal the uninterrupted run's. That holds at
    every kill point except one class, and the exception is not a defect -- it is the property
    working. If the process dies between a request reaching the world and its acknowledgement
    being recorded, nobody can tell afterwards whether it took effect. ``charge_card`` is
    declared ``idempotent=False``, so the runtime refuses to guess: it dead-letters that effect
    and halts rather than risk a second charge. The run then reaches a PREFIX of the clean
    run's effects, and says so in its ledger.

    Declaring the tool idempotent would make the equality hold at every point, by redelivering
    a charge that may already have gone through. That is the trade this test exists to make
    visible rather than to hide, so what is asserted is the pair of properties that actually
    matter:

    * **never duplicated** -- no idempotency key reaches the world twice, at any kill point
    * **never invented** -- the effects reached are always a prefix of the clean run's, in
      order, so a resume can fall short but can never do something the clean run did not

    Logged as a deviation from the Verify's literal wording in the Progress Log.
    """
    _clean_dir, _clean_run, clean_effects, _clean_normalised = clean_run(tmp_path)

    # The fastest of several runs, not one: the first run on a cold machine is the slowest,
    # and a window taken from it put most delays past the end of every later, faster run --
    # on CI only 2 of 15 processes were killed.
    windows = []
    for warm in range(3):
        calibration = tmp_path / f"calibrate-{warm}"
        calibration.mkdir()
        windows.append(work_ms_of(run_agent(calibration, f"01CALIB{warm:019d}"[:26], -1)))
    work_ms = min(windows)
    assert work_ms > 1.0, f"the run takes {work_ms:.2f}ms; too fast to land a kill inside it"

    rng = random.Random(20260916)
    killed = 0
    complete = 0
    halted = 0
    duplicates = 0
    for attempt in range(15):
        directory = tmp_path / f"kill-{attempt}"
        directory.mkdir()
        run_id = f"01KILL{attempt:020d}"[:26]
        delay_ms = rng.uniform(0.05, work_ms * 0.9)
        if run_agent(directory, run_id, delay_ms).returncode != 0:
            killed += 1

        resumed = run_agent(directory, run_id, -1, resuming=True)
        assert resumed.returncode == 0, f"resume failed at {attempt}: {resumed.stderr[-600:]}"

        journal = Journal(directory / "journal.db")
        assert journal.verify_chain(run_id).ok, f"chain broken after kill {attempt}"

        reached = logical_effects(directory)
        assert reached == clean_effects[: len(reached)], (
            f"kill {attempt}: the resumed run reached effects the clean run never did:\n"
            f"  resumed: {reached}\n  clean:   {clean_effects}"
        )
        if reached == clean_effects:
            complete += 1
        else:
            halted += 1
            ledger = build_ledger(journal, run_id)
            assert any(row.status == "DEAD_LETTER" for row in ledger.rows), (
                f"kill {attempt}: the run fell short of the clean run without dead-lettering "
                "anything, so it stopped for a reason it did not record"
            )

        here = duplicate_deliveries(directory)
        assert here == 0, f"kill {attempt}: {here} effect(s) reached the world twice"
        duplicates += here

    assert killed >= 5, (
        f"only {killed}/15 subprocesses were actually killed mid-run; the test would be green "
        "without exercising the window it exists to test"
    )
    assert complete >= 1, "no kill point resumed to completion; the resume path is not exercised"
    assert duplicates == 0
    print(
        f"\nkill/resume over 15 points: {killed} killed, {complete} resumed to completion, "
        f"{halted} halted on an ambiguous effect, {duplicates} duplicate deliveries"
    )


@pytest.mark.slow
def test_a_resumed_run_does_not_repeat_an_acked_effect(tmp_path: Path) -> None:
    """The dedupe table, across a process boundary."""
    directory = tmp_path / "once"
    directory.mkdir()
    run_id = "01ONCERUNAAAAAAAAAAAAAAAAA"
    assert run_agent(directory, run_id, -1).returncode == 0
    before = logical_effects(directory)
    deliveries_before = len([e for e in world_log(directory) if e.get("kind") == "mutation"])

    assert run_agent(directory, run_id, -1, resuming=True).returncode == 0
    assert logical_effects(directory) == before
    assert (
        len([e for e in world_log(directory) if e.get("kind") == "mutation"]) == deliveries_before
    ), "a resume of a finished run delivered something again"
