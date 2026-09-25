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
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger
from specunode.journal.replay import RecordedTurns, recover
from specunode.verify.equivalence import normalise_for_equivalence

HELPER = Path(__file__).with_name("_kill_agent.py")


def run_agent(
    directory: Path,
    run_id: str,
    kill: str = "-1",
    resuming: bool = False,
    changed_mind: bool = False,
    billing: bool = False,
) -> subprocess.CompletedProcess[str]:
    """``kill`` is ``-1``, ``op:N``, ``send:N`` or ``mutation:N``; see ``_kill_agent.py``."""
    args = [sys.executable, str(HELPER), str(directory), run_id, kill]
    if resuming:
        args.append("resume")
    if changed_mind:
        args.append("changed-mind")
    if billing:
        args.append("billing")
    return subprocess.run(args, capture_output=True, text=True, timeout=180)


def reported(result: subprocess.CompletedProcess[str], name: str) -> int:
    """A count the helper printed at the end of a run it finished."""
    for token in result.stdout.split():
        if token.startswith(f"{name}="):
            return int(token.split("=", 1)[1])
    raise AssertionError(f"the helper did not report {name}: {result.stdout!r}")


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


def clean_run(
    tmp_path: Path, *, changed_mind: bool = False, billing: bool = False
) -> tuple[list[tuple[str, str]], bytes, list[str]]:
    """A reference run: its effects, its normalised ledger, and every point it can die at."""
    directory = tmp_path / f"{'billing' if billing else 'support'}-{changed_mind}-clean"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = "01CLEANRUNAAAAAAAAAAAAAAAA"
    result = run_agent(directory, run_id, changed_mind=changed_mind, billing=billing)
    assert result.returncode == 0, result.stderr[-600:]
    ledger = build_ledger(Journal(directory / "journal.db"), run_id)
    effects = reported(result, "mutations")
    points = [f"op:{n}" for n in range(1, reported(result, "ops") + 1)]
    points += [f"send:{n}" for n in range(1, effects + 1)]
    points += [f"mutation:{n}" for n in range(1, effects + 1)]
    return logical_effects(directory), normalise_for_equivalence(ledger), points


@dataclass(frozen=True)
class OnDisk:
    """What a killed run left behind: everything a resume has to go on."""

    #: The run has at least one journal entry.
    journaled: bool
    #: The model's decision stands: its node retired, or something it decided may have been
    #: sent -- so a resume is served it rather than asking again.
    decision_stands: bool
    #: Tools with a dispatch claimed and no outcome recorded: sent, or maybe not.
    in_flight: tuple[str, ...]
    #: Tools whose effect the world durably applied.
    in_world: tuple[str, ...]


def on_disk(directory: Path, run_id: str, deciding_node: str) -> OnDisk:
    journal = Journal(directory / "journal.db")
    entries = list(journal.read(run_id))
    attempts = {
        str(entry.payload["branch_id"])
        for entry in entries
        if entry.kind == "branch_forked" and entry.payload.get("node_id") == deciding_node
    }
    kept = recover(journal, run_id).retired_branches | RecordedTurns(journal, run_id).acted
    return OnDisk(
        journaled=bool(entries),
        decision_stands=bool(attempts & kept),
        in_flight=tuple(str(row["tool"]) for row in journal.unresolved_dispatches(run_id)),
        in_world=tuple(tool for tool, _ in logical_effects(directory)),
    )


@pytest.mark.parametrize("billing", [False, True], ids=["support", "billing"])
def test_an_uninterrupted_run_is_the_reference(tmp_path: Path, billing: bool) -> None:
    effects, normalised, points = clean_run(tmp_path, billing=billing)
    assert [tool for tool, _ in effects] == ["charge_card", "send_receipt"]
    assert normalised
    # Both effects are points of their own, either side, and the journal writes around them.
    assert {"send:2", "mutation:2"} <= set(points) and len(points) > 10, points
    # The model that changes its mind really does decide something else.
    changed, _, _ = clean_run(tmp_path, changed_mind=True, billing=billing)
    assert [tool for tool, _ in changed] == ["charge_card", "send_receipt"]
    assert changed != effects


@pytest.mark.slow
@pytest.mark.parametrize(
    ("billing", "deciding_node"),
    [(False, "decide#0"), (True, "bill#0")],
    ids=["support", "billing"],
)
def test_killing_and_resuming_never_duplicates_and_never_invents(
    tmp_path: Path, billing: bool, deciding_node: str
) -> None:
    """Spec task 2.5's Verify, at every point the run can die, each with its exact outcome.

    The run is killed at each point in turn, and resumed by a process whose model would decide
    differently if it were asked -- a real model asked the same question twice may. What the
    resume must do is fixed by what the kill left on disk:

    * **nothing journaled** -- there is nothing to resume, and a resume that started the run
      afresh would dispatch its writes as if they were new, so it must refuse;
    * **a dispatch claimed with no outcome** -- the request may or may not have reached the
      world, and ``charge_card`` and ``send_receipt`` are declared ``idempotent=False``, so
      the runtime must not guess: it dead-letters exactly that effect and halts, with the world
      holding what came before it, and it too if it landed;
    * **anything else** -- the resume must finish, with exactly the effects of the run it is
      continuing: the dead process's decision if it stands -- its node retired, or something it
      decided may have been sent, so the resume is served it rather than asking again -- and
      the new one if not, since nothing of that decision can have left.

    And at every point, no idempotency key reaches the world twice.

    The spec asks that a resumed run's effects equal the uninterrupted run's. The dead letter is
    the one exception, and it is the property working: guessing that a charge whose reply was
    lost did not happen is how a card gets charged twice. A tool can end the exception by
    declaring how to ask the upstream (``reconcile``); these do not.

    Two graphs. The support agent decides in one node and sends in another, so a resume is
    never served a turn there: what ``decide`` said stands only once it retires. The billing
    node asks and charges in one step, which is where serving matters -- a crash after the
    charge and before the node retired must not let a model that changed its mind charge again.

    The points are every durable journal write, and either side of every world mutation -- each
    a place a process can die with what is on disk differing from the point before. They used
    to be fifteen random delays calibrated against a timed run: on a CI runner the same delays
    landed before the run had journaled anything on one attempt and after it had finished on
    another, and an outcome was checked only against "a prefix of the clean run" -- which a
    resume that sent nothing new at all would pass.
    """
    clean_effects, _normalised, points = clean_run(tmp_path, billing=billing)
    changed_effects, _, _ = clean_run(tmp_path, changed_mind=True, billing=billing)

    Ran = subprocess.CompletedProcess[str]

    def kill_then_resume(index: int) -> tuple[Path, str, Ran, OnDisk, Ran]:
        directory = tmp_path / points[index].replace(":", "-")
        directory.mkdir()
        run_id = f"01KILL{index:020d}"
        died = run_agent(directory, run_id, points[index], billing=billing)
        left = on_disk(directory, run_id, deciding_node)
        resumed = run_agent(directory, run_id, resuming=True, changed_mind=True, billing=billing)
        return directory, run_id, died, left, resumed

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(kill_then_resume, range(len(points))))

    refused = finished = halted = served = 0
    for point, (directory, run_id, died, left, resumed) in zip(points, outcomes, strict=True):
        assert died.returncode == 9 and "done" not in died.stdout, (
            f"{point}: the run was not killed there (exit {died.returncode}): {died.stdout}"
        )
        if not left.journaled:
            assert resumed.returncode != 0 and "nothing to resume" in resumed.stderr, (
                f"{point}: resuming a run with no journal entries did not refuse: "
                f"{resumed.stdout[-300:]} {resumed.stderr[-300:]}"
            )
            assert logical_effects(directory) == []
            refused += 1
            continue

        assert resumed.returncode == 0, f"resume failed at {point}: {resumed.stderr[-600:]}"
        journal = Journal(directory / "journal.db")
        assert journal.verify_chain(run_id).ok, f"chain broken after kill at {point}"
        replies = [entry.payload for entry in journal.read(run_id, kinds=["model_response"])]
        served += any("recorded_from" in reply for reply in replies)

        expected = clean_effects if left.decision_stands else changed_effects
        reached = logical_effects(directory)
        dead = tuple(
            row.call.name
            for row in build_ledger(journal, run_id).rows
            if row.status == "DEAD_LETTER"
        )
        if left.in_flight:
            ambiguous = left.in_flight[0]
            before = [tool for tool, _ in expected].index(ambiguous)
            landed = ambiguous in left.in_world
            assert "ok=False" in resumed.stdout and dead == left.in_flight, (
                f"{point}: {ambiguous} was claimed with no outcome, and the resume did not "
                f"stop for a human on it: {resumed.stdout.strip()} dead={dead}"
            )
            assert reached == expected[: before + landed], (
                f"{point}: halted on {ambiguous}, and the world holds {reached}; it should hold "
                f"{expected[: before + landed]}"
            )
            halted += 1
        else:
            assert "ok=True" in resumed.stdout and not dead, (
                f"{point}: nothing was ambiguous, and the resume did not finish: "
                f"{resumed.stdout.strip()} dead={dead}"
            )
            assert reached == expected, (
                f"{point}: the resumed run's effects are not the run it continues -- "
                f"{'the decision that stands' if left.decision_stands else 'a new decision'}:\n"
                f"  resumed: {reached}\n  expected: {expected}"
            )
            finished += 1

        here = duplicate_deliveries(directory)
        assert here == 0, f"{point}: {here} effect(s) reached the world twice"

    assert refused == 1, f"{refused} kill points left nothing to resume; expected only op:1"
    assert finished >= 1 and halted >= 1, (finished, halted)
    if billing:
        assert served >= 1, "no resume was served a journaled turn, so that was never exercised"
    else:
        assert served == 0, "decide sends nothing, so nothing of its answer needs serving"
    print(
        f"\nkill/resume over {len(points)} points: {finished} finished, {halted} stopped for a "
        f"human on an ambiguous effect, {refused} refused with nothing to resume; {served} "
        "resumes were served the journaled decision; 0 duplicate deliveries"
    )


@pytest.mark.slow
def test_a_resumed_run_does_not_repeat_an_acked_effect(tmp_path: Path) -> None:
    """The dedupe table, across a process boundary."""
    directory = tmp_path / "once"
    directory.mkdir()
    run_id = "01ONCERUNAAAAAAAAAAAAAAAAA"
    assert run_agent(directory, run_id).returncode == 0
    before = logical_effects(directory)
    deliveries_before = len([e for e in world_log(directory) if e.get("kind") == "mutation"])

    assert run_agent(directory, run_id, resuming=True).returncode == 0
    assert logical_effects(directory) == before
    assert (
        len([e for e in world_log(directory) if e.get("kind") == "mutation"]) == deliveries_before
    ), "a resume of a finished run delivered something again"
