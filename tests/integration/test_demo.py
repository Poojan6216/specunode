"""Demo 1 is an acceptance test, not marketing (spec section 2).

The spec is explicit that the demos have to work from the published package, and that every
number they print comes from the run rather than from a table someone wrote. So the demo is
exercised here the same way a reader would exercise it, and the column that carries the claim
is asserted rather than eyeballed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_demo() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "demo.py"), "--demo", "leak", "--json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def arm(report: dict[str, object], name: str) -> dict[str, int]:
    arms = report["arms"]
    assert isinstance(arms, list)
    for entry in arms:
        if entry["runtime"] == name:
            return entry
    raise AssertionError(f"no {name} arm in the demo output")


def test_the_naive_runtime_leaks_and_the_demo_says_so() -> None:
    """Without a store buffer, a wrong guess has already charged the card.

    This arm exists to fail, and if it ever stops failing the demo has stopped demonstrating
    anything -- so a leak count of zero here is itself a failure.
    """
    naive = arm(run_demo(), "naive-parallel")
    assert naive["effects_from_squashed_branches"] > 0
    assert naive["effects_reaching_world"] > naive["runs"]


def test_specunode_reaches_the_world_exactly_as_the_sequential_run_does() -> None:
    """The second and third rows agreeing is the claim."""
    report = run_demo()
    spec = arm(report, "specunode")
    sequential = arm(report, "sequential")
    assert spec["effects_from_squashed_branches"] == 0
    assert spec["effects_reaching_world"] == sequential["effects_reaching_world"]


def test_the_speculative_arm_really_mispredicted(tmp_path: Path) -> None:
    """Otherwise it reached the sequential row by never having speculated at all."""
    spec = arm(run_demo(), "specunode")
    assert spec["mispredictions"] > 0
    assert spec["staged_and_discarded"] == spec["mispredictions"]


def test_the_committed_results_match_a_fresh_run() -> None:
    """Hard Rule 12: the numbers in the repo are the ones the command produces today."""
    committed = json.loads((REPO / "bench" / "results" / "demo_leak.json").read_text())
    assert committed == run_demo()


# -- Demo 2 ------------------------------------------------------------------------------------
#
# Nothing here asserts a wall-clock figure. Demo 2's timings are measured on the machine that
# runs it and vary between runs, so pinning one would either be flaky or be a number nobody
# measured. What is asserted is everything that is *not* a timing: the three arms agree about
# what reached the world, the specunode arm really overlapped the read, and the two baselines
# really did not.


def run_past_write() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "demo.py"), "--demo", "past-write", "--json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_all_three_arms_change_the_world_identically() -> None:
    """The claim the whole design rests on: faster, and identical in effect."""
    report = run_past_write()
    assert report["worlds_identical"] is True
    digests = {a["world_digest"] for a in report["arms"]}  # type: ignore[union-attr]
    assert len(digests) == 1
    for entry in report["arms"]:  # type: ignore[union-attr]
        assert entry["effects_reaching_world"] == 2


def test_the_specunode_arm_really_overlapped_the_read() -> None:
    """Without this, an arm that simply ran faster by accident would satisfy the table."""
    report = run_past_write()
    assert isinstance(report["read_overlapped_ms"], (int, float))
    assert report["read_overlapped_ms"] > 0, (
        "no part of the independent read ran while the write was staged, so the demo's "
        "explanation of where its saving comes from is not what happened"
    )


def test_the_readonly_baseline_did_not_run_ahead() -> None:
    """PASTE's rule, exercised rather than asserted in prose.

    Turn 1 emits the write first, so a runtime that stops at the first tool with side effects
    has nothing before it to run ahead into. If the read ever starts before the write finishes
    in that arm, the baseline is not implementing the rule it is named for.
    """
    report = run_past_write()
    arms = {a["runtime"]: a for a in report["arms"]}  # type: ignore[union-attr]
    calls = {c["name"]: c for c in arms["readonly-spec"]["calls"]}
    assert calls["fetch_runbook"]["started_ms"] >= calls["restart_job"]["finished_ms"]


def test_the_demo_reports_its_injected_latencies() -> None:
    """A timeline whose inputs are hidden is a drawing rather than a measurement."""
    injected = run_past_write()["injected_latency_ms"]
    assert isinstance(injected, dict)
    assert set(injected) == {"read", "write", "stream_block", "model_turn"}
    assert all(value > 0 for value in injected.values())


# -- Demo 3 ------------------------------------------------------------------------------------
#
# Demo 3 kills a real subprocess, so it is marked slow. What is asserted is the set of claims
# the demo makes in prose: the kill landed, the resume neither duplicated nor invented, the
# journal still verifies, replay refuses a changed prompt *with the diff*, and replay with
# speculation off completes.


def run_replay_demo() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(REPO / "bench" / "demo.py"), "--demo", "replay", "--json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


@pytest.mark.slow
def test_the_demo_actually_kills_the_process() -> None:
    """A kill demo that killed nothing would print every claim below and mean none of them."""
    report = run_replay_demo()
    assert report["process_was_killed"] is True
    delay, work = report["kill_delay_ms"], report["work_ms"]
    assert isinstance(delay, (int, float)) and isinstance(work, (int, float))
    assert 0 < delay < work, "the kill was not scheduled inside the run's own work"


@pytest.mark.slow
def test_the_resume_neither_duplicates_nor_invents() -> None:
    """The two properties that matter, stated the way the chaos suite states them."""
    report = run_replay_demo()
    assert report["applied_twice"] == 0
    # Only a tool that declared a repeat harmless may be handed its token again.
    assert set(report["absorbed_repeats"]) <= {"restart_job"}
    assert report["resumed_is_prefix_of_clean"] is True
    assert report["journal_chain_verifies_after_kill"] is True


@pytest.mark.slow
def test_replay_refuses_a_changed_prompt_and_says_what_changed() -> None:
    """Refusing is half of it. The spec asks for the step index and the diff, so check both.

    A refusal that only said "node act failed" would satisfy the boolean and be useless to the
    operator it exists for -- which is exactly what this reported before the scheduler was
    changed to carry the branch's failure reason out of the drive loop.
    """
    report = run_replay_demo()
    assert report["replay_with_changed_prompt_refused"] is True
    refusal = report["replay_refusal"]
    assert isinstance(refusal, str)
    assert "diverged at step" in refusal
    assert "system:" in refusal, "the refusal did not name the field that changed"
    assert "cautious operator" in refusal, "the refusal did not show the new value"


@pytest.mark.slow
def test_replay_with_speculation_off_completes() -> None:
    report = run_replay_demo()
    assert report["replay_with_speculation_off_ok"] is True
    assert report["replay_ledger_digest"]


def test_demo_ones_specunode_row_is_produced_by_the_real_runtime() -> None:
    """The front page's evidence table has to exercise the thing it is evidence for.

    This row used to be a conditional in ``bench/baselines.py``, which by design cannot import
    ``specunode.buffer`` -- the naive-parallel baseline has to be *able* to leak or the demo
    measures nothing. That isolation is right for the baselines and meant SpecuNode's own row
    was a description of a store buffer rather than a run of one. The numbers did not change
    when it was rewired, which is the point: they were not wrong, they were just not
    measurements of this software.

    Asserted structurally rather than by eye, because "did this number come from the product?"
    is exactly the question a table cannot answer about itself.
    """
    import bench.real_arm as real_arm

    source = (REPO / "bench" / "real_arm.py").read_text(encoding="utf-8")
    for required in (
        "from specunode.core.scheduler import Scheduler",
        "from specunode.buffer.store_buffer import StoreBuffer",
        "from specunode.buffer.dispatcher import Dispatcher",
    ):
        assert required in source, f"the specunode arm no longer uses the real runtime: {required}"
    assert hasattr(real_arm, "run_specunode_arm")

    # And the counts it reports come from the runtime's own counters, not from the arm's
    # arithmetic: a discarded effect is one the store buffer actually held back.
    spec = arm(run_demo(), "specunode")
    assert spec["staged_and_discarded"] == spec["mispredictions"] > 0
    assert spec["effects_from_squashed_branches"] == 0


@pytest.mark.slow
def demo3_scheduler(journal: object, world: object, skip: int = 0) -> Any:
    """Demo 3's run, in process, against ``journal`` -- so a test can kill it at an exact point."""
    sys.path.insert(0, str(REPO))
    from bench.demo import BLOCK_MS, TURN_1, TURN_2, TURN_MS, PastWriteGraph, demo3_registry

    from specunode.buffer.dispatcher import Dispatcher
    from specunode.buffer.store_buffer import StoreBuffer
    from specunode.core.model import JournaledModel
    from specunode.core.policy import Policy
    from specunode.core.scheduler import Scheduler
    from specunode.testing.models import ScriptedModel, tool_turn

    registry = demo3_registry(world)  # type: ignore[arg-type]
    return Scheduler(
        graph=PastWriteGraph("specunode"),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,  # type: ignore[arg-type]
        buffer=StoreBuffer(journal=journal, run_id=""),  # type: ignore[arg-type]
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=1.0),
        target=JournaledModel(
            ScriptedModel(
                turns=[tool_turn(*TURN_1, turn=0), tool_turn(*TURN_2, turn=1)],
                block_delay_ms=BLOCK_MS,
                complete_delay_ms=TURN_MS,
                consumed=skip,
            ),
            journal,  # type: ignore[arg-type]
            provider="scripted",
        ),
        policy=Policy(speculation=True),
    )


def test_a_kill_at_any_point_of_the_demo_resumes_to_a_prefix_without_duplicates(
    tmp_path: Path,
) -> None:
    """The demo draws one kill point, so a narrow window can hide from it -- one did, and CI
    found it. Here the kill walks the whole run instead, and every resume is held to the two
    properties that matter: nothing applied twice, and nothing the clean run did not do.

    Applied, not delivered: a kill that lands after ``restart_job`` reached the world and before
    its reply was recorded is resumed by handing it the same token again, because it declared a
    repeat harmless, and the world absorbs it. This test counted that as a duplicate, and so
    failed whenever a kill happened to land there -- rarely, which is how the claim it made
    survived as long as it did."""
    sys.path.insert(0, str(REPO))
    from bench.demo import _absorbed, _applied_twice, _delivered, _helper, _work_ms

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    work = _work_ms(_helper(clean_dir, "01CLEANAAAAAAAAAAAAAAAAAAA", -1)[1])
    clean = _delivered(clean_dir)
    killed = 0
    for point in range(24):
        directory = tmp_path / f"kill-{point}"
        directory.mkdir()
        run_id = f"01KILL{point:020d}"[:26]
        killed += _helper(directory, run_id, work * (0.02 + 0.96 * point / 23))[0] != 0
        _helper(directory, run_id, -1, resume=True)
        resumed = _delivered(directory)
        assert resumed == clean[: len(resumed)], (point, resumed, clean)
        assert _applied_twice(directory) == 0, point
        assert set(_absorbed(directory)) <= {"restart_job"}, (point, _absorbed(directory))
    assert killed >= 12, f"only {killed} of 24 kill points landed inside the run"


async def test_a_resume_asks_again_a_turn_whose_decision_sent_nothing(tmp_path: Path) -> None:
    """The window the sweep above can step over, hit exactly: the process dies after turn 1's
    reply is journaled and before turn 1's branch is confirmed, so nothing of turn 1 was sent.

    Nothing went out on turn 1's answer, so nothing needs protecting from a different one: the
    resume asks turn 1 again rather than being served the journaled reply, and the script must
    hand turn 1 over again. The demo's script once counted every recorded reply as answered, so
    the resume was handed turn 2 instead -- and posted a summary of a restart it never made.
    """
    import asyncio
    from collections.abc import Mapping

    sys.path.insert(0, str(REPO))
    from bench.demo import _delivered, answered_turns, demo3_world

    from specunode.canonical import JsonValue
    from specunode.ids import new_ulid
    from specunode.journal.journal import Journal

    class Died(BaseException):
        pass

    class DiesBeforeTurnOneIsConfirmed(Journal):
        replied = False
        dead = False

        async def append_async(
            self, run_id: str, kind: str, payload: Mapping[str, JsonValue]
        ) -> int:
            if self.dead:
                raise Died("a dead process writes nothing")
            self.replied = self.replied or kind == "model_response"
            if self.replied and kind == "branch_resolved" and payload.get("status") == "confirmed":
                self.dead = True
                raise Died("killed after turn 1's reply, before its branch was confirmed")
            return await super().append_async(run_id, kind, payload)

    run_id = new_ulid()
    world = demo3_world(tmp_path)
    with pytest.raises(Died):
        await demo3_scheduler(DiesBeforeTurnOneIsConfirmed(tmp_path / "journal.db"), world).run(
            run_id, {}
        )
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in leftovers:
        task.cancel()
    await asyncio.gather(*leftovers, return_exceptions=True)
    world.close()
    assert _delivered(tmp_path) == [], "the kill landed after turn 1 had already sent something"

    journal = Journal(tmp_path / "journal.db")
    assert len(list(journal.read(run_id, kinds=["model_response"]))) == 1
    skip = answered_turns(journal, run_id)
    assert skip == 0, "nothing of turn 1 was sent, so the resume asks it again"
    resumed_world = demo3_world(tmp_path)
    result = await demo3_scheduler(journal, resumed_world, skip).resume(run_id)
    resumed_world.close()
    assert result.ok, result.error
    assert [tool for tool, _ in _delivered(tmp_path)] == ["restart_job", "post_summary"]

    replies = [e.payload for e in journal.read(run_id, kinds=["model_response"])]
    assert not any("recorded_from" in reply for reply in replies), "a turn was served"
    assert len(replies) == 3, "turn 1, turn 1 asked again, and turn 2"


async def test_a_lost_reply_to_an_idempotent_write_is_redelivered_and_applied_once(
    tmp_path: Path,
) -> None:
    """The window the sweep hit by chance on a loaded machine, hit exactly.

    The process dies after ``restart_job`` reached the world and before its reply was recorded.
    Nobody can tell whether it took effect, and ``restart_job`` declared a repeat harmless, so
    the resume hands it the same token again -- the same token only because the resumed node is
    served the decision the dead one made -- and the world absorbs the repeat. Delivered twice,
    applied once: that is at-least-once dispatch doing what it says.
    """
    import asyncio
    from collections.abc import Mapping

    sys.path.insert(0, str(REPO))
    from bench.demo import _absorbed, _applied_twice, _delivered, answered_turns, demo3_world

    from specunode.canonical import JsonValue
    from specunode.ids import new_ulid
    from specunode.journal.journal import Journal

    class Died(BaseException):
        pass

    class DiesBeforeTheRestartIsAcked(Journal):
        dead = False

        async def settle_dispatch(self, **settled: Any) -> int:
            payload: Mapping[str, JsonValue] = settled["payload"]
            if self.dead or payload.get("tool") == "restart_job":
                self.dead = True
                raise Died("killed after the restart reached the world, before its reply")
            return await super().settle_dispatch(**settled)

        async def append_async(
            self, run_id: str, kind: str, payload: Mapping[str, JsonValue]
        ) -> int:
            if self.dead:
                raise Died("a dead process writes nothing")
            return await super().append_async(run_id, kind, payload)

    run_id = new_ulid()
    world = demo3_world(tmp_path)
    with pytest.raises(Died):
        await demo3_scheduler(DiesBeforeTheRestartIsAcked(tmp_path / "journal.db"), world).run(
            run_id, {}
        )
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in leftovers:
        task.cancel()
    await asyncio.gather(*leftovers, return_exceptions=True)
    world.close()
    assert [tool for tool, _ in _delivered(tmp_path)] == ["restart_job"]

    journal = Journal(tmp_path / "journal.db")
    resumed_world = demo3_world(tmp_path)
    result = await demo3_scheduler(journal, resumed_world, answered_turns(journal, run_id)).resume(
        run_id
    )
    resumed_world.close()
    assert result.ok, result.error
    assert [tool for tool, _ in _delivered(tmp_path)] == ["restart_job", "post_summary"]
    assert _absorbed(tmp_path) == ["restart_job"], "the lost reply was not redelivered"
    assert _applied_twice(tmp_path) == 0
