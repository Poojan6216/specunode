"""Resume (spec task 2.5) and the CLI's read-only views (task 2.6).

A resumed run must not re-send what already went out, and must not build on work that was
thrown away. Both halves are tested here: the dedupe table stops the first, and the recovery
rule -- only branches the journal records as RETIRED contribute state or steps -- stops the
second.

The second half is easy to get wrong in a way no other test notices. A branch that was
confirmed but never retired had its drain in flight when the process died; resuming from it
would dispatch effects that were never context-checked, and the ledger would look exactly like
one from a clean run.
"""

from __future__ import annotations

from pathlib import Path

from examples.support_agent.agent import build

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.effects import ToolRegistry
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.journal.replay import attested_origins, recover
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})


def make(tmp_path: Path, world: World, db: str = "journal.db") -> tuple[Scheduler, Journal]:
    adapter, registry = build(world)
    assert isinstance(registry, ToolRegistry)
    journal = Journal(tmp_path / db)
    model = ScriptedModel(turns=[tool_turn(CHARGE, turn=0), tool_turn(CHARGE, turn=1)])
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=False),
    )
    return scheduler, journal


async def test_a_finished_run_reports_itself_finished(tmp_path: Path) -> None:
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})

    recovery = recover(journal, run_id)
    assert recovery.finished
    assert recovery.retired_branches
    assert recovery.confirmed_not_retired == ()
    assert recovery.unresolved_dispatches == ()


async def test_resuming_a_finished_run_sends_nothing_again(tmp_path: Path) -> None:
    """The dedupe table, not the scheduler's memory, is what makes this safe."""
    world = standard_world()
    scheduler, _journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    before = [(m.tool, m.args_hash) for m in world.mutations]
    assert before

    resumed, _ = make(tmp_path, world)
    result = await resumed.resume(run_id)
    after = [(m.tool, m.args_hash) for m in world.mutations]
    assert after == before, "a resume must not re-send an effect that already went out"
    assert result.ok


async def test_recovery_rebuilds_committed_state_from_retired_branches(tmp_path: Path) -> None:
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    result = await scheduler.run(run_id, {"customer_id": "cus-1"})

    recovery = recover(journal, run_id)
    assert recovery.state.get("receipted") is True
    assert recovery.state.get("customer") == result.state.get("customer")


async def test_recovery_continues_the_step_counter_rather_than_restarting_it(
    tmp_path: Path,
) -> None:
    """Hard Rule 8: a key derived after a crash must equal the one derived before it."""
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    assert recover(journal, run_id).step_index > 0


async def test_a_confirmed_but_unretired_branch_is_not_built_on(tmp_path: Path) -> None:
    """Its drain was in flight when the process died, so its work is evidence, not input."""
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})

    # A branch that reached confirmed and stopped there, as a crash mid-drain leaves one.
    await journal.append_async(
        run_id,
        "branch_resolved",
        {"v": 1, "branch_id": "br-mid-drain", "step": 99, "status": "confirmed"},
    )
    await journal.append_async(
        run_id,
        "state_delta_applied",
        {
            "v": 1,
            "branch_id": "br-mid-drain",
            "step": 99,
            "patch": [{"op": "add", "path": "/ghost", "value": True}],
            "patch_hash": "h",
            "result_state_hash": "h",
        },
    )

    recovery = recover(journal, run_id)
    assert "br-mid-drain" in recovery.confirmed_not_retired
    assert "ghost" not in recovery.state, "an unretired branch's delta must not be applied"
    assert recovery.step_index < 99, "nor may its steps move the counter"


# -- attestation (correction C1) -------------------------------------------------------


async def test_attested_origins_spans_the_retired_chain(tmp_path: Path) -> None:
    """Lineage resets at every retirement, so a lineage-only filter hides earlier turns.

    A rebuild under that filter matches nothing, every branch reports a context divergence,
    and the step is redone forever. This is the union that stops it.
    """
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})

    retired = {
        str(e.payload["branch_id"])
        for e in journal.read(run_id, kinds=["branch_resolved"])
        if e.payload.get("status") == "retired"
    }
    attested = attested_origins(journal, run_id, lineage=["br-current"], upto_step=1000)
    assert retired <= attested, "every retired branch's output is readable back"
    assert "br-current" in attested, "so is this branch's own lineage"


async def test_an_unresolved_sibling_is_not_attested(tmp_path: Path) -> None:
    """The channel Hard Rule 6 exists to close.

    The tempting predicate once a lineage-only filter fails is 'anything not squashed', and it
    admits exactly this: a sibling that has not resolved yet.
    """
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    await journal.append_async(
        run_id,
        "branch_forked",
        {
            "v": 1,
            "branch_id": "br-sibling",
            "lineage": ["br-sibling"],
            "fork_step": 1,
            "predicted_hash": "h",
            "tier": 1,
        },
    )
    attested = attested_origins(journal, run_id, lineage=["br-current"], upto_step=1000)
    assert "br-sibling" not in attested


async def test_a_squashed_branch_is_not_attested(tmp_path: Path) -> None:
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})
    await journal.append_async(
        run_id,
        "branch_resolved",
        {"v": 1, "branch_id": "br-lost", "step": 1, "status": "squashed", "reason": "mismatch"},
    )
    assert "br-lost" not in attested_origins(journal, run_id, lineage=[], upto_step=1000)


async def test_the_state_delta_is_durable_before_the_retirement_that_depends_on_it(
    tmp_path: Path,
) -> None:
    """Ordering, not bookkeeping: the other order loses a crash window with teeth.

    If ``branch_resolved{retired}`` lands before ``state_delta_applied`` and the process dies
    between them, the branch reads as retired while its state change is gone. A resume then
    re-runs that node -- from a different program position, deriving different idempotency
    keys, so the dedupe table misses and effects that already went out go out again. It looks
    like a resume bug and is an ordering bug, which is why it is pinned here rather than left
    to the chaos test to rediscover.
    """
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})

    order = [
        (entry.offset, entry.kind, entry.payload.get("branch_id"), entry.payload.get("status"))
        for entry in journal.read(run_id)
        if entry.kind in ("state_delta_applied", "branch_resolved")
    ]
    for offset, kind, branch_id, status in order:
        if kind != "branch_resolved" or status != "retired":
            continue
        deltas = [
            other_offset
            for other_offset, other_kind, other_branch, _ in order
            if other_kind == "state_delta_applied" and other_branch == branch_id
        ]
        assert all(delta_offset < offset for delta_offset in deltas), (
            f"branch {branch_id} was journaled retired at offset {offset} before its state "
            f"delta at {deltas}"
        )


async def test_the_cursor_is_restored_from_the_journal_not_inferred(tmp_path: Path) -> None:
    """A resume must land on exactly the program position the last retirement committed."""
    world = standard_world()
    scheduler, journal = make(tmp_path, world)
    run_id = new_ulid()
    await scheduler.run(run_id, {"customer_id": "cus-1"})

    retirements = [
        entry.payload
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    ]
    assert retirements, "a completed run retires at least one branch"
    last = retirements[-1].get("cursor_after")
    assert isinstance(last, dict) and "step_index" in last and "visits" in last

    recovery = recover(journal, run_id)
    assert recovery.cursor.step_index == last["step_index"]
    assert [list(v) for v in recovery.cursor.visits] == [list(v) for v in last["visits"]]
