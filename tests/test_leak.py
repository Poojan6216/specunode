"""THE LEAK TEST (spec task 1.6, Hard Rule 3). Mandatory. Never skipped. Never xfailed.

The whole project reduces to one claim: **nothing reaches the world from a branch that did not
retire.** Every other property -- replay, equivalence, the ledger -- is bookkeeping around that.
So this test generates random branch trees with random resolution outcomes under random faults,
and after every single run asserts it.

Two invariants are checked, because the spec's own statement of the property does not catch the
bug the spec names as the planted one:

``I1`` -- the spec's, verbatim
    ``{m.branch_id for m in world.mutations} ⊆ {b.id for b in branches if b.status is RETIRED}``

``I2`` -- authorisation
    every mutation traces to a ``branch_resolved{confirmed}`` entry that was **durable before
    the effect was dispatched**. I1 alone cannot see a drain that happened before its
    confirming entry was fsynced, because in a single process the branch is CONFIRMED either
    way and the mutation is attributed to a branch that does retire. That is precisely the bug
    task 1.6 says a planted version of must make this file fail, so it needs an invariant that
    can see it.

Both invariants have planted-bug proofs at the bottom: the check is disabled, the same
generator runs, and the test asserts the invariant *fails*. A leak test that would pass with
the safety mechanism removed is not evidence of anything.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from bench.workloads import WORKLOADS, Workload
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.branch import Branch, BranchClosed, BranchStatus
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec, forward_keys_from_template
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.world import World, standard_world

WRITE_TOOLS = ("restart_job", "charge_card", "post_summary")


# -- the generated shape ------------------------------------------------------------------------


@dataclass(frozen=True)
class StepPlan:
    """One decision point: how many siblings fork, which one the model confirms, what breaks."""

    siblings: int
    #: Index of the sibling the model's real decision matches, or None -- every guess wrong,
    #: so the canonical path proceeds without a speculative winner.
    winner: int | None
    effects_per_branch: tuple[int, ...]
    tools: tuple[str, ...]


step_plans = st.builds(
    lambda siblings, winner_seed, effects, tools: StepPlan(
        siblings=siblings,
        winner=None if winner_seed is None else winner_seed % siblings,
        effects_per_branch=tuple(effects[:siblings]) + (0,) * max(0, siblings - len(effects)),
        tools=tuple(tools[:8]) or ("restart_job",),
    ),
    siblings=st.integers(min_value=1, max_value=3),
    winner_seed=st.one_of(st.none(), st.integers(min_value=0, max_value=5)),
    effects=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=3),
    tools=st.lists(st.sampled_from(WRITE_TOOLS), min_size=1, max_size=8),
)

fault_plans = st.fixed_dictionaries(
    {
        "partition_at": st.one_of(st.none(), st.integers(min_value=1, max_value=6)),
        "duplicate": st.one_of(st.none(), st.sampled_from(WRITE_TOOLS)),
        "timeout": st.one_of(st.none(), st.sampled_from(WRITE_TOOLS)),
    }
)


def _registry(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="restart_job",
            effect=EffectClass.WRITE,
            fn=world.restart_job,
            idempotent=True,
            forward_keys=forward_keys_from_template("job:{args.job_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="charge_card",
            effect=EffectClass.WRITE,
            fn=world.charge_card,
            forward_keys=forward_keys_from_template("customer:{args.customer_id}"),
        )
    )
    registry.register(
        ToolSpec(
            name="post_summary",
            effect=EffectClass.IRREVERSIBLE,
            fn=world.post_summary,
            forward_keys=forward_keys_from_template("channel:{args.channel}"),
        )
    )
    return registry


def _args(tool: str, seed: int) -> dict[str, object]:
    if tool == "restart_job":
        return {"job_id": f"etl-{seed % 4 + 1}"}
    if tool == "charge_card":
        return {"customer_id": f"cus-{seed % 5 + 1}", "amount": float(seed % 7 + 1)}
    return {"channel": f"#ops-{seed % 3}", "text": f"note {seed}"}


@dataclass
class RunResult:
    world: World
    journal: Journal
    run_id: str
    branches: list[Branch]


async def simulate(
    journal_path: Path,
    plans: list[StepPlan],
    faults: dict[str, object],
    *,
    skip_status_check: bool = False,
    skip_durability_check: bool = False,
    drain_before_confirm: bool = False,
) -> RunResult:
    """Drive a branch tree to completion, staging and resolving as the real scheduler would."""
    run_id = new_ulid()
    journal = Journal(journal_path)
    world = standard_world()
    registry = _registry(world)
    dispatcher = Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5, cap_delay_ms=1.0)
    buffer = StoreBuffer(journal=journal, run_id=run_id)

    if faults["partition_at"] is not None:
        world.partition(at=int(faults["partition_at"]))  # type: ignore[arg-type]
    if faults["duplicate"] is not None:
        world.duplicate_delivery(str(faults["duplicate"]))
    if faults["timeout"] is not None:
        world.timeout(str(faults["timeout"]))

    canon = Branch(id=new_ulid(), status=BranchStatus.CONFIRMED)
    canon.context_verified = True
    branches: list[Branch] = [canon]
    seed = 0

    for step, plan in enumerate(plans):
        # Fork the siblings and stage each one's effects. Nothing may reach the world here.
        siblings: list[Branch] = []
        for index in range(plan.siblings):
            child = canon.fork(
                new_ulid(), predicted=ToolCall(plan.tools[0], {}), step=step, tier=index
            )
            child.cursor = child.cursor.advance(step + 1)
            siblings.append(child)
            branches.append(child)
            for _ in range(plan.effects_per_branch[index]):
                seed += 1
                tool = plan.tools[seed % len(plan.tools)]
                # One program position per call, as the real tool port takes.
                child.advance_step()
                await buffer.stage(child, ToolCall(tool, _args(tool, seed)), registry.get(tool))

        before = len(world.mutations)
        assert len(world.mutations) == before, "staging must not reach the world"

        # The model's real decision arrives. At most one sibling matches; the rest squash.
        winner = siblings[plan.winner] if plan.winner is not None else None
        for sibling in siblings:
            if sibling is winner:
                continue
            sibling.squash("mismatch")
            await buffer.discard_and_journal(sibling, "squashed")

        if winner is None:
            continue

        winner.confirm()
        winner.context_verified = True
        if drain_before_confirm:
            # The confirming entry is NOT written yet. That is the bug: the drain below runs
            # first, against the offset the entry is about to take.
            confirmed_offset = (journal.last_offset(run_id) or -1) + 1
        else:
            confirmed_offset = await journal.append_async(
                run_id,
                "branch_resolved",
                {"v": 1, "branch_id": winner.id, "step": step, "status": "confirmed"},
            )
        if skip_durability_check:
            # PLANTED BUG: claim the confirming entry is at an offset that does not exist yet.
            # Kept because it is a different failure -- a dispatch against nothing at all --
            # but it is strictly easier to catch than the one above.
            confirmed_offset = (journal.last_offset(run_id) or 0) + 1000
        await _drain(
            buffer,
            winner,
            dispatcher,
            confirmed_offset=confirmed_offset,
            skip_status_check=skip_status_check,
            skip_durability_check=skip_durability_check or drain_before_confirm,
        )
        if drain_before_confirm:
            # Appended now, at the offset the drain already claimed. The finished journal is
            # complete and self-consistent -- every entry present, every offset real. Only the
            # order betrays it, which is why an invariant that does not compare offsets cannot
            # see this and an invariant that fabricates a missing offset never had to.
            await journal.append_async(
                run_id,
                "branch_resolved",
                {"v": 1, "branch_id": winner.id, "step": step, "status": "confirmed"},
            )
        winner.retire()
        await journal.append_async(
            run_id,
            "branch_resolved",
            {"v": 1, "branch_id": winner.id, "step": step, "status": "retired"},
        )
        # Retirement is terminal, and lineage resets at each one: the next step forks from a
        # fresh canonical frontier carrying the committed cursor, not from the retired branch.
        canon = Branch(id=new_ulid(), status=BranchStatus.CONFIRMED, cursor=winner.cursor)
        canon.context_verified = True
        branches.append(canon)

    return RunResult(world=world, journal=journal, run_id=run_id, branches=branches)


async def _drain(
    buffer: StoreBuffer,
    branch: Branch,
    dispatcher: Dispatcher,
    *,
    confirmed_offset: int,
    skip_status_check: bool,
    skip_durability_check: bool,
) -> None:
    if skip_status_check or skip_durability_check:
        # The planted-bug path: reach the drain body with a precondition removed.
        await buffer._drain_locked(branch, dispatcher, confirmed_offset, confirmed_offset)
        return
    await buffer.drain(
        branch,
        dispatcher,
        confirmed_offset=confirmed_offset,
        authorised_by_offset=confirmed_offset,
    )


# -- the invariants ------------------------------------------------------------------------------


def leaked_branches(result: RunResult) -> set[str]:
    """I1: branches that touched the world without retiring."""
    retired = {b.id for b in result.branches if b.status is BranchStatus.RETIRED}
    touched = {m.branch_id for m in result.world.mutations} - {"<external>"}
    return touched - retired


def unauthorised_effects(result: RunResult) -> list[str]:
    """I2: effects whose confirming journal entry was not durable when they were dispatched.

    The property is an **ordering** one, and both facts are in the journal: the confirming
    ``branch_resolved{confirmed}`` entry and the ``effect_dispatched`` entry each have an
    offset, and the first must come before the second. If it does not, the effect left before
    anything authorised it, and a crash at that instant leaves a world that changed for a
    decision the journal never recorded.

    This used to check two much weaker things -- that the branch has *some* confirming entry
    anywhere in the finished journal, and that the offset the dispatch names is not past the
    *final* head. An entry appended after the dispatch satisfies both. The planted-bug proof
    passed only because it fabricated ``head + 1000``, an offset that never exists at all, which
    is a strictly stronger falsification than the bug being modelled: plant the real shape --
    drain against the offset the confirming entry is about to occupy, then append it -- and both
    invariants reported clean. ``test_i2_catches_the_real_shape_of_the_planted_bug`` now plants
    exactly that.
    """
    confirmed_at: dict[str, int] = {}
    for entry in result.journal.read(result.run_id):
        if entry.kind == "branch_resolved" and entry.payload.get("status") == "confirmed":
            branch_id = entry.payload.get("branch_id")
            if isinstance(branch_id, str):
                confirmed_at.setdefault(branch_id, entry.offset)

    bad: list[str] = []
    for entry in result.journal.read(result.run_id, kinds=["effect_dispatched"]):
        branch_id = entry.payload.get("branch_id")
        claimed = entry.payload.get("confirmed_by_offset")
        effect_id = entry.payload.get("effect_id")
        if not isinstance(branch_id, str) or branch_id not in confirmed_at:
            bad.append(f"{effect_id}: no confirming entry for {branch_id}")
            continue
        confirmed_offset = confirmed_at[branch_id]
        if confirmed_offset >= entry.offset:
            bad.append(
                f"{effect_id}: dispatched at offset {entry.offset}, but its branch was only "
                f"confirmed at offset {confirmed_offset} -- the effect left before anything "
                "authorised it"
            )
        elif isinstance(claimed, int) and claimed != confirmed_offset:
            bad.append(
                f"{effect_id}: dispatched against offset {claimed}, which is not its branch's "
                f"confirming entry (that is at {confirmed_offset})"
            )
    return bad


# -- the test ------------------------------------------------------------------------------


#: A plan that definitely stages and definitely retires, used where a test needs effects to
#: actually reach the world rather than relying on what the generator happened to draw.
ALWAYS_LEAKS = [
    StepPlan(siblings=2, winner=0, effects_per_branch=(2, 2), tools=("charge_card", "restart_job"))
]
NO_FAULTS: dict[str, object] = {"partition_at": None, "duplicate": None, "timeout": None}


@pytest.fixture(scope="module")
def shared_journal(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One journal file across every example, which exercises the concurrent-run path too."""
    return tmp_path_factory.mktemp("leak") / "journal.db"


@given(
    plans=st.lists(step_plans, min_size=1, max_size=5),
    faults=fault_plans,
)
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_no_effect_ever_reaches_the_world_from_an_unretired_branch(
    shared_journal: Path, plans: list[StepPlan], faults: dict[str, object]
) -> None:
    result = asyncio.run(simulate(shared_journal, plans, faults))

    assert not leaked_branches(result), (
        f"Hard Rule 3 violated: {leaked_branches(result)} touched the world without retiring"
    )
    assert not unauthorised_effects(result), (
        "an effect was dispatched without a durable confirming entry:\n"
        + "\n".join(unauthorised_effects(result))
    )


def test_the_generator_actually_reaches_the_world(tmp_path: Path) -> None:
    """A leak test that never dispatched anything would hold its invariant vacuously."""
    result = asyncio.run(simulate(tmp_path / "j.db", ALWAYS_LEAKS, NO_FAULTS))
    assert result.world.mutations, "the generator produced no effects at all"
    assert not leaked_branches(result)
    assert not unauthorised_effects(result)
    retired = {b.id for b in result.branches if b.status is BranchStatus.RETIRED}
    assert {m.branch_id for m in result.world.mutations} == retired


# -- planted-bug proofs -------------------------------------------------------------------------
#
# Task 1.6's Verify: a deliberately planted bug must make this file fail. Without these, a leak
# test that had silently stopped exercising anything would stay green forever.


def test_the_invariant_catches_a_drain_on_a_squashed_branch(tmp_path: Path) -> None:
    async def leak() -> RunResult:
        run_id = new_ulid()
        journal = Journal(tmp_path / "j.db")
        world = standard_world()
        registry = _registry(world)
        buffer = StoreBuffer(journal=journal, run_id=run_id)
        dispatcher = Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.1)

        doomed = Branch(id=new_ulid(), status=BranchStatus.SPECULATIVE)
        await buffer.stage(
            doomed, ToolCall("charge_card", _args("charge_card", 1)), registry.get("charge_card")
        )
        doomed.squash("mismatch")
        offset = await journal.append_async(
            run_id,
            "branch_resolved",
            {"v": 1, "branch_id": doomed.id, "step": 0, "status": "squashed"},
        )
        # PLANTED BUG: dispatch the squashed branch's buffer anyway.
        await buffer._drain_locked(doomed, dispatcher, offset, offset)
        return RunResult(world=world, journal=journal, run_id=run_id, branches=[doomed])

    result = asyncio.run(leak())
    assert leaked_branches(result), "I1 did not notice a squashed branch reaching the world"


def test_the_invariant_catches_a_dispatch_against_an_offset_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """The easy half: a dispatch claiming an offset nothing will ever occupy."""
    result = asyncio.run(
        simulate(tmp_path / "j.db", ALWAYS_LEAKS, NO_FAULTS, skip_durability_check=True)
    )
    assert unauthorised_effects(result), (
        "I2 did not notice an effect dispatched against a journal offset that does not exist"
    )


def test_i2_catches_the_real_shape_of_the_planted_bug(tmp_path: Path) -> None:
    """The bug task 1.6 actually names: drain first, append the confirming entry after.

    This is the harder half and the one that matters. Nothing is fabricated -- every entry is
    present in the finished journal and every offset is real. Only the *order* is wrong, which
    is precisely what draining before the fsync means.

    For a long time I2 could not see this. It checked that the branch had some confirming entry
    somewhere in the finished journal, and that the offset the dispatch named was not past the
    final head; an entry appended after the dispatch satisfies both. The proof above passed
    only because it fabricated ``head + 1000``, a strictly stronger falsification than the bug
    being modelled -- so the invariant looked proven while the fault class it was written for
    went straight through it.
    """
    result = asyncio.run(
        simulate(tmp_path / "real.db", ALWAYS_LEAKS, NO_FAULTS, drain_before_confirm=True)
    )

    assert result.world.mutations, "nothing was dispatched, so this proves nothing"
    problems = unauthorised_effects(result)
    assert problems, (
        "I2 did not notice an effect dispatched before its confirming entry was written -- "
        "which is the exact bug it exists to catch"
    )
    assert any("before anything authorised it" in problem for problem in problems), problems

    # And I1 is confirmed blind to it, which is why I2 exists at all: the branch does retire,
    # so the subset invariant holds perfectly while the effect left too early.
    assert not leaked_branches(result), (
        "this planted bug is supposed to be invisible to I1; if I1 sees it, the two invariants "
        "are not testing different things and one of them is redundant"
    )


def test_the_real_drain_refuses_both_planted_bugs(tmp_path: Path) -> None:
    """And the shipped code, with its preconditions intact, refuses both."""
    asyncio.run(_squashed_attempt(tmp_path))
    asyncio.run(_durability_attempt(tmp_path))


async def _squashed_attempt(tmp_path: Path) -> None:
    run_id = new_ulid()
    journal = Journal(tmp_path / "k.db")
    world = standard_world()
    registry = _registry(world)
    buffer = StoreBuffer(journal=journal, run_id=run_id)
    dispatcher = Dispatcher(registry=registry, max_attempts=1)
    doomed = Branch(id=new_ulid(), status=BranchStatus.SPECULATIVE)
    doomed.advance_step()
    call = ToolCall("charge_card", _args("charge_card", 1))
    await buffer.stage(doomed, call, registry.get("charge_card"))
    doomed.squash("mismatch")
    with pytest.raises(BranchClosed):
        await buffer.drain(doomed, dispatcher, confirmed_offset=0, authorised_by_offset=0)
    assert world.mutations == []


async def _durability_attempt(tmp_path: Path) -> None:
    """The shipped drain refuses an offset the journal does not have."""
    run_id = new_ulid()
    journal = Journal(tmp_path / "n.db")
    world = standard_world()
    registry = _registry(world)
    buffer = StoreBuffer(journal=journal, run_id=run_id)
    dispatcher = Dispatcher(registry=registry, max_attempts=1)
    branch = Branch(id=new_ulid(), status=BranchStatus.SPECULATIVE)
    branch.advance_step()
    await buffer.stage(
        branch, ToolCall("charge_card", _args("charge_card", 1)), registry.get("charge_card")
    )
    branch.confirm()
    branch.context_verified = True
    with pytest.raises(BranchClosed, match="durable"):
        await buffer.drain(branch, dispatcher, confirmed_offset=999, authorised_by_offset=999)
    assert world.mutations == []


_TIERS = ("t0", "t1")


def _workload_cases() -> list[object]:
    return [pytest.param(w, tier, id=f"{w.name}-{tier}") for w in WORKLOADS for tier in _TIERS]


async def _run_workload(
    tmp_path: Path, workload: Workload, world: World, *, tier: str, db: str
) -> tuple[object, Journal, str]:
    from specunode.core.model import JournaledModel
    from specunode.core.policy import Policy
    from specunode.core.scheduler import Scheduler
    from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex

    adapter, registry = workload.make(world)
    journal = Journal(tmp_path / db)
    predictor = None
    if tier == "t1":
        index = PatternIndex(order=2)
        index.train([workload.decisions()])
        predictor = PatternDrafter(index=index)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(workload.model(), journal, provider="scripted"),
        policy=Policy(speculation=True),
        predictor=predictor,
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, dict(workload.seed))
    return result, journal, run_id


# -- the same invariant, over the real runtime on the real workloads ----------------------------
#
# Everything above drives a purpose-built simulator, which is what lets hypothesis explore 500
# branch trees a real workload would take hours to reach. The cost is that it proves Rule 3 about
# the simulator. These two assert the same invariant about the actual scheduler running the actual
# sample apps, at both drafter tiers -- narrow coverage, but of the thing that ships.


@pytest.mark.parametrize(("workload", "tier"), _workload_cases())
async def test_the_real_runtime_leaks_nothing_on_a_real_workload(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    """Hard Rule 3, over a run of a shipped workload rather than a generated tree."""
    world = standard_world()
    result, journal, run_id = await _run_workload(
        tmp_path, workload, world, tier=tier, db=f"leak-{workload.name}-{tier}.db"
    )
    assert result.ok

    retired = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    touched = world.mutating_branches()
    assert touched, "the workload changed nothing, so the invariant would hold vacuously"
    assert touched <= retired, (
        f"Hard Rule 3 violated: {sorted(touched - retired)} touched the world without retiring"
    )
    # I2 as well, over the real scheduler and not only over the simulator above. Without this,
    # if ``Scheduler._retire`` ever stopped going through the guarded ``drain()`` entry point
    # and dispatched before appending its confirming entry, nothing anywhere would notice --
    # which is the exact bug I2 was strengthened to catch, checked only in a hand-written
    # simulator that calls ``drain`` itself.
    assert not unauthorised_effects(
        RunResult(world=world, journal=journal, run_id=run_id, branches=[])
    ), "an effect was dispatched before its branch's confirming entry was durable"
    # And the count is the workload's declared one, so a run that dispatched less than it
    # should cannot pass by leaking nothing.
    assert len(world.mutations) == workload.expect_effects


@pytest.mark.parametrize(("workload", "tier"), _workload_cases())
async def test_a_squashed_branch_on_a_real_workload_leaves_nothing_behind(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    """Every branch that was squashed must be absent from the world's mutation log.

    Stated separately from the subset above because the subset holds vacuously when nothing was
    squashed, and on these workloads that is usually what happens: two of them never speculate
    at all, and on ``ops_agent`` the index is trained on the run's own trace, so it is right.
    The non-vacuous case -- a prediction that is actually wrong -- is the test below, which
    forces one rather than hoping for it.
    """
    world = standard_world()
    result, journal, run_id = await _run_workload(
        tmp_path, workload, world, tier=tier, db=f"squash-{workload.name}-{tier}.db"
    )
    assert result.ok

    squashed = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "squashed"
    }
    assert not (world.mutating_branches() & squashed)


async def test_a_deliberately_wrong_prediction_on_a_real_workload_leaks_nothing(
    tmp_path: Path,
) -> None:
    """The non-vacuous leak case: a real workload, a real drafter, and a guess that is wrong.

    The index is trained on a trace that restarts a *different* job, so the prediction is
    well-formed, fillable and wrong. That is the shape that matters: a branch that got far
    enough to stage a write before the model contradicted it. A prediction that never forms
    tests the policy that refused it, not the store buffer.
    """
    from specunode.core.model import JournaledModel
    from specunode.core.policy import Policy
    from specunode.core.scheduler import Scheduler
    from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex

    workload = next(w for w in WORKLOADS if w.tier_1_can_predict)
    world = standard_world()
    adapter, registry = workload.make(world)

    # Same shape, wrong target: the drafter will predict restart_job with the job_id it finds
    # in the status read, which the seeded row makes etl-2 -- so we train it to expect the
    # restart at a point where the model instead asks for something else entirely.
    wrong = [
        ToolCall("get_pipeline_status", {"pipeline_id": "etl-2"}),
        ToolCall("restart_job", {"job_id": "etl-2"}),
    ]
    index = PatternIndex(order=2)
    index.train([wrong])

    journal = Journal(tmp_path / "wrong-prediction.db")
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(workload.model(), journal, provider="scripted"),
        policy=Policy(speculation=True),
        predictor=PatternDrafter(index=index),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, dict(workload.seed))
    assert result.ok

    resolutions = [entry.payload for entry in journal.read(run_id, kinds=["branch_resolved"])]
    squashed = {
        str(payload["branch_id"]) for payload in resolutions if payload.get("status") == "squashed"
    }
    assert squashed, "the mistrained index still guessed right; this proves nothing"

    # Hard Rule 3: nothing the squashed branch did reached the world.
    assert not (world.mutating_branches() & squashed)
    # And the run still did its whole job -- a squash must not cost the run an effect.
    assert len(world.mutations) == workload.expect_effects
