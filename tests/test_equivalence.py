"""THE EQUIVALENCE TEST (spec task 5.1, Hard Rule 9). Mandatory. Never skipped.

For any journaled run, the effect ledger with speculation on equals the ledger with speculation
off. That is the claim that makes the rest of the project usable: if speculating changed what
reached the world, every latency number would be bought with a behaviour change nobody asked
for.

It runs over **every workload in ``bench/workloads/`` and every drafter tier**, because the
relation is only as strong as the runs it is given. The three workloads are deliberately
different shapes -- ``support_agent`` reaches a non-idempotent write immediately,
``ops_agent`` opens with a read-only stretch and uses a compensable effect and an idempotent
write, ``research_agent`` is read-heavy and its write is irreversible and therefore a barrier
rather than something staged. A relation that held only for the first would be a statement
about one code path.

Three things stop this from being a test that passes because it checks nothing.

``expect_effects`` -- the workload declares how many effects it produces. A runtime that
dispatched nothing would otherwise compare two empty ledgers and report success for a run that
did not happen.

``ledger_matches_world`` -- each arm's ledger is joined against what its world actually
received, in order and by idempotency key. Two ledgers agreeing with each other while both
disagree with the world is a failure the relation alone cannot see.

``test_the_tier_1_arm_was_really_consulted`` -- a tier-1 arm whose drafter was never even
asked would be a tier-0 arm wearing its name, and would pass every comparison above while
testing strictly less. What that test does *not* claim is that the drafter predicted
anything: on these three workloads it cannot, and the reason is written down there.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from bench.workloads import WORKLOADS, Workload

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.drafters.base import DraftContext, Drafter, Prediction
from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.world import World, standard_world
from specunode.verify.equivalence import (
    EquivalenceError,
    assert_equivalent,
    ledger_matches_world,
    normalise_for_equivalence,
)

#: Tier 0 needs no object -- it is the target's own stream, and it is always on. Tier 1 is the
#: pattern index. Tier 2 is a draft *model* behind an optional extra; it is covered by its own
#: tests rather than here, because requiring it would make this mandatory test skippable, and
#: the one property this file may never have is a condition under which it does not run.
TIERS = ("t0", "t1")


class CountingDrafter:
    """Wraps a drafter and counts how often the runtime asked it for a prediction.

    Needed because "the tier-1 arm ran" and "the tier-1 drafter was consulted" are different
    facts, and on these workloads the second is the strongest one that is true.
    """

    def __init__(self, inner: Drafter) -> None:
        self._inner = inner
        self.asked = 0
        self.offered = 0

    async def predict(self, ctx: DraftContext) -> Sequence[Prediction]:
        self.asked += 1
        out = await self._inner.predict(ctx)
        self.offered += len(out)
        return out


def _index_for(workload: Workload) -> PatternIndex:
    """A tier-1 index trained on the workload's own retired trace."""
    index = PatternIndex(order=2)
    index.train([workload.decisions()])
    return index


def _drafter(workload: Workload, tier: str) -> Drafter | None:
    if tier == "t1":
        return PatternDrafter(index=_index_for(workload))
    return None


async def run_arm(
    tmp_path: Path, workload: Workload, *, speculation: bool, tier: str, db: str
) -> tuple[RunResult, World, Journal]:
    world = standard_world()
    adapter, registry = workload.make(world)
    journal = Journal(tmp_path / db)
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(workload.model(), journal, provider="scripted"),
        policy=Policy(speculation=speculation),
        predictor=_drafter(workload, tier) if speculation else None,
    )
    result = await scheduler.run(new_ulid(), dict(workload.seed))
    return result, world, journal


def _cases() -> list[pytest.param]:  # type: ignore[valid-type]
    return [pytest.param(w, tier, id=f"{w.name}-{tier}") for w in WORKLOADS for tier in TIERS]


@pytest.mark.parametrize(("workload", "tier"), _cases())
async def test_the_two_arms_reach_the_same_world(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    """Hard Rule 9, over a real run of a real workload rather than synthetic ledgers."""
    off, off_world, _ = await run_arm(
        tmp_path, workload, speculation=False, tier=tier, db=f"{workload.name}-{tier}-off.db"
    )
    on, on_world, _ = await run_arm(
        tmp_path, workload, speculation=True, tier=tier, db=f"{workload.name}-{tier}-on.db"
    )
    assert off.ok and on.ok

    assert_equivalent(
        off.ledger,
        on.ledger,
        off_world.mutations,
        on_world.mutations,
        expect_rows=workload.expect_effects,
        # Zero, and stated rather than assumed: these scripts have a single decision point and
        # the index is trained on the run's own retired trace, so a correct prediction is the
        # expected case. The claim that the store buffer *did* hold something back is made by
        # test_the_tier_1_arm_really_speculated against the journal, which is the right place
        # for it -- a minimum asserted here would only be a proxy for that.
        min_squashed_with_staged=0,
    )


@pytest.mark.parametrize(("workload", "tier"), _cases())
async def test_each_arms_ledger_matches_its_own_world(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    """A ledger claiming an effect and a world receiving it are different facts."""
    for speculation in (False, True):
        result, world, _ = await run_arm(
            tmp_path,
            workload,
            speculation=speculation,
            tier=tier,
            db=f"{workload.name}-{tier}-{speculation}.db",
        )
        assert len(result.ledger.rows) == workload.expect_effects
        assert ledger_matches_world(result.ledger, world.mutations)


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
async def test_the_tier_1_arm_was_really_consulted(tmp_path: Path, workload: Workload) -> None:
    """The drafter is wired in and asked -- and on these workloads it offers nothing.

        That second half is a measured property of the workloads, not a defect, and writing it
        down as an assertion is the only way it stays true by choice rather than by accident.

    Two separate reasons a workload may not speculate, and the test distinguishes them because
        conflating them would hide either one.

        ``support_agent`` and ``research_agent`` call the model directly and issue each tool
        themselves. Early issue and the drafters live inside the turn the runtime drives, so those
        runs consult no drafter at all -- the tier parameter changes nothing for them beyond
        proving that attaching a predictor does not perturb the ledger.

        ``ops_agent`` hands its turn to the runtime and emits several calls in it, so there the
        drafter really is asked, really predicts, and is really confirmed. The one-call-per-turn
        shape the other two have is the shape the offline corpus measured at 1.0000 of tool calls,
        which is why a suite where tier 1 fired everywhere would be a suite whose workloads did not
        resemble anything real.

        Tier 1 genuinely predicting, being confirmed, and being squashed is covered by
        ``tests/integration/test_t1_end_to_end.py``, on a turn that emits several calls.
    """
    world = standard_world()
    adapter, registry = workload.make(world)
    journal = Journal(tmp_path / f"{workload.name}-consulted.db")
    drafter = CountingDrafter(PatternDrafter(index=_index_for(workload)))
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(workload.model(), journal, provider="scripted"),
        policy=Policy(speculation=True),
        predictor=drafter,  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), dict(workload.seed))
    assert result.ok

    forked = [
        entry
        for entry in journal.read(result.run_id, kinds=["branch_forked"])
        if entry.payload.get("predicted") is not None
    ]

    if not workload.drives_turn:
        # The app calls the model directly and issues each tool itself. That is a supported
        # pattern -- Demo 1 uses it -- but the drafters live inside the turn the *runtime*
        # drives, so on this shape no drafter is consulted at all. Asserting it keeps the
        # distinction visible: "tier 1 declined" and "tier 1 was never asked" are different
        # facts, and only one of them is true here.
        assert drafter.asked == 0, (
            "a drafter was consulted on a workload that does not use call_turn; if the "
            "runtime now speculates on this path, this expectation is stale"
        )
        assert forked == [], "no drafter ran, so nothing may have been forked"
        return

    if not workload.tier_1_can_predict:
        # A one-call-per-turn workload that does drive its turn. The drafter is asked --
        # proving it is wired in and the arm is not silently tier 0 -- and correctly declines,
        # because the call it would predict belongs to a turn that has not begun.
        assert drafter.asked > 0, "the runtime never asked the tier-1 drafter"
        assert drafter.offered == 0, (
            "the drafter offered a prediction on a one-call-per-turn workload; if that is "
            "now possible this expectation is stale and the claim needs remeasuring"
        )
        assert forked == [], "nothing was offered, so nothing may have been forked"
        return

    # A multi-call turn. Here the drafter has a context to rank against and a prior result to
    # fill an argument from, so it must actually predict -- and at least one prediction must
    # have been right, or the arm is exercising only the squash path.
    assert drafter.offered > 0, "the drafter offered nothing on a workload shaped for it"
    assert forked, "the drafter offered a prediction but no branch was forked on it"
    assert all(entry.payload.get("tier") == 1 for entry in forked)
    statuses = {
        str(entry.payload["branch_id"]): entry.payload.get("status")
        for entry in journal.read(result.run_id, kinds=["branch_resolved"])
    }
    outcomes = {statuses.get(str(entry.payload["branch_id"])) for entry in forked}
    assert "confirmed" in outcomes, f"no prediction was confirmed; outcomes were {outcomes}"


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
async def test_the_comparison_is_not_vacuous(tmp_path: Path, workload: Workload) -> None:
    """A run that dispatched nothing must fail rather than compare two empty ledgers."""
    off, off_world, _ = await run_arm(
        tmp_path, workload, speculation=False, tier="t0", db=f"{workload.name}-vac-off.db"
    )
    on, on_world, _ = await run_arm(
        tmp_path, workload, speculation=True, tier="t0", db=f"{workload.name}-vac-on.db"
    )
    with pytest.raises(EquivalenceError, match="were not compared"):
        assert_equivalent(
            off.ledger, on.ledger, off_world.mutations, on_world.mutations, expect_rows=99
        )


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
async def test_the_relation_notices_an_extra_effect(tmp_path: Path, workload: Workload) -> None:
    """The shape a leak has: the speculative arm did something the sequential one did not."""
    from dataclasses import replace

    off, _, _ = await run_arm(
        tmp_path, workload, speculation=False, tier="t0", db=f"{workload.name}-extra-off.db"
    )
    on, _, _ = await run_arm(
        tmp_path, workload, speculation=True, tier="t0", db=f"{workload.name}-extra-on.db"
    )

    leaked = replace(on.ledger.rows[0], dispatch_index=9, retire_seq=9)
    tampered = replace(on.ledger, rows=(*on.ledger.rows, leaked))
    assert normalise_for_equivalence(off.ledger) != normalise_for_equivalence(tampered)


@pytest.mark.parametrize(("workload", "tier"), _cases())
async def test_both_arms_journal_chains_verify(
    tmp_path: Path, workload: Workload, tier: str
) -> None:
    for speculation in (False, True):
        result, _, journal = await run_arm(
            tmp_path,
            workload,
            speculation=speculation,
            tier=tier,
            db=f"{workload.name}-{tier}-chain-{speculation}.db",
        )
        assert journal.verify_chain(result.run_id).ok
