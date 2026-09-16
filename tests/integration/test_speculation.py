"""Real branch speculation (spec tasks 3.3, 3.7 and 3.9).

Tier 0 hides latency without guessing. This is the part that guesses: while the model is still
streaming block *j*, the pattern index predicts block *j+1* and a branch is forked to run it.
If the model then emits what was predicted, that work is kept and the call is not made twice.
If it emits something else, the branch is squashed -- its task cancelled, its store buffer
discarded unsent.

The tests below are mostly about the mismatch case, because that is where a wrong guess either
costs nothing or costs a charge. A confirmed speculation saving time is easy; a squashed one
leaving no trace in the world is the claim.
"""

from __future__ import annotations

from pathlib import Path

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import JournaledModel, Message, RequestEnvelope, TextBlock
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.drafters.base import DraftContext, Prediction
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

TURN = (
    ("get_pipeline_status", {"pipeline_id": "etl-1"}),
    ("fetch_runbook", {"section": "restart"}),
    ("restart_job", {"job_id": "etl-1"}),
)


class FixedDrafter:
    """Predicts one fixed call, ``limit`` times, so a test controls exactly what is guessed.

    ``limit`` defaults to one lookahead, which is what a real drafter does: it is asked again
    after every block, and a drafter with nothing to say returns nothing. A fixture that
    answered every time would fork a branch after every block and make the counts hard to read
    without testing anything extra.
    """

    def __init__(self, decision: Decision, tier: int = 1, limit: int = 1) -> None:
        self._decision = decision
        self.tier = tier
        self.limit = limit
        self.calls = 0

    async def predict(self, ctx: DraftContext) -> list[Prediction]:
        self.calls += 1
        if self.calls > self.limit:
            return []
        return [Prediction(decision=self._decision, tier=1, score=0.9)]


def registry_for(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=world.get_pipeline_status,
            witness=True,
        )
    )
    registry.register(
        ToolSpec(name="fetch_runbook", effect=EffectClass.READ, fn=world.fetch_runbook)
    )
    registry.register(
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    registry.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))
    return registry


class OneTurnGraph:
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(drives_itself=False, framework="plain")

    def nodes(self) -> list[NodeRef]:
        return [NodeRef(name="agent")]

    def decision_kind(self, node: NodeRef) -> str:
        return "tool_call"

    def next(self, state: object) -> NextNode:
        return END if isinstance(state, dict) and state.get("done") else NodeRef(name="agent")

    async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
        assert session.call_turn is not None
        await session.call_turn(
            RequestEnvelope(
                model="scripted",
                messages=(Message(role="user", content=(TextBlock(text="go"),)),),
                max_tokens=128,
                stream=True,
            )
        )
        session.state["done"] = True
        return ToolCall("restart_job", {"job_id": "etl-1"})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


def build(
    tmp_path: Path, drafter: object | None, *, db: str = "journal.db"
) -> tuple[Scheduler, World, Journal, str]:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / db)
    model = ScriptedModel(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=15.0)
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=True, max_speculation_depth=3),
        predictor=drafter,  # type: ignore[arg-type]
    )
    return scheduler, world, journal, new_ulid()


async def test_a_correct_prediction_is_adopted_rather_than_run_twice(tmp_path: Path) -> None:
    """The latency win: the predicted read was already in flight when the model asked for it."""
    drafter = FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}))
    scheduler, world, _journal, run_id = build(tmp_path, drafter)
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    turn = scheduler._turns[0]
    assert turn.confirmed >= 1, "the drafter's correct guess was never confirmed"
    assert turn.adopted >= 1, "a confirmed speculation's work must be kept, not redone"
    runbook_reads = [r for r in world.reads if r.tool == "fetch_runbook"]
    assert len(runbook_reads) == 1, "the adopted call must not be made a second time"


async def test_a_wrong_prediction_leaves_nothing_in_the_world(tmp_path: Path) -> None:
    """The claim. A guessed write is staged, and a squashed branch's buffer is discarded unsent."""
    drafter = FixedDrafter(ToolCall("charge_card", {"customer_id": "cus-1", "amount": 99.0}))
    scheduler, world, journal, run_id = build(tmp_path, drafter)
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    turn = scheduler._turns[0]
    assert turn.squashed >= 1, "the drafter's wrong guess was never squashed"
    assert world.mutations_by("charge_card") == [], (
        "a charge the model never asked for reached the world"
    )

    retired = {
        str(e.payload["branch_id"])
        for e in journal.read(run_id, kinds=["branch_resolved"])
        if e.payload.get("status") == "retired"
    }
    touched = {m.branch_id for m in world.mutations} - {"<external>"}
    assert touched <= retired, f"{touched - retired} changed the world without retiring"


async def test_a_squashed_branchs_staged_effects_are_journaled_as_discarded(
    tmp_path: Path,
) -> None:
    """Discarded, and counted -- the ledger has to be able to say what was thrown away."""
    drafter = FixedDrafter(ToolCall("charge_card", {"customer_id": "cus-1", "amount": 99.0}))
    scheduler, _world, journal, run_id = build(tmp_path, drafter)
    await scheduler.run(run_id, {})

    discards = list(journal.read(run_id, kinds=["effect_discarded"]))
    assert discards, "a squashed branch with a staged write journaled no discard"
    assert sum(int(e.payload["count"]) for e in discards) >= 1
    squashes = [
        e
        for e in journal.read(run_id, kinds=["branch_resolved"])
        if e.payload.get("status") == "squashed"
    ]
    assert squashes and squashes[0].payload["reason"] == "mismatch"


async def test_a_speculative_read_that_was_squashed_still_counts_as_spent(
    tmp_path: Path,
) -> None:
    """Attack 7.2: a wrong guess costs whatever its reads cost, and the ledger says so."""
    drafter = FixedDrafter(ToolCall("get_pipeline_status", {"pipeline_id": "etl-99"}))
    scheduler, world, _journal, run_id = build(tmp_path, drafter)
    await scheduler.run(run_id, {})

    squashed_reads = [r for r in world.reads if r.args_hash and r.speculative]
    assert squashed_reads, "the speculation made no upstream read, so nothing was risked"
    assert scheduler.counters.speculative_reads_upstream > 0


async def test_speculation_is_off_when_the_policy_says_so(tmp_path: Path) -> None:
    """The same code path with no candidates: the sequential case is not a second runtime."""
    drafter = FixedDrafter(ToolCall("charge_card", {"customer_id": "cus-1", "amount": 99.0}))
    scheduler, world, _journal, run_id = build(tmp_path, drafter, db="off.db")
    scheduler.policy = Policy(speculation=False)
    result = await scheduler.run(run_id, {})

    assert result.ok
    turn = scheduler._turns[0]
    assert turn.squashed == 0 and turn.confirmed == 0
    assert drafter.calls == 0, "a disabled policy must not even ask the drafter"
    assert [m.tool for m in world.mutations] == ["restart_job"]


async def test_the_budget_stops_speculation_when_the_reads_run_out(tmp_path: Path) -> None:
    """Hard Rule 10: wasted work is bounded, and the bound has a name."""
    drafter = FixedDrafter(ToolCall("get_pipeline_status", {"pipeline_id": "etl-99"}))
    scheduler, _world, _journal, run_id = build(tmp_path, drafter, db="budget.db")
    scheduler.policy = Policy(speculation=True, max_speculative_reads=0)
    await scheduler.run(run_id, {})
    assert not scheduler.budget.may_speculate()
    assert scheduler.budget.exhausted() == "max_speculative_reads"


async def test_the_run_still_does_what_it_was_asked_whatever_the_drafter_guessed(
    tmp_path: Path,
) -> None:
    """Whether the guess was right or wrong, the world ends up the same."""
    right = FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}))
    wrong = FixedDrafter(ToolCall("charge_card", {"customer_id": "cus-1", "amount": 99.0}))

    outcomes = []
    for index, drafter in enumerate((right, wrong)):
        scheduler, world, _journal, run_id = build(tmp_path, drafter, db=f"same-{index}.db")
        await scheduler.run(run_id, {})
        outcomes.append([(m.tool, m.args_hash) for m in world.mutations])
    assert outcomes[0] == outcomes[1]
