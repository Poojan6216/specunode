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
    """Hard Rule 10's read budget, exercised rather than restated.

    This used to set ``max_speculative_reads=0`` and assert the gate was closed -- which
    ``0 >= 0`` satisfies before a single read has happened. It would have passed identically
    with ``Budget.record_speculative_read`` deleted, and in effect it was: nothing in the
    runtime called it, so ``speculative_reads_used`` was 0 on every run, the READ_BUDGET hazard
    could never fire, and the limit attack 7.2's writeup calls "the budget that bounds the cost
    of a wrong guess" bounded nothing.

    Now the limit is 1, a real speculative read is made, and the assertions are that it was
    *counted*, that the gate then closed, and that the closure is in the durable record.
    """
    drafter = FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}))
    scheduler, world, journal, run_id = build(tmp_path, drafter, db="reads.db")
    scheduler.policy = Policy(speculation=True, max_speculative_reads=1)
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    assert scheduler.budget.speculative_reads_used >= 1, (
        "a speculative read was made and never counted against the budget"
    )
    assert not scheduler.budget.may_speculate()
    assert scheduler.budget.exhausted() == "max_speculative_reads"
    assert scheduler.budget.speculation_disabled, "the gate closed but disable() never ran"

    events = [e.payload for e in journal.read(run_id, kinds=["policy_event"])]
    closed = [e for e in events if e.get("event") == "speculation_disabled"]
    assert len(closed) == 1, f"expected exactly one closure event, got {len(closed)}"
    assert closed[0]["reason"] == "max_speculative_reads"
    # The world is untouched by budget accounting: the run still did its job.
    assert [m.tool for m in world.mutations] == ["restart_job"]


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


async def test_a_stalled_speculation_reaches_the_ledger(tmp_path: Path) -> None:
    """``Branch.stall`` had no caller, so ``BranchStatus.STALLED`` was unreachable.

    ``record_stall`` only appended to an in-memory list and bumped a counter; no branch ever
    reached STALLED and no ``branch_resolved{stalled}`` entry was ever written. The ledger's
    stall reader selects on exactly that status, so ``Ledger.stalls`` was empty and
    ``branches_stalled`` was 0 on every run ever produced -- including runs where hazards
    demonstrably fired. ``branch.py``'s own docstring says a branch ends "in exactly one of
    retired, squashed or stalled"; one of the three could not happen.

    An irreversible prediction with ``stage_irreversible`` off is a barrier, which is the
    cheapest real hazard to provoke.
    """
    from specunode.core.effects import EffectClass, ToolSpec
    from specunode.journal.ledger import build_ledger

    world = standard_world()
    registry = registry_for(world)
    registry.register(
        ToolSpec(name="send_email", effect=EffectClass.IRREVERSIBLE, fn=world.send_email)
    )
    journal = Journal(tmp_path / "stalled.db")
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(
                turns=[
                    tool_turn(
                        ("fetch_runbook", {"section": "restart"}),
                        ("restart_job", {"job_id": "etl-1"}),
                        turn=0,
                    )
                ],
                block_delay_ms=20.0,
            ),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True, stage_irreversible=False),
        predictor=FixedDrafter(ToolCall("send_email", {"to": "a", "subject": "s", "body": "b"})),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    ledger = build_ledger(journal, run_id)
    assert ledger.stalls, "a hazard fired and the ledger recorded no stall"
    assert scheduler.counters.branches_stalled == len(ledger.stalls)
    assert ledger.stalls[0].hazard is not None, "the stall reached the ledger without its hazard"
    # And the branch really is journaled as stalled, which is the status the reader selects on.
    stalled = [
        entry.payload
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "stalled"
    ]
    assert stalled, "no branch_resolved{stalled} entry was written"


async def test_under_pastes_rule_a_predicted_read_runs_and_a_predicted_write_is_refused(
    tmp_path: Path,
) -> None:
    """The benchmark's PASTE arm has to speculate on reads and refuse writes -- both halves.

    It used to be ``speculation=True`` with no predictor, which forks nothing at all: the
    sequential arm under another name. So "does staging writes beat read-only speculation?",
    the comparison this project's thesis rests on, was made against nothing. The arm now has
    the same predictor as ``B_specunode`` and ``speculate_writes=False``, and this asserts it
    does what its label says.
    """
    from specunode.core.hazards import Hazard

    read_guess = FixedDrafter(ToolCall("fetch_runbook", {"section": "restart"}))
    scheduler, world, journal, run_id = build(tmp_path, read_guess, db="paste-read.db")
    scheduler.policy = Policy(speculation=True, speculate_writes=False)
    assert (await scheduler.run(run_id, {})).ok
    assert scheduler.counters.branches_forked > 1, "a predicted read was not speculated"

    write_guess = FixedDrafter(ToolCall("restart_job", {"job_id": "etl-1"}))
    scheduler, world, journal, run_id = build(tmp_path, write_guess, db="paste-write.db")
    scheduler.policy = Policy(speculation=True, speculate_writes=False)
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error
    stalls = [
        e.payload
        for e in journal.read(run_id, kinds=["branch_resolved"])
        if e.payload.get("status") == "stalled"
    ]
    assert [s["hazard"] for s in stalls] == [Hazard.WRITE_ON_PATH.name], stalls
    # Refused before anything was staged, so the world saw only the model's own write.
    assert [m.tool for m in world.mutations] == ["restart_job"]
