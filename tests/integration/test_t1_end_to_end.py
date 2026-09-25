"""The real tier-1 drafter, driven through the real scheduler (spec task 3.2).

``test_speculation.py`` proves the runtime handles a prediction, using a stub that returns a
fixed answer. That leaves the obvious gap: nothing showed that the *actual* ``PatternDrafter``
ever produces a prediction the runtime confirms. A predictor that silently returns nothing
passes every equivalence and leak test there is, because a run that never speculates trivially
matches a run that never speculates.

So both directions are asserted here against the journal: a correct guess is confirmed and its
work adopted, and a wrong guess is squashed with nothing left in the world.

**Why this needs a multi-call turn.** The drafter is asked after each ``tool_use`` block, and
its history is the calls *within the current turn*. A turn that emits one call therefore offers
nothing to predict from -- which is not an artefact of these tests but the shape the offline
corpus measured at 1.0000 of tool calls, and the reason the headline speculable span is zero.
The three sample apps all have that shape, so this file builds the other one explicitly.

**Why it needs three calls and not two.** The drafter is asked immediately after a block is
parsed, at which point the read that block asked for has only just been *issued* -- its task is
created and the drafter is consulted before it has had a chance to run. So argument filling can
never draw on the result of the call it was asked about; the earliest result it can use is one
from a block at least two back. That is a structural property of early issue rather than a
tuning problem, and it halves the reach of PASTE's data-flow idea inside this runtime. It is
written down in ``docs/limitations.md`` and the test is shaped around it rather than against it.
"""

from __future__ import annotations

from pathlib import Path

from tests.integration.test_speculation import OneTurnGraph, registry_for

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolSpec
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

#: The chain the index learns: read a ticket, then look up the customer the ticket names. The
#: second call's argument exists only inside the first call's *result*, which is the case the
#: drafter's argument filling is for and the one a values-from-arguments index cannot serve.
TRACE = [
    ToolCall("get_ticket", {"ticket_id": "tkt-1"}),
    ToolCall("fetch_runbook", {"section": "restart"}),
    ToolCall("lookup_customer", {"customer_id": "cus-1"}),
]


def world_with_ticket(customer_id: str = "cus-1") -> World:
    world = standard_world()
    world.seed("tickets", "tkt-1", customer_id=customer_id, title="Card declined", status="open")
    return world


def build(tmp_path: Path, world: World, *, db: str) -> tuple[Scheduler, Journal, str]:
    registry = registry_for(world)
    registry.register(
        ToolSpec(name="get_ticket", effect=EffectClass.READ, fn=world.get_ticket, witness=True)
    )
    registry.register(
        ToolSpec(
            name="lookup_customer",
            effect=EffectClass.READ,
            fn=world.lookup_customer,
            witness=True,
        )
    )
    index = PatternIndex(order=2)
    index.train([TRACE])
    journal = Journal(tmp_path / db)
    # A delay between blocks so block 1's early-issued read has finished by the time the
    # drafter is asked after block 2. Without it the result the prediction needs may not exist
    # yet, and the test would be measuring a race rather than the predictor. 25 ms was enough
    # on a laptop and is not on a CI runner, whose journal fsyncs are slower; the margin is for
    # the machine, not the runtime.
    model = ScriptedModel(
        turns=[tool_turn(*[(c.name, dict(c.args)) for c in TRACE], turn=0)],
        block_delay_ms=250.0,
    )
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=True),
        predictor=PatternDrafter(index=index),
    )
    return scheduler, journal, new_ulid()


def predicted_forks(journal: Journal, run_id: str) -> list[dict[str, object]]:
    return [
        dict(entry.payload)
        for entry in journal.read(run_id, kinds=["branch_forked"])
        if entry.payload.get("predicted") is not None
    ]


async def test_the_real_drafter_predicts_and_the_gate_confirms(tmp_path: Path) -> None:
    """The whole tier-1 path: mine, rank, fill from a result, fork, confirm, adopt."""
    world = world_with_ticket()
    scheduler, journal, run_id = build(tmp_path, world, db="confirm.db")
    result = await scheduler.run(run_id, {})
    assert result.ok

    forks = predicted_forks(journal, run_id)
    assert forks, "the real PatternDrafter never produced a prediction"
    assert all(fork["tier"] == 1 for fork in forks)

    ids = {str(fork["branch_id"]) for fork in forks}
    resolved = {
        str(entry.payload["branch_id"]): entry.payload
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if str(entry.payload.get("branch_id")) in ids
    }
    statuses = sorted(str(payload.get("status")) for payload in resolved.values())

    # Two predictions, and both outcomes are expected. The drafter is asked again after the
    # last block, so it guesses a call the model was never going to make and that branch is
    # squashed when the turn ends. Asserting only on the confirmed one would leave the
    # ordinary end-of-turn squash untested here.
    assert statuses == ["confirmed", "squashed"], statuses

    # "confirmed", not "retired": only the canonical branch retires. A confirmed speculation
    # is absorbed into it, and the entry names which branch absorbed it so the durable record
    # closes the child's lifecycle rather than leaving it dangling.
    confirmed = [p for p in resolved.values() if p.get("status") == "confirmed"]
    assert confirmed[0].get("adopted_by")
    # The point of being right: the confirmed branch's read was reused, not repeated. Each
    # predicted branch made exactly one read -- the one it was forked to run.
    assert len(world.reads_from(ids)) == len(forks)


async def test_a_wrong_prediction_leaves_the_world_alone(tmp_path: Path) -> None:
    """The ticket names a different customer, so the filled argument is wrong.

    The prediction is still *made* -- that is what makes this a test of the predictor rather
    than of the policy that refuses to make one -- and then the model asks for ``cus-1`` and
    the branch is squashed.
    """
    world = world_with_ticket(customer_id="cus-9")
    scheduler, journal, run_id = build(tmp_path, world, db="squash.db")
    result = await scheduler.run(run_id, {})
    assert result.ok

    forks = predicted_forks(journal, run_id)
    assert forks, "the drafter offered nothing, so nothing was tested"
    assert forks[0]["predicted"] == {
        "kind": "tool_call",
        "name": "lookup_customer",
        "args": {"customer_id": "cus-9"},
    }

    squashed = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "squashed"
    }
    assert squashed & {str(fork["branch_id"]) for fork in forks}
    # Hard Rule 3, stated over this run: nothing a squashed branch did reached the world.
    assert not (world.mutating_branches() & squashed)


async def test_a_confirmed_prediction_of_a_write_reaches_the_world_exactly_once(
    tmp_path: Path,
) -> None:
    """A regression test for a deadlock, and for the case the whole project is named for.

    Speculating *past* a write and being right is the thing SpecuNode exists to do, and until
    this test existed it hung the process. The child branch staged the write, the canonical
    branch adopted the child's task and awaited its ack, and nothing ever drained the child's
    buffer -- so the node body waited on a future no code path could ever resolve.

    It survived every other test because no test had ever confirmed a prediction of a *write*.
    The speculation suite's stub predicts a read for its confirm case (reads stage nothing) and
    a write only for its squash cases (squashed buffers are discarded, not drained). The three
    sample apps cannot produce a tier-1 prediction at all. The gap was exactly one cell of that
    table, and it was the important one.

    The assertion is against the world rather than the ledger: a ledger rebuilt by the same
    code path that dispatched can agree with a dispatcher that did nothing.
    """
    world = standard_world()
    world.seed("tickets", "tkt-1", job_id="etl-1", status="open")
    registry = registry_for(world)
    registry.register(
        ToolSpec(name="get_ticket", effect=EffectClass.READ, fn=world.get_ticket, witness=True)
    )
    write_trace = [
        ToolCall("get_ticket", {"ticket_id": "tkt-1"}),
        ToolCall("fetch_runbook", {"section": "restart"}),
        ToolCall("restart_job", {"job_id": "etl-1"}),
    ]
    index = PatternIndex(order=2)
    index.train([write_trace])
    journal = Journal(tmp_path / "confirmed-write.db")
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(
            ScriptedModel(
                turns=[tool_turn(*[(c.name, dict(c.args)) for c in write_trace], turn=0)],
                # The drafter asked after the second block fills ``job_id`` from the first
                # read's result, so that read must be back by then. At 25 ms it was on a laptop
                # and was not on a CI runner, whose journal fsyncs are slower: no restart_job
                # was offered, and a later guess forked instead. The margin is for the
                # machine, not the runtime.
                block_delay_ms=250.0,
            ),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True),
        predictor=PatternDrafter(index=index),
    )
    run_id = new_ulid()
    result = await scheduler.run(run_id, {})
    assert result.ok

    forks = predicted_forks(journal, run_id)
    assert [fork["predicted"] for fork in forks] == [
        {"kind": "tool_call", "name": "restart_job", "args": {"job_id": "etl-1"}}
    ]

    restarts = world.mutations_by("restart_job")
    assert len(restarts) == 1, f"the confirmed write reached the world {len(restarts)} times"
    # It was adopted rather than left on the branch that staged it, and the journal says so.
    adoptions = list(journal.read(run_id, kinds=["effect_adopted"]))
    assert len(adoptions) == 1
    assert adoptions[0].payload["from_branch_id"] == forks[0]["branch_id"]
    assert adoptions[0].payload["count"] == 1
    # Hard Rule 3 over this run: the branch the effect dispatched under is one that retired.
    retired = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    assert world.mutating_branches() <= retired
