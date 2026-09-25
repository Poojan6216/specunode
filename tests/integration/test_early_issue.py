"""Tier-0 early issue (spec task 3.1).

Task 3.1's Verify, stated exactly: with a scripted model streaming three tool calls over
300 ms, the READ among them completes before the stream ends, asserted via timestamps; the
WRITEs stage and retire only after the stream's end-of-turn entry is journaled.

That pair is the whole claim of tier 0. The saving is real -- a read the model emitted first
runs while it is still writing the rest of its answer -- and it costs nothing in safety,
because the writes in the same turn are staged and cannot reach the world until the turn is on
disk and the branch has retired.
"""

from __future__ import annotations

from pathlib import Path

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.core.decision import Decision, ToolCall
from specunode.core.effects import (
    EffectClass,
    ToolRegistry,
    ToolSpec,
    forward_keys_from_template,
)
from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    RequestEnvelope,
    TextBlock,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World, standard_world

#: A read first, then two writes -- the shape where early issue pays.
TURN = (
    ("get_pipeline_status", {"pipeline_id": "etl-1"}),
    ("restart_job", {"job_id": "etl-1"}),
    ("post_summary", {"channel": "#ops", "text": "restarted"}),
)
#: 100 ms per block, so a 3-block turn streams over ~300 ms as the spec describes.
BLOCK_DELAY_MS = 100.0


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
        ToolSpec(
            name="restart_job", effect=EffectClass.WRITE, fn=world.restart_job, idempotent=True
        )
    )
    registry.register(
        ToolSpec(name="post_summary", effect=EffectClass.WRITE, fn=world.post_summary)
    )
    return registry


class OneTurnGraph:
    """A single node that runs one model turn with early issue and stops."""

    def __init__(self) -> None:
        self.results: list[object] = []

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
        envelope = RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text="restart etl-1"),)),),
            max_tokens=256,
            stream=True,
        )
        self.results = list(await session.call_turn(envelope))
        session.state["done"] = True
        return ToolCall("post_summary", {"channel": "#ops", "text": "restarted"})

    async def drive(self, session: RunSession, inputs: object) -> object:
        raise NotImplementedError


def build(
    tmp_path: Path, *, policy: Policy | None = None
) -> tuple[Scheduler, World, Journal, str, OneTurnGraph]:
    world = standard_world()
    registry = registry_for(world)
    journal = Journal(tmp_path / "journal.db")
    graph = OneTurnGraph()
    model = ScriptedModel(turns=[tool_turn(*TURN, turn=0)], block_delay_ms=BLOCK_DELAY_MS)
    run_id = new_ulid()
    scheduler = Scheduler(
        graph=graph,  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=policy or Policy(speculation=True),
    )
    return scheduler, world, journal, run_id, graph


async def test_the_read_completes_before_the_stream_ends(tmp_path: Path) -> None:
    """The whole point of tier 0, asserted on the clock rather than on the design."""
    scheduler, _world, _journal, run_id, _graph = build(tmp_path)
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    turn = scheduler._turns[0]
    assert turn.reads_issued_early == 1
    assert turn.stream_ended_at > 0.0
    read_finished = turn.completed_at[0]
    assert read_finished > 0.0, "the read never completed"
    assert read_finished < turn.stream_ended_at, (
        f"the read finished at {read_finished:.4f} but the stream only ended at "
        f"{turn.stream_ended_at:.4f}: it was not issued early"
    )


async def test_the_read_really_ran_and_returned_a_value(tmp_path: Path) -> None:
    """A read that was never issued would also trivially finish 'before' the stream ended."""
    scheduler, world, _journal, run_id, graph = build(tmp_path)
    await scheduler.run(run_id, {})
    # Two, not one: the model asked for this read, and lattice rule E3 re-fetches it at
    # retirement to check the witness has not moved. That second call is a real upstream call
    # and the design counts it rather than hiding it -- speculation is not free, and a read
    # that is validated costs two.
    assert [r.tool for r in world.reads] == ["get_pipeline_status", "get_pipeline_status"]
    assert isinstance(graph.results[0], dict) and "witness" in graph.results[0]


async def test_the_writes_reach_the_world_only_after_the_turn_is_durable(
    tmp_path: Path,
) -> None:
    """Hard Rule 3 through the early-issue path: streaming changes when, never whether."""
    scheduler, world, journal, run_id, _graph = build(tmp_path)
    await scheduler.run(run_id, {})

    kinds = [e.kind for e in journal.read(run_id)]
    response_at = kinds.index("model_response")
    for kind in ("effect_staged", "effect_dispatched"):
        assert kinds.index(kind) > response_at, (
            f"{kind} was journaled before the turn that authorised it was durable"
        )
    assert [m.tool for m in world.mutations] == ["restart_job", "post_summary"]


async def test_the_read_is_recorded_as_speculative_and_the_writes_are_not(
    tmp_path: Path,
) -> None:
    """The read went out before the turn was durable, so it is counted as a speculation."""
    scheduler, world, _journal, run_id, _graph = build(tmp_path)
    await scheduler.run(run_id, {})
    # The model-emitted read was issued before its turn was durable, so it is speculative.
    # E3's retirement-time re-fetch is not: by then the turn is journaled and the branch is
    # confirmed, and marking the validation probe speculative would inflate the very count
    # attack 7.2 exists to report honestly.
    assert [r.speculative for r in world.reads] == [True, False]
    assert all(not m.speculative for m in world.mutations)


async def test_results_come_back_in_program_order(tmp_path: Path) -> None:
    """The model asked for these in this order and must be shown them in it (Hard Rule 13)."""
    scheduler, _world, _journal, run_id, graph = build(tmp_path)
    await scheduler.run(run_id, {})
    assert len(graph.results) == 3
    # The read's result first, then the two staged writes' acks, matching the emitted order.
    assert isinstance(graph.results[0], dict) and "witness" in graph.results[0]
    for ack in graph.results[1:]:
        assert isinstance(ack, dict) and ack.get("ok") is True


async def test_every_emitted_call_is_journaled_once(tmp_path: Path) -> None:
    scheduler, _world, journal, run_id, _graph = build(tmp_path)
    await scheduler.run(run_id, {})
    requests = [e.payload["tool"] for e in journal.read(run_id, kinds=["tool_request"])]
    assert requests == ["get_pipeline_status", "restart_job", "post_summary"]


async def test_a_read_still_in_flight_at_retirement_does_not_demote_the_branch(
    tmp_path: Path,
) -> None:
    """A regression test for a race that early issue reaches by design, not by accident.

    ``_timed_read`` used to mark its read speculative by saving the branch's ``status``,
    setting it to SPECULATIVE, and restoring the saved value in a ``finally``. A read still
    running when the scheduler confirmed the branch therefore put SPECULATIVE back afterwards,
    silently undoing the confirmation; the next drain refused, and the run died with a Hard
    Rule 3 message about a branch that had in fact been confirmed correctly.

    Overlapping a read with the drain is the whole point of early issue, so this was reachable
    on the ordinary path rather than under a fault. The same shape could also have promoted: a
    read that began on a CONFIRMED branch which was then squashed would have restored CONFIRMED
    over the squash, and a squashed branch that can drain is Rule 3 itself.

    **The read has to be the turn's last block.** A read emitted first finishes long before
    retirement even when it is slow, and the race never opens -- which is exactly why the first
    version of this test passed against the unfixed code and had to be rewritten.
    """
    world = standard_world()
    world.slow("read", 300)
    # Declared, so the runtime can see the read does not touch what the write does. It used to
    # read the pipeline status -- the very row the restart writes -- which an early read got
    # wrong, and which now runs after the write instead of racing it.
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
            name="fetch_runbook",
            effect=EffectClass.READ,
            fn=world.fetch_runbook,
            forward_keys=forward_keys_from_template("runbook:{args.section}"),
        )
    )
    journal = Journal(tmp_path / "in-flight-read.db")
    model = ScriptedModel(
        # Write first, independent read second: the read is issued as the stream's last block
        # and is still running while the write is staged, confirmed and drained.
        turns=[
            tool_turn(
                ("restart_job", {"job_id": "etl-1"}),
                ("fetch_runbook", {"section": "restart"}),
                turn=0,
            )
        ],
        block_delay_ms=20.0,
    )
    scheduler = Scheduler(
        graph=OneTurnGraph(),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=True),
    )
    run_id = new_ulid()

    result = await scheduler.run(run_id, {})

    assert result.ok, result.error
    assert len(result.ledger.rows) == 1, "the write must still have reached the world"
    # And the read is still reported as speculative: it did reach upstream before any durable
    # decision authorised it, which is the fact attack 7.2 counts.
    assert [hit for hit in world.reads if hit.speculative], (
        "the early-issued read stopped being reported as speculative"
    )


async def test_a_read_after_a_write_in_the_same_turn_is_seen_as_a_hazard(
    tmp_path: Path,
) -> None:
    """The one shape ``READ_AFTER_STAGED_WRITE`` exists for, and it could not fire.

    A turn stages its writes only after the stream ends -- deliberately, so a staged effect
    never precedes a durable turn -- while reads are issued as their blocks parse. So when
    hazard analysis ran for an early-issued read, the same turn's earlier write was not in the
    store buffer yet and ``staged_keys`` was empty. A read that follows a write *inside a single
    model turn* therefore returned a pre-write value with no stall, no forwarding and no note in
    the ledger, which is precisely the read-after-write the hazard is named for.

    Both tools declare ``forward_keys`` over the same resource, so the conflict is declared
    rather than inferred -- an under-declared ``forward_keys`` is its own documented limitation.
    """
    from specunode.core.effects import ToolSpec, forward_keys_from_template

    world = standard_world()
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
            name="get_pipeline_status",
            effect=EffectClass.READ,
            fn=world.get_pipeline_status,
            witness=True,
            forward_keys=forward_keys_from_template("job:{args.pipeline_id}"),
        )
    )
    journal = Journal(tmp_path / "raw.db")
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
                        ("restart_job", {"job_id": "etl-1"}),
                        ("get_pipeline_status", {"pipeline_id": "etl-1"}),
                        turn=0,
                    )
                ],
                block_delay_ms=20.0,
            ),
            journal,
            provider="scripted",
        ),
        policy=Policy(speculation=True),
    )
    result = await scheduler.run(new_ulid(), {})

    assert result.ok, result.error
    assert "READ_AFTER_STAGED_WRITE" in scheduler.counters.stalls_by_hazard, (
        "a read touching the same key as a write emitted earlier in the same turn was not "
        f"reported as a hazard; saw {dict(scheduler.counters.stalls_by_hazard)}"
    )


async def test_early_issue_can_be_switched_off_and_then_nothing_is_issued_early(
    tmp_path: Path,
) -> None:
    """Tier 0 needed a switch before any benchmark could say what it was worth.

    It is not gated by ``speculation``, and it never was: it forks no branch and guesses
    nothing. So every arm of the latency benchmark ran with it on -- including the arm named
    ``B_seq`` -- the control group contained the treatment, and no measurement could see the
    mechanism that does most of the work on a workload where nothing is predicted. Off is what
    "wait for the turn, then call the tools in order" actually means.
    """
    scheduler, _, journal, run_id, _ = build(
        tmp_path, policy=Policy(speculation=False, early_issue=False)
    )
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error
    assert scheduler._turns, "no turn ran, so this proves nothing"
    assert all(turn.reads_issued_early == 0 for turn in scheduler._turns)
    assert scheduler.counters.speculative_reads_upstream == 0, (
        "a read reached upstream ahead of its turn with early issue switched off"
    )
    # The durable record says which it was, so a ledger cannot be read as the other one.
    started = next(iter(journal.read(run_id, kinds=["run_started"])))
    assert started.payload["policy"]["early_issue"] is False

    on, _, _, on_run, _ = build(tmp_path, policy=Policy(speculation=False))
    assert (await on.run(on_run, {})).ok
    assert sum(turn.reads_issued_early for turn in on._turns) > 0, "the default stopped issuing"


async def test_staleness_is_rechecked_for_every_read_not_only_the_unauthorised_ones(
    tmp_path: Path,
) -> None:
    """Two questions that once shared one answer, and narrowing the wrong one loses a guarantee.

    "Did this read reach upstream with no durable decision behind it?" is what attack 7.2
    counts and the read budget charges. "Could this read have gone stale before the effects it
    informed reach the world?" is what lattice rule E3 re-checks at retirement -- and staleness
    is about elapsed time, not about authorisation. A read made on the ordinary path, after its
    turn was journaled, is authorised and can still be stale.

    Fixing the first question by asking ``predicted is not None`` silently answered the second
    one too, and E3 stopped re-checking ordinary reads.
    """
    scheduler, _, journal, run_id, _ = build(
        tmp_path, policy=Policy(speculation=False, early_issue=False)
    )
    result = await scheduler.run(run_id, {})
    assert result.ok, result.error

    # Nothing was issued ahead of its turn, so nothing is reported as unauthorised ...
    assert scheduler.counters.speculative_reads_upstream == 0
    # ... and the witnessed read was still re-checked before the writes went out.
    assert scheduler.counters.reads_validated > 0, (
        "rule E3 skipped a read because it was authorised, which is not what staleness means"
    )
    validated = [e.payload for e in journal.read(run_id, kinds=["read_validated"])]
    assert validated, "no read_validated entry was journaled"
    assert validated[-1]["total"] > 0


async def test_a_read_after_a_write_in_the_same_reply_sees_the_write(tmp_path: Path) -> None:
    """The reply asks to charge a customer and then to look them up. Issued early, as blocks
    parse, the lookup ran before the charge -- which is staged when the turn ends -- and the
    node was handed the balance from before it. The hazard check noticed, counted a stall, and
    let the read go ahead anyway. A read of something a write earlier in the turn changes now
    runs in program order, after the write is sent."""
    from examples.support_agent.agent import build_tools
    from tests.chaos._kill_agent import seeded_world

    from specunode.integrations.plain import PlainAdapter, node, registry_of

    seen: dict[str, object] = {}

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        ask = RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text="charge, then show"),)),),
            stream=True,
        )
        results = await session.call_turn(ask)  # type: ignore[misc]
        seen["balance"] = results[1]["value"]["balance"]
        session.state["billed"] = True
        return ToolCall("lookup_customer", {"customer_id": "cus-1"})

    world = seeded_world(tmp_path)
    registry = registry_of(build_tools(world))
    journal = Journal(tmp_path / "journal.db")
    scheduler = Scheduler(
        graph=PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill"),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(
            ScriptedModel(
                turns=[
                    tool_turn(
                        ("charge_card", {"customer_id": "cus-1", "amount": 25.0}),
                        ("lookup_customer", {"customer_id": "cus-1"}),
                    )
                ],
                block_delay_ms=5,
            ),
            journal,
        ),
        policy=Policy(speculation=False, early_issue=True),
    )
    result = await scheduler.run(new_ulid(), {})
    world.close()
    assert result.ok, result.error
    assert seen["balance"] == 75.0, "the read ran before the write the model asked for first"


async def test_a_read_between_two_writes_sees_the_first_and_not_the_guessed_second(
    tmp_path: Path,
) -> None:
    """The reply charges 25, looks the customer up, and charges 10 -- the charge a guess had
    already staged. The guess joined the branch when the turn ended, so the drain that sent the
    first charge sent the second with it, and the lookup, deferred until after the first, ran
    after both: the node was handed 65, a balance the customer never had at that point in the
    reply. A confirmed guess now joins the branch only when the reply's order reaches it. Found
    by the eleventh review."""
    import asyncio
    import time
    from collections.abc import AsyncIterator

    from examples.support_agent.agent import build_tools
    from tests.chaos._kill_agent import seeded_world

    from specunode.core.model import StreamEvent, ToolUseComplete, TurnComplete
    from specunode.drafters.base import Prediction
    from specunode.integrations.plain import PlainAdapter, node, registry_of

    second = ("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    seen: dict[str, object] = {}
    scheduler: Scheduler | None = None

    class GuessesTheSecondCharge:
        async def predict(self, context: object) -> list[Prediction]:
            history = getattr(context, "history", ())
            if history and history[-1].name == "lookup_customer":
                return [Prediction(decision=ToolCall(*second), tier=1, score=0.9)]
            return []

    class Model:
        async def complete(self, envelope: RequestEnvelope) -> object:
            raise NotImplementedError

        async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
            reply = tool_turn(
                ("charge_card", {"customer_id": "cus-1", "amount": 25.0}),
                ("lookup_customer", {"customer_id": "cus-1"}),
                second,
            )
            yield ToolUseComplete(index=0, block=reply.content[0])  # type: ignore[arg-type]
            yield ToolUseComplete(index=1, block=reply.content[1])  # type: ignore[arg-type]
            # The guess stages its charge before the model confirms it.
            deadline = time.monotonic() + 10.0
            while (  # noqa: ASYNC110
                scheduler is None or scheduler.counters.effects_staged < 1
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            yield ToolUseComplete(index=2, block=reply.content[2])  # type: ignore[arg-type]
            yield TurnComplete(response=reply)

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        ask = RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text="charge, show, charge"),)),),
            stream=True,
        )
        results = await session.call_turn(ask)  # type: ignore[misc]
        seen["balance"] = results[1]["value"]["balance"]
        session.state["billed"] = True
        return ToolCall(*second)

    world = seeded_world(tmp_path)
    registry = registry_of(build_tools(world))
    journal = Journal(tmp_path / "journal.db")
    scheduler = Scheduler(
        graph=PlainAdapter.of([bill], lambda s: None if s.get("billed") else "bill"),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=1.0),
        target=JournaledModel(Model(), journal),  # type: ignore[arg-type]
        policy=Policy(speculation=True, early_issue=True),
        predictor=GuessesTheSecondCharge(),  # type: ignore[arg-type]
    )
    result = await scheduler.run(new_ulid(), {})
    balance = world.tables["customers"]["cus-1"]["balance"]
    world.close()
    assert result.ok, result.error
    assert sum(turn.adopted for turn in scheduler._turns) == 1, "the guess was not confirmed"
    assert [m.tool for m in world.mutations] == ["charge_card", "charge_card"]
    assert balance == 65.0
    assert seen["balance"] == 75.0, "the read ran after a write the model asked for after it"
