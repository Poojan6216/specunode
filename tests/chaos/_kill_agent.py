"""Subprocess helper: run the support agent, optionally dying at an exact point in the run.

``python _kill_agent.py <dir> <run_id> <kill> [resume] [changed-mind] [billing]``, where ``kill``
is one of

* ``-1`` -- run to the end, and print how many kill points the run has
* ``op:N`` -- die just before the run's N-th durable journal write: an entry appended, a
  dispatch claimed, a claim marked unsent, or an ack settled
* ``send:N`` -- die just before the world applies its N-th mutation: the claim is on disk and
  the request never arrived -- the lost request
* ``mutation:N`` -- die just after the world has durably applied its N-th mutation, before the
  runtime has recorded that it did -- the request went out and the reply was lost

``changed-mind`` scripts a model that decides differently -- charges 30 rather than 25 -- which
is what a real model asked the same question twice may do. A resumed process is given it to
show that a decision which may already have sent something is served from the journal, and one
that sent nothing is asked for again.

``billing`` runs one node that asks the model what to charge and charges it in the same step,
instead of the support agent, whose ``decide`` node asks and sends nothing and whose ``charge``
node sends and asks nothing -- a shape in which no resume is ever served a turn.

The process dies by ``os._exit``: no cleanup, no flush, no finally blocks, as if the machine
lost power, except that what the OS already holds survives. The world writes to a durable log,
so what the dead process sent is still visible to the process that resumes -- which is the whole
point: a resumed run that could not see the dead one's effects could not avoid re-sending them.

Kill points used to be delays in milliseconds, calibrated against a timed run. A delay is not a
point in the run: on a CI runner the same delay landed before the run had journaled anything on
one attempt and after it had finished on another, so the test failed on either side of a
window it could not see. Counting names each point at which the journal, or the effects in
the world, change on disk -- on every machine, every time.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench._kill_points import arm, watch
from examples.support_agent.agent import build, build_tools

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, ToolCall
from specunode.core.graph import RunSession
from specunode.core.model import (
    JournaledModel,
    Message,
    RequestEnvelope,
    TextBlock,
    decisions_of,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.integrations.plain import PlainAdapter, node, registry_of
from specunode.journal.journal import Journal
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World

CHARGE = ("charge_card", {"customer_id": "cus-1", "amount": 25.0})
CHANGED_MIND = ("charge_card", {"customer_id": "cus-1", "amount": 30.0})


def seeded_world(directory: Path) -> World:
    world = World(log_path=directory / "world.jsonl")
    if not world.tables["customers"]:
        for index in range(1, 4):
            world.seed("customers", f"cus-{index}", name=f"Customer {index}", balance=100.0)
    return world


def build_billing(world: World) -> tuple[PlainAdapter, object]:
    """One node that asks the model what to charge, charges it, and sends the receipt."""

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        envelope = RequestEnvelope(
            model="scripted",
            messages=(Message(role="user", content=(TextBlock(text="Bill cus-1."),)),),
            max_tokens=64,
        )
        decision = decisions_of(await session.model.complete(envelope))[0]
        assert isinstance(decision, ToolCall)
        ack = await session.call_tool(decision.name, dict(decision.args))
        charge_id = str(ack.get("charge_id")) if isinstance(ack, dict) else "none"
        await session.call_tool("send_receipt", {"customer_id": "cus-1", "charge_id": charge_id})
        session.state["billed"] = True
        return decision

    def route(state: dict[str, JsonValue]) -> str | None:
        return None if state.get("billed") else "bill"

    return PlainAdapter.of([bill], route), registry_of(build_tools(world))  # type: ignore[arg-type]


async def main() -> None:
    directory = Path(sys.argv[1])
    run_id = sys.argv[2]
    kill = sys.argv[3]
    resuming = "resume" in sys.argv[4:]
    charge = CHANGED_MIND if "changed-mind" in sys.argv[4:] else CHARGE

    points = arm(kill)
    world = seeded_world(directory)
    watch(world, points)
    adapter, registry = build_billing(world) if "billing" in sys.argv[4:] else build(world)
    journal = Journal(directory / "journal.db")
    model = ScriptedModel(turns=[tool_turn(charge, turn=0), tool_turn(charge, turn=1)])
    scheduler = Scheduler(
        graph=adapter,
        registry=registry,  # type: ignore[arg-type]
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=1.0),  # type: ignore[arg-type]
        target=JournaledModel(model, journal, provider="scripted"),
        policy=Policy(speculation=False),
    )
    result = (
        await scheduler.resume(run_id)
        if resuming
        else await scheduler.run(run_id, {"customer_id": "cus-1"})
    )
    world.close()
    print(f"done ok={result.ok} rows={len(result.ledger.rows)} {points.report()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
