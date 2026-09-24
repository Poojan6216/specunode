"""Pull the plug: what a crash sends twice.

The same billing run, five ways. Look up three customers, charge each one, send each a receipt,
then post one summary: seven effects on the world, none of which may happen twice. The process
is killed at each of the seven -- once just before the request reaches the upstream, and once
just after the upstream took effect but before the reply came back -- and restarted the way
each system restarts. Fourteen crashes each, and one run with no crash as a control.

- ``plain_loop`` -- an ordinary async loop. Nothing is persisted, so a restart starts over.
- ``langgraph_nodes`` -- LangGraph with a checkpointer, one node per customer. A restart
  resumes from the last completed node and re-runs the one that was interrupted.
- ``langgraph_tasks`` -- LangGraph's recommended pattern for side effects: each call is a
  ``@task`` of a functional ``@entrypoint``, checkpointed with ``durability="sync"``, so a
  restart replays completed tasks' results instead of re-running them.
- ``specunode`` -- the runtime in this repository. Every effect is claimed in the journal under
  a deterministic key before it is sent; a restart re-derives the same keys.
- ``specunode_reconcile`` -- the same, with each tool telling the runtime how to ask the
  upstream whether a request under a given key took effect.

A crash leaves one of four outcomes: the run finished and every effect happened once
(**exact**); the run stopped for a human with nothing sent twice (**held**); something was sent
twice (**duplicated**); or the run finished with an effect missing (**lost**).

No model and no network: every system runs the same deterministic business logic against the
same in-memory upstream, and the only thing that differs is what each one does about a crash.

``python bench/offline/run_crash_safety.py --out bench/results/crash_safety.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText
from specunode.core.effects import EffectClass
from specunode.core.graph import RunSession
from specunode.core.model import JournaledModel
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal, close_all_writers
from specunode.testing.models import ScriptedModel
from specunode.testing.world import World, standard_world

CUSTOMERS = ("cus-1", "cus-2", "cus-3")
AMOUNT = 25.0
#: A charge and a receipt per customer, then one summary.
WRITES = 2 * len(CUSTOMERS) + 1
WHENS = ("request_lost", "reply_lost")
SYSTEMS = (
    "plain_loop",
    "langgraph_nodes",
    "langgraph_tasks",
    "specunode",
    "specunode_reconcile",
)


class Crash(BaseException):
    """The process dying. A BaseException, so nothing on the way out mistakes it for an error."""


@dataclass
class Plug:
    """Pulls the plug once, at one write: before it reaches the upstream, or after."""

    at: int | None
    when: str
    writes: int = 0
    pulled: bool = False

    async def write(self, send: Callable[[], Awaitable[JsonValue]]) -> JsonValue:
        self.writes += 1
        fire = not self.pulled and self.writes == self.at
        if fire and self.when == "request_lost":
            self.pulled = True
            raise Crash(f"died before write {self.at} reached the upstream")
        result = await send()
        if fire and self.when == "reply_lost":
            self.pulled = True
            raise Crash(f"died after write {self.at} took effect, before its reply came back")
        return result


class Billing:
    """The business logic every system runs: the same calls, with the same data dependencies."""

    def __init__(self, world: World, plug: Plug) -> None:
        self.world = world
        self.plug = plug

    async def lookup(self, customer_id: str) -> JsonValue:
        return await self.world.lookup_customer(customer_id=customer_id)

    async def charge(self, customer_id: str) -> Mapping[str, Any]:
        result = await self.plug.write(
            lambda: self.world.charge_card(customer_id=customer_id, amount=AMOUNT)
        )
        assert isinstance(result, Mapping)
        return result

    async def receipt(self, customer_id: str, charge_id: str) -> JsonValue:
        return await self.plug.write(
            lambda: self.world.send_receipt(customer_id=customer_id, charge_id=charge_id)
        )

    async def summary(self) -> JsonValue:
        return await self.plug.write(
            lambda: self.world.post_summary(
                channel="#billing", text=f"charged {len(CUSTOMERS)} customers"
            )
        )


# -- the systems: each is (first run, restart), both returning whether the run finished --------


class System:
    #: Why a restart stopped short, when the system says.
    why: str | None = None

    async def first(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    async def restart(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class PlainLoop(System):
    def __init__(self, world: World, plug: Plug, _root: Path) -> None:
        self.billing = Billing(world, plug)

    async def _run(self) -> bool:
        for customer in CUSTOMERS:
            await self.billing.lookup(customer)
            charge = await self.billing.charge(customer)
            await self.billing.receipt(customer, str(charge["charge_id"]))
        await self.billing.summary()
        return True

    async def first(self) -> bool:
        return await self._run()

    async def restart(self) -> bool:
        # Nothing was persisted, so there is nothing to resume from.
        return await self._run()


class LangGraphNodes(System):
    def __init__(self, world: World, plug: Plug, _root: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        self.world, self.plug = world, plug
        self.saver = InMemorySaver()  # outlives the "process", as a durable checkpointer does
        self.config: Any = {"configurable": {"thread_id": "billing"}}

    def _app(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        class State(TypedDict):
            billed: Annotated[list[str], operator.add]

        billing = Billing(self.world, self.plug)

        def bill(customer: str) -> Callable[[State], Awaitable[dict[str, list[str]]]]:
            async def run(state: State) -> dict[str, list[str]]:
                await billing.lookup(customer)
                charge = await billing.charge(customer)
                await billing.receipt(customer, str(charge["charge_id"]))
                return {"billed": [customer]}

            return run

        async def summary(state: State) -> dict[str, list[str]]:
            await billing.summary()
            return {"billed": []}

        builder = StateGraph(State)
        previous = START
        for customer in CUSTOMERS:
            builder.add_node(f"bill_{customer}", bill(customer))  # type: ignore[call-overload]
            builder.add_edge(previous, f"bill_{customer}")
            previous = f"bill_{customer}"
        builder.add_node("summary", summary)  # type: ignore[call-overload]
        builder.add_edge(previous, "summary")
        builder.add_edge("summary", END)
        return builder.compile(checkpointer=self.saver)

    async def first(self) -> bool:
        await self._app().ainvoke({"billed": []}, self.config, durability="sync")
        return True

    async def restart(self) -> bool:
        await self._app().ainvoke(None, self.config, durability="sync")
        return True


class LangGraphTasks(System):
    def __init__(self, world: World, plug: Plug, _root: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        self.world, self.plug = world, plug
        self.saver = InMemorySaver()
        self.config: Any = {"configurable": {"thread_id": "billing"}}

    def _app(self) -> Any:
        from langgraph.func import entrypoint, task

        billing = Billing(self.world, self.plug)

        @task
        async def lookup(customer: str) -> JsonValue:
            return await billing.lookup(customer)

        @task
        async def charge(customer: str) -> Mapping[str, Any]:
            return await billing.charge(customer)

        @task
        async def receipt(customer: str, charge_id: str) -> JsonValue:
            return await billing.receipt(customer, charge_id)

        @task
        async def summary() -> JsonValue:
            return await billing.summary()

        @entrypoint(checkpointer=self.saver)
        async def workflow(customers: list[str]) -> list[str]:
            for customer in customers:
                await lookup(customer)
                charged = await charge(customer)
                await receipt(customer, str(charged["charge_id"]))
            await summary()
            return list(customers)

        return workflow

    async def first(self) -> bool:
        await self._app().ainvoke(list(CUSTOMERS), self.config, durability="sync")
        return True

    async def restart(self) -> bool:
        await self._app().ainvoke(None, self.config, durability="sync")
        return True


class SpecuNode(System):
    reconcile = False

    def __init__(self, world: World, plug: Plug, root: Path) -> None:
        self.world, self.plug = world, plug
        self.db = root / "journal.db"
        self.run_id = new_ulid()

    def _asker(self, tool_name: str, reply: Callable[[str], JsonValue]) -> Any:
        """How a tool answers "did the request under this key take effect?"."""
        world = self.world

        async def reconcile(key: str, args: Mapping[str, JsonValue]) -> JsonValue:
            record = world.record_of(tool_name, key)
            return None if record is None else reply(record.row_id)

        return reconcile if self.reconcile else None

    def _scheduler(self) -> Scheduler:
        billing = Billing(self.world, self.plug)

        @tool(effect=EffectClass.READ, witness=True)
        async def lookup_customer(customer_id: str) -> JsonValue:
            return await billing.lookup(customer_id)

        @tool(
            effect=EffectClass.WRITE,
            reconcile=self._asker("charge_card", lambda row: {"ok": True, "charge_id": row}),
        )
        async def charge_card(customer_id: str) -> JsonValue:
            return dict(await billing.charge(customer_id))

        @tool(
            effect=EffectClass.WRITE,
            reconcile=self._asker("send_receipt", lambda row: {"ok": True, "id": row}),
        )
        async def send_receipt(customer_id: str, charge_id: str) -> JsonValue:
            return await billing.receipt(customer_id, charge_id)

        @tool(
            effect=EffectClass.WRITE,
            reconcile=self._asker("post_summary", lambda row: {"ok": True, "message_id": row}),
        )
        async def post_summary() -> JsonValue:
            return await billing.summary()

        def bill(customer: str) -> object:
            @node(name=f"bill_{customer}")
            async def run(session: RunSession) -> Decision:
                await session.call_tool("lookup_customer", {"customer_id": customer})
                charged = await session.call_tool("charge_card", {"customer_id": customer})
                assert isinstance(charged, Mapping)
                await session.call_tool(
                    "send_receipt",
                    {"customer_id": customer, "charge_id": str(charged["charge_id"])},
                )
                session.state[f"billed:{customer}"] = True
                return FreeText.of(customer)

            return run

        @node(name="summary")
        async def summary(session: RunSession) -> Decision:
            await session.call_tool("post_summary", {})
            session.state["summarised"] = True
            return FreeText.of("summary")

        def route(state: Mapping[str, JsonValue]) -> str | None:
            pending = [c for c in CUSTOMERS if f"billed:{c}" not in state]
            if pending:
                return f"bill_{pending[0]}"
            return None if state.get("summarised") else "summary"

        adapter = PlainAdapter.of([*(bill(c) for c in CUSTOMERS), summary], route)  # type: ignore[list-item]
        registry = registry_of([lookup_customer, charge_card, send_receipt, post_summary])  # type: ignore[list-item]
        journal = Journal(self.db)
        return Scheduler(
            graph=adapter,
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.5),
            target=JournaledModel(ScriptedModel(turns=[]), journal, provider="scripted"),
            policy=Policy(speculation=False),
        )

    async def first(self) -> bool:
        return (await self._scheduler().run(self.run_id, {})).ok

    async def restart(self) -> bool:
        result = await self._scheduler().resume(self.run_id)
        self.why = result.error
        return result.ok


class SpecuNodeReconcile(SpecuNode):
    reconcile = True


BUILDERS: Mapping[str, type[System]] = {
    "plain_loop": PlainLoop,
    "langgraph_nodes": LangGraphNodes,
    "langgraph_tasks": LangGraphTasks,
    "specunode": SpecuNode,
    "specunode_reconcile": SpecuNodeReconcile,
}


# -- scoring -----------------------------------------------------------------------------------


def expected_effects() -> Counter[str]:
    wanted: Counter[str] = Counter()
    for customer in CUSTOMERS:
        wanted[f"charge {customer}"] = 1
        wanted[f"receipt {customer}"] = 1
    wanted["summary"] = 1
    return wanted


def effects_in(world: World) -> Counter[str]:
    seen: Counter[str] = Counter()
    for row in world.tables["charges"].values():
        seen[f"charge {row['customer_id']}"] += 1
    for row in world.tables["messages"].values():
        if "charge_id" in row:
            seen[f"receipt {row['to']}"] += 1
        elif row.get("channel") == "#billing":
            seen["summary"] += 1
    return seen


def receipts_that_lie(world: World) -> list[str]:
    """Receipts naming a charge that does not exist, or that belongs to someone else."""
    charges = world.tables["charges"]
    wrong = []
    for message_id, row in world.tables["messages"].items():
        if "charge_id" not in row:
            continue
        charge = charges.get(str(row["charge_id"]))
        if charge is None or charge.get("customer_id") != row.get("to"):
            wrong.append(message_id)
    return wrong


def score(world: World, finished: bool) -> dict[str, Any]:
    wanted, seen = expected_effects(), effects_in(world)
    duplicated = {k: seen[k] - 1 for k in wanted if seen[k] > 1}
    missing = sorted(k for k in wanted if seen[k] == 0)
    lying = receipts_that_lie(world)
    if lying:
        outcome = "inconsistent"
    elif duplicated:
        outcome = "duplicated"
    elif finished and not missing:
        outcome = "exact"
    elif finished:
        outcome = "lost"
    else:
        outcome = "held"
    return {
        "outcome": outcome,
        "duplicated": duplicated,
        "missing": missing,
        "finished": finished,
        "receipts_that_lie": lying,
    }


async def _bury() -> None:
    """End whatever the crashed process left scheduled, as its death would have."""
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    for task in leftovers:
        task.cancel()
    await asyncio.gather(*leftovers, return_exceptions=True)


async def one(system: str, at: int | None, when: str) -> dict[str, Any]:
    world = standard_world()
    plug = Plug(at=at, when=when)
    with tempfile.TemporaryDirectory() as directory:
        runner = BUILDERS[system](world, plug, Path(directory))
        try:
            finished = await runner.first()
        except Crash:
            await _bury()
            finished = False
        why = None
        if plug.pulled:
            try:
                finished = await runner.restart()
                why = runner.why
            except Exception as exc:
                finished, why = False, f"{type(exc).__name__}: {exc}"
        close_all_writers()
    return {"system": system, "at": at, "when": when, "why": why, **score(world, finished)}


async def measure(systems: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in systems:
        control = await one(system, None, "none")
        points = [await one(system, at, when) for at in range(1, WRITES + 1) for when in WHENS]
        counts = Counter(point["outcome"] for point in points)
        out[system] = {
            "control": control["outcome"],
            "crashes": len(points),
            "exact": counts["exact"],
            "held": counts["held"],
            "duplicated": counts["duplicated"],
            "lost": counts["lost"],
            "inconsistent": counts["inconsistent"],
            "duplicate_effects": sum(sum(p["duplicated"].values()) for p in points),
            "points": points,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("bench/results/crash_safety.json"))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    args = parser.parse_args(argv)
    systems = [s for s in str(args.systems).split(",") if s]

    started = time.monotonic()
    results = asyncio.run(measure(systems))
    import importlib.metadata as metadata

    report = {
        "bench": "crash_safety",
        "writes": WRITES,
        "crash_points": WRITES * len(WHENS),
        "langgraph": metadata.version("langgraph"),
        "systems": results,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nPULL THE PLUG: {report['crash_points']} crashes per system, {WRITES} effects each")
    print(f"  {'system':22s} control  exact  held  duplicated  lost  duplicate effects")
    for name, row in results.items():
        print(
            f"  {name:22s} {row['control']:8s} {row['exact']:5d} {row['held']:5d} "
            f"{row['duplicated']:11d} {row['lost']:5d} {row['duplicate_effects']:18d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
