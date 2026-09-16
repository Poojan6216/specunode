"""A fake external world that records who changed it.

Every claim this project makes about effects -- that nothing reaches the world from an
unretired branch, that draining twice sends nothing twice, that a squashed branch's writes
are discarded -- is only as good as the instrument that measures it. This is that instrument.

Two properties make it usable as evidence:

**Every mutation is attributed.** State changes only through :meth:`World._mutate`, which
appends a :class:`Mutation` carrying the branch that issued the call. The leak test is then
one set comparison: the branches that touched the world must be a subset of the branches that
retired. A write path that forgot to record would be a hole in that proof, so
``tests/unit/test_world.py`` walks every public tool and asserts each appends exactly one
record.

**Its tools have *true* semantics, which the runtime does not get to see.** A
:class:`WorldTool` knows whether it really writes and whether it is really idempotent. A
:class:`~specunode.core.effects.ToolSpec` records what the *developer declared*. Those are
different objects on purpose: the gap between them is the runtime's trust boundary, and
attacks 7.1 (a tool declared READ that writes) and 7.10 (a READ whose upstream enqueues work)
live in exactly that gap. The world never consults the declaration.

Reads are recorded too. A speculative read reaches upstream even when its branch is squashed,
which costs real money on a metered API; :attr:`World.reads` is what lets the ledger report
that honestly instead of quietly.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, TextIO

from specunode.canonical import JsonValue, canonical, chash
from specunode.core.model import CallScope, call_scope
from specunode.testing.faults import Faults

__all__ = [
    "CallContext",
    "Mutation",
    "ReadHit",
    "TrueEffect",
    "World",
    "WorldTool",
    "call_scope",
    "current_call_context",
]


class TrueEffect(Enum):
    """What a tool *actually* does upstream, which the runtime never gets to inspect."""

    READ = "read"
    WRITE = "write"
    #: Synchronously returns something harmless; asynchronously enqueues work that writes.
    #: There is no way for a caller to tell this apart from READ by observing the response,
    #: which is the whole point of attack 7.10.
    ASYNC_WRITE = "async_write"


@dataclass(frozen=True, slots=True)
class Mutation:
    """One change that reached the world, and who is responsible for it."""

    sequence: int
    branch_id: str
    effect_key: str
    tool: str
    args_hash: str
    table: str
    row_id: str
    ts: float
    speculative: bool


@dataclass(frozen=True, slots=True)
class ReadHit:
    """One read that reached upstream. Costs money whether or not its branch retires."""

    sequence: int
    branch_id: str
    tool: str
    args_hash: str
    ts: float
    speculative: bool


#: The world reads the *runtime's* call scope rather than keeping its own. One attribution
#: channel means the dispatcher and the instrument cannot disagree about which branch issued a
#: call -- and that attribution is the entire quantity the leak test is built on. A ContextVar
#: because the tool signature belongs to the developer, not to us, and because each asyncio
#: task gets its own copy, so one branch cannot scribble on another's (Hard Rule 6).
CallContext = CallScope

#: What a call made outside any run is attributed to: test setup, or the competing writer of
#: attack 7.3. It must never look like a branch, or seeding a fixture would read as a leak.
EXTERNAL = CallScope(branch_id="<external>")


def current_call_context() -> CallScope:
    scope = call_scope.get()
    return scope if scope is not None and scope.branch_id else EXTERNAL


@contextmanager
def _bound(context: CallScope) -> Iterator[None]:
    token = call_scope.set(context)
    try:
        yield
    finally:
        call_scope.reset(token)


@dataclass(frozen=True)
class WorldTool:
    """A tool as the world really implements it."""

    name: str
    true_effect: TrueEffect
    fn: Callable[..., Awaitable[JsonValue]]
    #: Whether a second delivery of the same call really is a no-op upstream.
    truly_idempotent: bool = False
    #: Whether reads of this tool return a version alongside the value.
    witness: bool = False


@dataclass
class World:
    """An in-memory world with tables, tools, a mutation log and injectable faults."""

    tables: dict[str, dict[str, dict[str, JsonValue]]] = field(default_factory=dict)
    versions: dict[tuple[str, str], int] = field(default_factory=dict)
    mutations: list[Mutation] = field(default_factory=list)
    reads: list[ReadHit] = field(default_factory=list)
    faults: Faults = field(default_factory=Faults)
    #: Work enqueued by an ASYNC_WRITE tool and not yet run. Attack 7.10 drains this after
    #: the branch has been squashed, to show the effect lands anyway.
    pending_jobs: list[tuple[str, str, Mapping[str, JsonValue]]] = field(default_factory=list)

    #: When set, every mutation and read is appended here and fsynced before the call
    #: returns, and reopening the world replays the file. Tasks 2.5 and 5.2 SIGKILL a
    #: subprocess and then compare the resumed run's mutations against the uninterrupted
    #: run's; an in-memory world dies with the process it was killed in, so the resumed
    #: process could not see a duplicate dispatch of an effect the dead one already sent --
    #: which is the whole thing those tests exist to catch.
    log_path: Path | None = None

    _sequence: int = 0
    #: Effect keys already applied, so a truly idempotent tool collapses repeat deliveries.
    _applied_keys: set[str] = field(default_factory=set)
    _tools: dict[str, WorldTool] = field(default_factory=dict, repr=False)
    _log: TextIO | None = field(default=None, repr=False)
    _replaying: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        for table in ("customers", "tickets", "jobs", "messages", "charges", "docs"):
            self.tables.setdefault(table, {})
        if not self._tools:
            self._tools = self._build_tools()
        if self.log_path is not None:
            self._open_log(self.log_path)

    # -- durability ------------------------------------------------------------------------

    def _open_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._recover(path)
        self._log = path.open("a", encoding="utf-8")

    def _recover(self, path: Path) -> None:
        """Rebuild state from the log.

        Replaying the recorded row rather than re-running the tool: a tool call can fail, be
        partitioned or be duplicated, and re-running one during recovery would have to
        reproduce all of that. The record already says what the row became.
        """
        self._replaying = True
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line means the process died mid-append. Everything before
                    # it is intact, which is exactly the guarantee these tests need.
                    break
                self._replay_event(event)
        finally:
            self._replaying = False

    def _replay_event(self, event: Mapping[str, JsonValue]) -> None:
        kind = event.get("kind")
        self._sequence = max(self._sequence, _as_int(event.get("sequence")))
        if kind == "read":
            self.reads.append(
                ReadHit(
                    sequence=_as_int(event.get("sequence")),
                    branch_id=str(event.get("branch_id", "")),
                    tool=str(event.get("tool", "")),
                    args_hash=str(event.get("args_hash", "")),
                    ts=_as_float(event.get("ts")),
                    speculative=bool(event.get("speculative", False)),
                )
            )
            return
        table, row_id = str(event.get("table", "")), str(event.get("row_id", ""))
        self.mutations.append(
            Mutation(
                sequence=_as_int(event.get("sequence")),
                branch_id=str(event.get("branch_id", "")),
                effect_key=str(event.get("effect_key", "")),
                tool=str(event.get("tool", "")),
                args_hash=str(event.get("args_hash", "")),
                table=table,
                row_id=row_id,
                ts=_as_float(event.get("ts")),
                speculative=bool(event.get("speculative", False)),
            )
        )
        self.versions[(table, row_id)] = self.versions.get((table, row_id), 0) + 1
        row_after = event.get("row_after")
        if isinstance(row_after, Mapping):
            self.tables.setdefault(table, {})[row_id] = dict(row_after)
        elif row_after is None:
            self.tables.setdefault(table, {}).pop(row_id, None)
        key = event.get("effect_key")
        if isinstance(key, str) and key:
            self._applied_keys.add(key)

    def _append_log(self, event: Mapping[str, JsonValue]) -> None:
        if self._log is None or self._replaying:
            return
        self._log.write(canonical(event).decode("utf-8") + "\n")
        self._log.flush()
        os.fsync(self._log.fileno())

    def _persist_mutation(self, record: Mutation, row_after: JsonValue) -> None:
        self._append_log(
            {
                "kind": "mutation",
                "sequence": record.sequence,
                "branch_id": record.branch_id,
                "effect_key": record.effect_key,
                "tool": record.tool,
                "args_hash": record.args_hash,
                "table": record.table,
                "row_id": record.row_id,
                "ts": record.ts,
                "speculative": record.speculative,
                "row_after": row_after,
            }
        )

    def _persist_read(self, record: ReadHit) -> None:
        self._append_log(
            {
                "kind": "read",
                "sequence": record.sequence,
                "branch_id": record.branch_id,
                "tool": record.tool,
                "args_hash": record.args_hash,
                "ts": record.ts,
                "speculative": record.speculative,
            }
        )

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def _build_tools(self) -> dict[str, WorldTool]:
        """The world's tools, with the semantics the world really has.

        ``true_effect`` and ``truly_idempotent`` here are facts about the upstream. What the
        runtime believes lives in a ToolSpec and can disagree; that disagreement is the trust
        boundary attacks 7.1 and 7.10 exercise.
        """
        r, w, a = TrueEffect.READ, TrueEffect.WRITE, TrueEffect.ASYNC_WRITE
        definitions: list[tuple[str, TrueEffect, bool, bool]] = [
            # name, true effect, truly idempotent, returns a witness
            ("lookup_customer", r, True, True),
            ("get_ticket", r, True, True),
            ("get_pipeline_status", r, True, True),
            ("fetch_runbook", r, True, False),
            ("search_docs", r, True, False),
            ("create_ticket", w, True, False),
            ("update_ticket", w, True, False),
            ("restart_job", w, True, False),
            ("post_summary", w, False, False),
            ("charge_card", w, False, False),
            ("send_receipt", w, False, False),
            ("send_email", w, False, False),
            ("reserve_capacity", w, False, False),
            ("release_capacity", w, False, False),
            ("enqueue_reindex", a, False, False),
        ]
        return {
            name: WorldTool(
                name=name,
                true_effect=effect,
                fn=getattr(self, name),
                truly_idempotent=idempotent,
                witness=witness,
            )
            for name, effect, idempotent, witness in definitions
        }

    def tools(self) -> Mapping[str, WorldTool]:
        """Every tool this world exposes, keyed by name."""
        return dict(self._tools)

    def truly_mutating_tools(self) -> set[str]:
        """Tools that really change something, including the ones that only look like reads."""
        return {
            name
            for name, tool in self._tools.items()
            if tool.true_effect in (TrueEffect.WRITE, TrueEffect.ASYNC_WRITE)
        }

    # -- attribution ---------------------------------------------------------------------

    def bind(
        self, *, branch_id: str, effect_key: str = "", speculative: bool = False
    ) -> AbstractContextManager[None]:
        """Attribute every world call inside this block to ``branch_id``."""
        return _bound(
            CallScope(branch_id=branch_id, effect_key=effect_key, speculative=speculative)
        )

    # -- the single funnel every state change passes through -----------------------------

    def _mutate(
        self,
        tool: str,
        args: Mapping[str, JsonValue],
        table: str,
        row_id: str,
        apply: Callable[[], None] | None = None,
    ) -> None:
        """Apply and record a mutation. The only place ``self.tables`` is allowed to change.

        The state change runs *inside* the funnel rather than beside it, for two reasons.
        It makes "no tool can move state without recording it" structural rather than a
        convention -- disabling this method leaves the tables untouched whatever the tool
        does. And it means the record can carry the resulting row, which is what lets the
        durable log be replayed on recovery without re-running any tool.
        """
        context = current_call_context()
        self._sequence += 1
        if apply is not None:
            apply()
        self.versions[(table, row_id)] = self.versions.get((table, row_id), 0) + 1
        row = self.tables.get(table, {}).get(row_id)
        record = Mutation(
            sequence=self._sequence,
            branch_id=context.branch_id,
            effect_key=context.effect_key,
            tool=tool,
            args_hash=chash(dict(args)),
            table=table,
            row_id=row_id,
            ts=time.monotonic(),
            speculative=context.speculative,
        )
        self.mutations.append(record)
        self._persist_mutation(record, dict(row) if row is not None else None)

    def _record_read(self, tool: str, args: Mapping[str, JsonValue]) -> None:
        context = current_call_context()
        self._sequence += 1
        record = ReadHit(
            sequence=self._sequence,
            branch_id=context.branch_id,
            tool=tool,
            args_hash=chash(dict(args)),
            ts=time.monotonic(),
            speculative=context.speculative,
        )
        self.reads.append(record)
        self._persist_read(record)

    # -- queries the tests ask ------------------------------------------------------------

    def mutating_branches(self) -> set[str]:
        """Every branch that changed the world. The leak test's left-hand side."""
        return {m.branch_id for m in self.mutations}

    def mutations_by(self, tool: str) -> list[Mutation]:
        return [m for m in self.mutations if m.tool == tool]

    def reads_from(self, branch_ids: set[str]) -> list[ReadHit]:
        """Reads charged to these branches -- the cost of a speculation that was thrown away."""
        return [r for r in self.reads if r.branch_id in branch_ids]

    def witness_of(self, table: str, row_id: str) -> str:
        """The per-row version counter, used for read validation at retirement (rule E3)."""
        return f"{table}:{row_id}:{self.versions.get((table, row_id), 0)}"

    def snapshot(self) -> JsonValue:
        return {
            table: {row: dict(values) for row, values in sorted(rows.items())}
            for table, rows in sorted(self.tables.items())
        }

    # -- fault controls, named as the spec names them --------------------------------------

    def partition(self, *, at: int | None = None) -> None:
        self.faults.partition(at=at)

    def heal(self) -> None:
        self.faults.heal()

    def timeout(self, tool: str) -> None:
        self.faults.timeout(tool)

    def duplicate_delivery(self, tool: str) -> None:
        self.faults.duplicate_delivery(tool)

    def slow(self, kind: Literal["read", "write"], ms: int) -> None:
        self.faults.slow(kind, ms)

    # -- seeding (attributed to <external>, so it never pollutes a leak assertion) ---------

    def seed(self, table: str, row_id: str, **values: JsonValue) -> None:
        self.tables.setdefault(table, {})[row_id] = dict(values)
        self.versions[(table, row_id)] = self.versions.get((table, row_id), 0)

    def drain_pending_jobs(self) -> int:
        """Run the background work an ASYNC_WRITE tool enqueued (attack 7.10).

        Runs it under the *original* branch's attribution, because that is the honest
        accounting: the branch that made the call is responsible for the effect, even though
        the effect lands after the branch has been squashed.
        """
        count = 0
        while self.pending_jobs:
            branch_id, tool, args = self.pending_jobs.pop(0)
            job_id = str(args.get("job_id", "unknown"))

            def apply(job_id: str = job_id) -> None:
                row = self.tables["jobs"].setdefault(job_id, {})
                row["reindexed"] = _as_int(row.get("reindexed")) + 1

            with self.bind(branch_id=branch_id, effect_key="<async>", speculative=False):
                self._mutate(tool, args, "jobs", job_id, apply)
            count += 1
        return count

    # -- reading ---------------------------------------------------------------------------

    def _read_row(self, table: str, row_id: str, *, witness: bool) -> JsonValue:
        row = self.tables.get(table, {}).get(row_id)
        value: JsonValue = dict(row) if row is not None else None
        if not witness:
            return value
        return {"value": value, "witness": self.witness_of(table, row_id)}

    async def _read(
        self, tool: str, args: Mapping[str, JsonValue], table: str, row_id: str, *, witness: bool
    ) -> JsonValue:
        self.faults.before_call(tool)
        # Recorded before the latency, not after: the request reaches upstream when it is
        # sent. A branch squashed mid-read still cost whatever that read costs, and
        # attack 7.2 is about reporting that rather than hiding it.
        self._record_read(tool, args)
        await self.faults.delay(write=False)
        return self._read_row(table, row_id, witness=witness)

    async def _write(
        self,
        tool: str,
        args: Mapping[str, JsonValue],
        table: str,
        row_id: str,
        apply: Callable[[], None] | None = None,
    ) -> JsonValue:
        """Apply a write once per delivery, recording every delivery.

        A truly idempotent tool collapses repeated deliveries into one state change but still
        logs both, because "the upstream received it twice" and "the upstream changed twice"
        are different facts and attack 7.4 needs to tell them apart.
        """
        self.faults.before_call(tool)
        await self.faults.delay(write=True)
        spec = self._tools[tool]
        applied = 0
        for _ in range(self.faults.deliveries(tool)):
            key = current_call_context().effect_key
            already = bool(spec.truly_idempotent and key and key in self._applied_keys)
            self._mutate(tool, args, table, row_id, None if already else apply)
            if not already:
                applied += 1
                if key:
                    self._applied_keys.add(key)
        return {"ok": True, "table": table, "id": row_id, "applied": applied}

    # -- the tools themselves --------------------------------------------------------------

    async def lookup_customer(self, customer_id: str) -> JsonValue:
        return await self._read(
            "lookup_customer", {"customer_id": customer_id}, "customers", customer_id, witness=True
        )

    async def get_ticket(self, ticket_id: str) -> JsonValue:
        return await self._read(
            "get_ticket", {"ticket_id": ticket_id}, "tickets", ticket_id, witness=True
        )

    async def get_pipeline_status(self, pipeline_id: str) -> JsonValue:
        return await self._read(
            "get_pipeline_status",
            {"pipeline_id": pipeline_id},
            "jobs",
            pipeline_id,
            witness=True,
        )

    async def fetch_runbook(self, section: str) -> JsonValue:
        """A read with no witness: there is no version to re-check at retirement.

        The ledger reports reads like this one as *unwitnessed*, never as fresh (attack 7.3).
        """
        return await self._read(
            "fetch_runbook", {"section": section}, "docs", section, witness=False
        )

    async def search_docs(self, query: str) -> JsonValue:
        self.faults.before_call("search_docs")
        self._record_read("search_docs", {"query": query})
        await self.faults.delay(write=False)
        hits = sorted(
            row_id
            for row_id, row in self.tables["docs"].items()
            if query in str(row.get("text", ""))
        )
        return {"query": query, "hits": hits}

    async def create_ticket(self, customer_id: str, title: str) -> JsonValue:
        ticket_id = f"tkt-{chash({'c': customer_id, 't': title})[:8]}"
        args: dict[str, JsonValue] = {"customer_id": customer_id, "title": title}

        def apply() -> None:
            self.tables["tickets"].setdefault(
                ticket_id, {"customer_id": customer_id, "title": title, "status": "open"}
            )

        result = await self._write("create_ticket", args, "tickets", ticket_id, apply)
        return {**result, "ticket_id": ticket_id}  # type: ignore[dict-item]

    async def update_ticket(self, ticket_id: str, status: str) -> JsonValue:
        args: dict[str, JsonValue] = {"ticket_id": ticket_id, "status": status}

        def apply() -> None:
            self.tables["tickets"].setdefault(ticket_id, {})["status"] = status

        return await self._write("update_ticket", args, "tickets", ticket_id, apply)

    async def restart_job(self, job_id: str) -> JsonValue:
        args: dict[str, JsonValue] = {"job_id": job_id}

        def apply() -> None:
            row = self.tables["jobs"].setdefault(job_id, {"restarts": 0, "status": "unknown"})
            row["status"] = "running"
            row["restarts"] = _as_int(row.get("restarts")) + 1

        return await self._write("restart_job", args, "jobs", job_id, apply)

    async def post_summary(self, channel: str, text: str) -> JsonValue:
        message_id = f"msg-{len(self.tables['messages']) + 1}"
        args: dict[str, JsonValue] = {"channel": channel, "text": text}

        def apply() -> None:
            self.tables["messages"][message_id] = {"channel": channel, "text": text}

        result = await self._write("post_summary", args, "messages", message_id, apply)
        return {**result, "message_id": message_id}  # type: ignore[dict-item]

    async def charge_card(self, customer_id: str, amount: float) -> JsonValue:
        """Not idempotent, by design. Demo 1 counts how many of these reach the world."""
        charge_id = f"chg-{len(self.tables['charges']) + 1}"
        args: dict[str, JsonValue] = {"customer_id": customer_id, "amount": amount}

        def apply() -> None:
            self.tables["charges"][charge_id] = {"customer_id": customer_id, "amount": amount}
            customer = self.tables["customers"].setdefault(customer_id, {"balance": 0.0})
            customer["balance"] = _as_float(customer.get("balance")) - float(amount)

        result = await self._write("charge_card", args, "charges", charge_id, apply)
        return {**result, "charge_id": charge_id}  # type: ignore[dict-item]

    async def send_receipt(self, customer_id: str, charge_id: str) -> JsonValue:
        message_id = f"rcpt-{len(self.tables['messages']) + 1}"
        args: dict[str, JsonValue] = {"customer_id": customer_id, "charge_id": charge_id}

        def apply() -> None:
            self.tables["messages"][message_id] = {"to": customer_id, "charge_id": charge_id}

        return await self._write("send_receipt", args, "messages", message_id, apply)

    async def send_email(self, to: str, subject: str, body: str) -> JsonValue:
        """Irreversible: there is no compensating action that unsends an email."""
        message_id = f"eml-{len(self.tables['messages']) + 1}"
        args: dict[str, JsonValue] = {"to": to, "subject": subject, "body": body}

        def apply() -> None:
            self.tables["messages"][message_id] = {"to": to, "subject": subject, "body": body}

        return await self._write("send_email", args, "messages", message_id, apply)

    async def reserve_capacity(self, job_id: str, units: int) -> JsonValue:
        args: dict[str, JsonValue] = {"job_id": job_id, "units": units}

        def apply() -> None:
            row = self.tables["jobs"].setdefault(job_id, {})
            row["reserved"] = _as_int(row.get("reserved")) + int(units)

        return await self._write("reserve_capacity", args, "jobs", job_id, apply)

    async def release_capacity(self, job_id: str, units: int) -> JsonValue:
        """The compensator for :meth:`reserve_capacity` -- a *second* effect, not an undo."""
        args: dict[str, JsonValue] = {"job_id": job_id, "units": units}

        def apply() -> None:
            row = self.tables["jobs"].setdefault(job_id, {})
            row["reserved"] = _as_int(row.get("reserved")) - int(units)

        return await self._write("release_capacity", args, "jobs", job_id, apply)

    async def enqueue_reindex(self, index: str) -> JsonValue:
        """Looks exactly like a read. Is not one (attack 7.10).

        The synchronous response is ``{"status": "queued", ...}`` and nothing about it tells
        a caller that a background job will later write. A tool whose upstream enqueues,
        schedules or triggers work is a WRITE whatever its HTTP verb says.
        """
        self.faults.before_call("enqueue_reindex")
        self._record_read("enqueue_reindex", {"index": index})
        await self.faults.delay(write=False)
        job_id = f"reindex-{index}-{len(self.pending_jobs) + 1}"
        self.pending_jobs.append(
            (
                current_call_context().branch_id,
                "enqueue_reindex",
                {"index": index, "job_id": job_id},
            )
        )
        return {"status": "queued", "job_id": job_id}


def _as_int(value: JsonValue, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number in a world row, got {value!r}")
    return int(value)


def _as_float(value: JsonValue, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number in a world row, got {value!r}")
    return float(value)


def standard_world() -> World:
    """A world seeded with the rows the three sample apps and the demos expect.

    Seeding is attributed to ``<external>``, so it never appears in a leak assertion.
    """
    world = World()
    for index in range(1, 6):
        world.seed("customers", f"cus-{index}", name=f"Customer {index}", balance=100.0, plan="pro")
    for index, status in enumerate(["running", "failed", "queued", "failed"], start=1):
        world.seed("jobs", f"etl-{index}", status=status, restarts=0, reserved=0)
    for section, text in [
        ("restart", "Restart the job, then confirm the status flips to running."),
        ("escalate", "Page the on-call engineer if two restarts fail."),
        ("billing", "Refunds are issued by finance, never by the agent."),
    ]:
        world.seed("docs", section, text=text)
    return world
