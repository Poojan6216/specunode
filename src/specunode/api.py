"""The front door: declare tools and nodes, build a graph, run it, resume it.

Every piece here exists elsewhere in the package. What this module adds is the wiring every
first run needs and nobody should have to write -- a journal, a store buffer, a dispatcher and a
scheduler, assembled with every safety check left on -- so that a run is::

    import specunode

    @specunode.tool(effect="write")
    async def charge_card(customer_id: str, amount: float) -> dict: ...

    @specunode.node()
    async def bill(session: specunode.RunSession) -> specunode.Decision: ...

    runtime = specunode.Runtime(specunode.graph([bill], route), tools=[charge_card])
    result = await runtime.run({"customer_id": "cus-1"})
    # killed half way?  await runtime.resume(result.run_id)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from specunode.buffer.dispatcher import Dispatcher
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.effects import ToolRegistry
from specunode.core.graph import GraphAdapter
from specunode.core.model import (
    JournaledModel,
    ModelClient,
    ModelError,
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    call_scope,
)
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.ids import new_ulid
from specunode.integrations.plain import PlainAdapter, registry_of
from specunode.journal.journal import Journal, is_postgres_dsn

__all__ = ["Runtime", "current_idempotency_key", "graph"]

#: Where a run's journal lives unless told otherwise: beside the code that runs it.
DEFAULT_JOURNAL = Path(".specunode") / "journal.db"


def graph(
    nodes: Iterable[Callable[..., Any]],
    route: Callable[[Mapping[str, JsonValue]], str | Sequence[str] | None],
) -> PlainAdapter:
    """A graph from decorated node functions and a router.

    The router is handed the committed state and returns the next node's name, a list of names
    to run side by side and retire in that order, or ``None`` to end the run.
    """
    return PlainAdapter.of(nodes, route)


def current_idempotency_key() -> str:
    """The key the runtime is dispatching this call under. Pass it to the upstream.

    Deterministic: the same call at the same point of the same run derives the same key on a
    resume or a replay, which is what lets an upstream that honours idempotency keys -- and a
    tool's ``reconcile`` -- recognise a request it has already seen. Only defined inside a tool
    while the runtime is dispatching it.
    """
    scope = call_scope.get()
    key = scope.effect_key if scope is not None else ""
    if not key:
        raise RuntimeError(
            "current_idempotency_key() is only defined inside a tool the runtime is "
            "dispatching; a read, or a call made outside a run, has no key"
        )
    return key


@dataclass
class _NoModel:
    """The target of a runtime built without a model: fine until something asks it."""

    def _refuse(self) -> ModelError:
        return ModelError(
            "this Runtime was built without a model, and a node asked one; pass model=..."
        )

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        raise self._refuse()

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        raise self._refuse()
        yield  # type: ignore[unreachable]  # pragma: no cover - makes this an async generator


@dataclass
class Runtime:
    """A graph, its tools and a journal: everything a run and its resume need.

    ``tools`` is a list of functions decorated with ``@specunode.tool``, or a registry.
    ``model`` is the model the graph's nodes ask; a graph whose nodes never ask one needs none.
    ``journal`` is a path to a SQLite file (created if absent) or a ``Journal``.
    """

    graph: GraphAdapter
    tools: ToolRegistry | Sequence[Callable[..., Any]] = ()
    model: ModelClient | None = None
    journal: Journal | str | Path = DEFAULT_JOURNAL
    #: Guessing is off: it needs a drafter, and without one it only adds bookkeeping to every
    #: receipt. Issuing a read the moment its block parses is not guessing, and stays on.
    policy: Policy = field(default_factory=lambda: Policy(speculation=False))
    reducers: Mapping[str, str] = field(default_factory=dict)
    #: Attempts per effect before it is dead-lettered, and the first backoff between them.
    max_attempts: int = 3
    base_delay_ms: float = 50.0

    def _journal(self) -> Journal:
        if isinstance(self.journal, Journal):
            return self.journal
        if is_postgres_dsn(self.journal):
            # Not through ``Path``, which folds the DSN's ``//`` into ``/`` and made a SQLite
            # file of it -- in a folder named ``postgresql:``.
            return Journal(str(self.journal))
        path = Path(self.journal)
        path.parent.mkdir(parents=True, exist_ok=True)
        return Journal(path)

    def _registry(self) -> ToolRegistry:
        if isinstance(self.tools, ToolRegistry):
            return self.tools
        return registry_of(self.tools)

    def scheduler(self) -> Scheduler:
        """A scheduler wired to this runtime's journal -- one per run or resume."""
        journal = self._journal()
        registry = self._registry()
        return Scheduler(
            graph=self.graph,
            registry=registry,
            journal=journal,
            buffer=StoreBuffer(journal=journal, run_id=""),
            dispatcher=Dispatcher(
                registry=registry,
                max_attempts=self.max_attempts,
                base_delay_ms=self.base_delay_ms,
            ),
            target=JournaledModel(self.model or _NoModel(), journal),
            policy=self.policy,
            reducers=dict(self.reducers),
        )

    async def run(
        self, inputs: Mapping[str, JsonValue] | None = None, *, run_id: str | None = None
    ) -> RunResult:
        """Run the graph to the end. ``result.run_id`` is what :meth:`resume` takes."""
        return await self.scheduler().run(run_id or new_ulid(), dict(inputs or {}))

    async def resume(self, run_id: str, *, ask_abandoned: bool = False) -> RunResult:
        """Continue a run a crash interrupted, without sending again what already went out.

        ``ask_abandoned``: where a node keeps waiting on a turn the crashed run had stopped
        waiting for, the model is asked again, live, at the point the node would be stopped
        with ``TurnAbandoned`` -- that turn only, and not one the node stops waiting for again
        as the crashed run did. Its answer may differ from what was acted on.
        """
        return await self.scheduler().resume(run_id, ask_abandoned=ask_abandoned)
