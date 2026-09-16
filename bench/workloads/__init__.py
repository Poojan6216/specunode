"""The three sample apps, as a registry the tests and the benchmarks both iterate.

Spec task 5.1 says the equivalence test runs "for every workload in ``bench/workloads/``".
This is that directory, and it exists so the list of workloads is written down *once*. When
the three mandatory tests each carried their own copy of "which apps exist", adding a fourth
app meant remembering three places, and the failure mode of forgetting was a test that still
passed while covering less than it claimed to.

Each entry carries what a run needs and, importantly, **what the run is supposed to produce**:
``expect_effects`` is the workload's own declaration of how many effects reach the world. The
equivalence test compares two ledgers, and two ledgers that are both empty are equal -- so a
runtime that dispatched nothing would pass a comparison it never really ran. The declared count
is what stops that, which is why it lives with the workload rather than in the test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from specunode.canonical import JsonValue
from specunode.core.decision import ToolCall
from specunode.core.effects import ToolRegistry
from specunode.core.graph import GraphAdapter
from specunode.core.model import ModelResponse
from specunode.testing.models import ScriptedModel, tool_turn
from specunode.testing.world import World

__all__ = ["WORKLOADS", "Workload", "workload_ids"]


@dataclass(frozen=True)
class Workload:
    """One sample app, plus everything needed to run it reproducibly."""

    name: str
    #: ``(world) -> (graph, registry)``; the app's own ``build``.
    build: Callable[[World], tuple[GraphAdapter, object]]
    #: The state the run starts from.
    seed: Mapping[str, JsonValue]
    #: The calls the scripted target model emits, in one turn. This is ground truth: a branch
    #: retires only if its guess equals the corresponding call, byte for byte. Most workloads
    #: emit one -- ``ops_agent`` emits several, which is what gives the drafters a context to
    #: predict from.
    decision: tuple[tuple[str, Mapping[str, JsonValue]], ...]
    #: How many effects must reach the world. Declared, so an empty comparison cannot pass.
    expect_effects: int
    #: Whether the app hands its model turn to the runtime via ``session.call_turn``. False
    #: means it calls the model directly and issues each tool itself -- a supported pattern,
    #: and the one Demo 1 uses, but one that routes around tier-0 early issue and the drafters
    #: entirely, so no drafter is ever consulted on such a run.
    drives_turn: bool = False
    #: Whether the tier-1 drafter can offer anything on this workload. False for the
    #: one-call-per-turn shape, which is most real traffic and all of the offline corpus; the
    #: equivalence test asserts this rather than assuming it, so it stays true on purpose.
    tier_1_can_predict: bool = False
    #: Latency between streamed blocks. Non-zero only where a later block's prediction needs an
    #: earlier block's read to have finished, which is otherwise a race rather than a test.
    block_delay_ms: float = 0.0
    #: The tool calls a retired run makes, in order, for training a tier-1 index without
    #: needing a prior journal. Written out rather than mined so that a bug in mining cannot
    #: quietly turn the tier-1 arm into a tier-0 arm that still reports as tier 1.
    trace: Sequence[tuple[str, Mapping[str, JsonValue]]] = field(default_factory=tuple)

    def turns(self) -> list[ModelResponse]:
        """The script. One turn: these workloads have exactly one model decision point."""
        return [tool_turn(*self.decision, turn=0)]

    def model(self) -> ScriptedModel:
        return ScriptedModel(turns=self.turns(), block_delay_ms=self.block_delay_ms)

    def decisions(self) -> list[ToolCall]:
        return [ToolCall(name=name, args=dict(args)) for name, args in self.trace]

    def make(self, world: World) -> tuple[GraphAdapter, ToolRegistry]:
        graph, registry = self.build(world)
        if not isinstance(registry, ToolRegistry):  # pragma: no cover - a build() contract bug
            raise TypeError(f"{self.name}.build did not return a ToolRegistry")
        return graph, registry


def _support() -> Workload:
    from examples.support_agent.agent import build

    return Workload(
        name="support_agent",
        build=build,
        seed={"customer_id": "cus-1"},
        decision=(("charge_card", {"customer_id": "cus-1", "amount": 25.0}),),
        # charge_card, send_receipt.
        expect_effects=2,
        trace=[
            ("lookup_customer", {"customer_id": "cus-1"}),
            ("charge_card", {"customer_id": "cus-1", "amount": 25.0}),
            ("send_receipt", {"customer_id": "cus-1", "charge_id": "chg-1"}),
        ],
    )


def _ops() -> Workload:
    from examples.ops_agent.agent import build

    return Workload(
        name="ops_agent",
        build=build,
        seed={"pipeline_id": "etl-2"},
        # Three calls in one turn: two reads the runtime issues early, then the write the
        # drafter can predict from the first read's result.
        decision=(
            ("get_pipeline_status", {"pipeline_id": "etl-2"}),
            ("fetch_runbook", {"section": "restart"}),
            ("restart_job", {"job_id": "etl-2"}),
        ),
        # restart_job, reserve_capacity, post_summary.
        expect_effects=3,
        drives_turn=True,
        tier_1_can_predict=True,
        # Long enough that block 1's early-issued read has returned before the drafter is asked
        # after block 2. Without it the prediction's argument is not yet knowable.
        block_delay_ms=25.0,
        trace=[
            ("get_pipeline_status", {"pipeline_id": "etl-2"}),
            ("fetch_runbook", {"section": "restart"}),
            ("restart_job", {"job_id": "etl-2"}),
        ],
    )


def _research() -> Workload:
    from examples.research_agent.agent import build

    return Workload(
        name="research_agent",
        build=build,
        seed={"customer_id": "cus-3", "query": "Restart"},
        decision=(
            ("send_email", {"to": "cus-3", "subject": "Findings", "body": "See the runbook."}),
        ),
        # send_email, post_summary.
        expect_effects=2,
        trace=[
            ("search_docs", {"query": "Restart"}),
            ("fetch_runbook", {"section": "restart"}),
            ("fetch_runbook", {"section": "escalate"}),
            ("lookup_customer", {"customer_id": "cus-3"}),
            ("send_email", {"to": "cus-3", "subject": "Findings", "body": "See the runbook."}),
            ("post_summary", {"channel": "#research", "text": "emailed cus-3: Findings"}),
        ],
    )


#: Built lazily on import of this module, in a fixed order, so parametrised test ids are stable.
WORKLOADS: tuple[Workload, ...] = (_support(), _ops(), _research())


def workload_ids() -> list[str]:
    return [w.name for w in WORKLOADS]
