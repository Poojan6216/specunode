"""Attacks on our own runtime, with measured rates (spec Phase 7).

Every strategy here is a way to defeat SpecuNode, and most of them work. That is the point: the
README's "What beats it" section is generated from this file's output rather than written from
memory, so a hole cannot quietly stop being reported.

A strategy that fails to run is an **error row**, never a dropped row. A suite that silently
drops the attacks it cannot execute reports a clean sheet for the wrong reason.

``python bench/adversarial/run_attacks.py --all``
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from specunode.canonical import JsonValue, canonical
from specunode.core.branch import Branch, BranchStatus
from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.core.hazards import handle_for, has_handle
from specunode.ids import new_ulid
from specunode.journal.journal import Journal
from specunode.testing.world import World, standard_world

__all__ = ["ATTACKS", "AttackResult", "run_all"]


@dataclass
class AttackResult:
    """One strategy's measured outcome."""

    id: str
    name: str
    #: What the runtime cannot prevent, stated before the number so the number is readable.
    claim: str
    measured: dict[str, float | int | str] = field(default_factory=dict)
    defeated_runtime: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def registry_with(world: World, **overrides: ToolSpec) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in overrides.values():
        registry.register(spec)
    return registry


async def attack_71_misdeclared_tool() -> AttackResult:
    """7.1 — a tool declared READ that actually writes.

    The runtime's trust boundary, exercised directly. A READ is executed speculatively, so a
    branch that never retires still performed the write. Nothing can detect this: the effect
    class is the developer's word and the runtime has no independent view of what a tool does.
    """
    world = standard_world()
    # Declared READ. Actually restart_job, which mutates.
    registry = registry_with(
        world,
        liar=ToolSpec(name="restart_job", effect=EffectClass.READ, fn=world.restart_job),
    )
    spec = registry.get("restart_job")
    doomed = Branch(id="br-doomed", status=BranchStatus.SPECULATIVE)

    # A speculative branch runs a READ immediately -- that is the whole design.
    with world.bind(branch_id=doomed.id, effect_key="", speculative=True):
        await spec.fn(job_id="etl-1")
    doomed.squash("mismatch")

    leaked = [m for m in world.mutations if m.branch_id == doomed.id]
    return AttackResult(
        id="7.1",
        name="misdeclared tool (READ that writes)",
        claim="defeats the store buffer completely; the runtime cannot detect it and does not try",
        measured={
            "effects_from_squashed_branch": len(leaked),
            "leak_rate": 1.0 if leaked else 0.0,
        },
        defeated_runtime=bool(leaked),
    )


async def attack_72_read_with_side_effects() -> AttackResult:
    """7.2 — a READ whose upstream bills, rate-limits or audits.

    Not a leak, a cost. Speculative reads reach real systems whether or not the branch retires,
    and the honest response is to count them rather than net them off against the latency.
    """
    world = standard_world()
    registry = registry_with(
        world,
        lookup=ToolSpec(
            name="lookup_customer", effect=EffectClass.READ, fn=world.lookup_customer, witness=True
        ),
    )
    spec = registry.get("lookup_customer")
    squashed = []
    for index in range(10):
        branch = Branch(id=f"br-{index}", status=BranchStatus.SPECULATIVE)
        with world.bind(branch_id=branch.id, effect_key="", speculative=True):
            await spec.fn(customer_id="cus-1")
        if index % 2:
            branch.squash("mismatch")
            squashed.append(branch.id)

    wasted = world.reads_from(set(squashed))
    return AttackResult(
        id="7.2",
        name="read with upstream side effects",
        claim=(
            "speculative reads reach upstream from branches that never retire; "
            "counted, never hidden"
        ),
        measured={
            "reads_issued": len(world.reads),
            "reads_from_squashed_branches": len(wasted),
        },
        defeated_runtime=bool(wasted),
    )


async def attack_73_stale_reads_under_contention() -> AttackResult:
    """7.3 — another actor mutates rows between the speculative read and confirmation.

    The measured number that matters is not the stale rate; it is the fraction of stale reads
    that were **unwitnessed**, because those are the ones the runtime cannot detect at all.
    """
    from specunode.canonical import chash
    from specunode.core.branch import ReadRecord
    from specunode.verify.witness import validate_reads

    world = standard_world()
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

    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    for tool, args in (
        ("get_pipeline_status", {"pipeline_id": "etl-1"}),
        ("fetch_runbook", {"section": "restart"}),
    ):
        value = await registry.get(tool).fn(**args)
        witness = value.get("witness") if isinstance(value, dict) else None
        branch.read_set.append(
            ReadRecord(
                tool=tool,
                args=args,
                args_hash=chash(args),
                result_hash=chash(value),
                witness=witness,
                at_step=1,
            )
        )

    # The competing actor. Both rows move; only one of the reads can notice.
    await world.restart_job(job_id="etl-1")
    world.seed("docs", "restart", text="the runbook changed under us")

    validation = await validate_reads(branch, registry)
    undetectable = validation.unwitnessed
    return AttackResult(
        id="7.3",
        name="stale reads under contention",
        claim="a read without a witness cannot be checked; that fraction is the honest number",
        measured={
            "reads": validation.total,
            "detected_stale": validation.stale,
            "unwitnessed_and_therefore_undetectable": undetectable,
            "undetectable_fraction": (
                round(undetectable / validation.total, 3) if validation.total else 0.0
            ),
        },
        defeated_runtime=undetectable > 0,
    )


async def attack_74_duplicate_delivery() -> AttackResult:
    """7.4 — the network duplicates a delivery of a non-idempotent tool.

    Dedupe covers the dispatcher's own retries. It cannot cover duplication beyond the
    dispatcher, and this measures what that costs.
    """
    world = standard_world()
    world.duplicate_delivery("charge_card")
    registry = ToolRegistry()
    registry.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))
    with world.bind(branch_id="br-1", effect_key="k-1"):
        await registry.get("charge_card").fn(customer_id="cus-1", amount=10.0)

    deliveries = len(world.mutations_by("charge_card"))
    return AttackResult(
        id="7.4",
        name="duplicate delivery of a non-idempotent tool",
        claim="dedupe protects the dispatcher's retries, not the network beyond it",
        measured={"intended": 1, "delivered": deliveries, "duplicates": deliveries - 1},
        defeated_runtime=deliveries > 1,
    )


async def attack_75_return_value_laundering() -> AttackResult:
    """7.5 — hide a staged write's handle so hazard analysis does not see it.

    Reported as a measured miss rate over a committed matrix of cases, so the docs and the
    bench cannot disagree about which transforms are caught.
    """
    handle = handle_for("01ABCDEFGHJKMNPQRSTVWXYZ01")
    cases: list[tuple[str, object]] = [
        ("L1 exact value", {"id": handle}),
        ("L2 embedded in a longer string", {"note": f"see {handle} for details"}),
        ("L3 nested three deep", {"a": {"b": [{"c": handle}]}}),
        ("L4 used as an object key", {handle: "value"}),
        ("L5 prefix case-mangled", {"id": handle.replace("specunode", "SpecuNode")}),
        ("L6 base64-encoded", {"id": base64.b64encode(handle.encode()).decode()}),
        ("L7 hex-encoded", {"id": handle.encode().hex()}),
        # Two different splits, because only one of them actually hides anything. Splitting
        # after the prefix leaves "$specunode.handle:" intact in one argument and the scan
        # still fires; splitting *inside* the prefix is the evasion.
        ("L8 split after the prefix", {"prefix": handle[:20], "suffix": handle[20:]}),
        ("L9 split inside the prefix", {"prefix": handle[:8], "suffix": handle[8:]}),
    ]
    caught = {}
    for label, args in cases:
        caught[label] = has_handle(canonical(args))  # type: ignore[arg-type]

    misses = [label for label, hit in caught.items() if not hit]
    return AttackResult(
        id="7.5",
        name="return-value laundering",
        claim="a handle transformed inside a string can evade the scan; encodings are not caught",
        measured={
            "cases": len(cases),
            "caught": sum(1 for hit in caught.values() if hit),
            "missed": len(misses),
            "miss_rate": round(len(misses) / len(cases), 3),
            "missed_cases": ", ".join(misses),
        },
        defeated_runtime=bool(misses),
    )


async def attack_76_prompt_injected_tool_call() -> AttackResult:
    """7.6 — a read result tells the model to send an email.

    Two halves. If only the drafter predicts it, it squashes and nothing is sent. If the target
    model actually emits it, it is dispatched -- because SpecuNode decides *when* a call may
    take effect, not *whether* it is allowed to.
    """
    from specunode.verify.gate import resolve_decision

    injected = ToolCall("send_email", {"to": "attacker@example.com", "subject": "x", "body": "y"})
    innocent = ToolCall("fetch_runbook", {"section": "restart"})

    # The drafter takes the bait; the model does not.
    predicted_only = resolve_decision(injected, innocent)
    # The model takes the bait.
    model_emitted = resolve_decision(injected, injected)

    return AttackResult(
        id="7.6",
        name="prompt-injected tool call",
        claim="not an authorization layer: a call the model actually emits is dispatched",
        measured={
            "dispatched_when_only_the_drafter_predicted_it": (
                0 if predicted_only is BranchStatus.SQUASHED else 1
            ),
            "dispatched_when_the_model_emitted_it": (
                1 if model_emitted is BranchStatus.CONFIRMED else 0
            ),
        },
        defeated_runtime=model_emitted is BranchStatus.CONFIRMED,
    )


async def attack_710_async_side_effect_behind_a_read() -> AttackResult:
    """7.10 — a READ whose synchronous response is harmless and whose upstream enqueues a write.

    The branch is squashed; the background job runs anyway. There is no way to tell this apart
    from a real read by looking at the response, which is why the rule is about the upstream
    rather than about the verb.
    """
    world = standard_world()
    doomed = Branch(id="br-doomed", status=BranchStatus.SPECULATIVE)
    with world.bind(branch_id=doomed.id, effect_key="", speculative=True):
        response = await world.enqueue_reindex(index="docs")
    doomed.squash("mismatch")
    before = len(world.mutations)
    landed = world.drain_pending_jobs()
    leaked = [m for m in world.mutations[before:] if m.branch_id == doomed.id]

    return AttackResult(
        id="7.10",
        name="asynchronous side effect behind a READ",
        claim="a tool that enqueues, schedules or triggers work is a WRITE whatever its verb says",
        measured={
            "synchronous_response_looks_like_a_read": 1
            if isinstance(response, dict) and response.get("status") == "queued"
            else 0,
            "mutations_at_squash_time": before,
            "effects_landing_after_the_squash": landed,
            "leaked_effects_per_squashed_branch": len(leaked),
        },
        defeated_runtime=bool(leaked),
    )


async def attack_79_staging_an_irreversible_effect() -> AttackResult:
    """7.9 — speculate past an IRREVERSIBLE effect with stage_irreversible on.

    Correct, and still worth showing: the effect retires on a decision the model did make, but
    an effect with no undo was held and released by machinery rather than by a person. The
    default is off, and the ledger names the row.
    """
    from specunode.core.hazards import Hazard, analyse
    from specunode.core.policy import Policy

    world = standard_world()
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="send_email", effect=EffectClass.IRREVERSIBLE, fn=world.send_email)
    )
    call = ToolCall("send_email", {"to": "a@b.c", "subject": "s", "body": "b"})
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)

    default_off = analyse(
        branch, call, registry.get("send_email"), Policy(stage_irreversible=False)
    )
    turned_on = analyse(branch, call, registry.get("send_email"), Policy(stage_irreversible=True))

    return AttackResult(
        id="7.9",
        name="staging an irreversible effect",
        claim="off by default; with it on, an effect with no undo is released by machinery",
        measured={
            "barrier_by_default": 1 if default_off is Hazard.IRREVERSIBLE_ON_PATH else 0,
            "staged_when_enabled": 1 if turned_on is None else 0,
        },
        defeated_runtime=turned_on is None,
    )


async def attack_77_drafter_poisoning() -> AttackResult:
    """7.7 — train the pattern index on traces whose "strong chain" ends in a write.

    An adversary who can influence what the index learns can make it confidently predict a
    write that the model never asks for. This runs the poisoned drafter inside the real
    scheduler, so every number below is something the runtime did: each guess forked a branch,
    staged the charge in the store buffer, and was squashed when the model's real block
    arrived or the turn ended; the alpha gate closed once its window filled with misses; and
    the remaining turns ran with no speculation at all.

    A pattern-index guess costs no model tokens, so ``wasted_tokens`` is genuinely zero here
    and is reported as zero rather than as an invented per-guess charge. What a poisoned
    index costs is the branches, the stagings that are thrown away, and the window's worth of
    misses before the gate closes. What it cannot cost is an effect, and that is the assertion.

    Two properties of the fixture worth stating, because the numbers depend on them. The first
    guess of each turn is the poisoned chain proper: history ``[quote]`` matches the order-1
    context and ``charge_card`` is ranked at 1.0. The second is an order-0 back-off, where
    ``charge_card`` and ``quote`` are tied at 0.5 and the tie is broken by signature string --
    so it happens to be a charge because ``charge_card`` sorts before ``quote``. Both stage and
    both are discarded, which is what the row counts. And the model waits for each guess to
    reach the buffer before moving on; if that wait ever expires while speculation is still
    enabled the attack raises rather than reporting a smaller number, because a count that
    quietly shrinks under load is worse than no count.
    """
    import tempfile
    import time
    from dataclasses import dataclass as _dataclass

    from specunode.buffer.dispatcher import Dispatcher
    from specunode.buffer.store_buffer import StoreBuffer
    from specunode.core.decision import Decision
    from specunode.core.graph import END, AdapterCapabilities, NextNode, NodeRef, RunSession
    from specunode.core.model import JournaledModel, Message, RequestEnvelope, TextBlock
    from specunode.core.policy import Budget, Policy
    from specunode.core.scheduler import Scheduler
    from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
    from specunode.journal.journal import close_all_writers
    from specunode.testing.models import ScriptedModel, tool_turn

    turns = 6
    window = 4

    class _TurnsGraph:
        """A node that hands the runtime one turn per visit and stops after ``turns``."""

        def capabilities(self) -> AdapterCapabilities:
            return AdapterCapabilities(drives_itself=False, framework="plain")

        def nodes(self) -> list[NodeRef]:
            return [NodeRef(name="agent")]

        def decision_kind(self, node: NodeRef) -> str:
            return "tool_call"

        def next(self, state: object) -> NextNode:
            done = isinstance(state, dict) and int(state.get("turns", 0)) >= turns
            return END if done else NodeRef(name="agent")

        async def run_node(self, node: NodeRef, session: RunSession) -> Decision:
            assert session.call_turn is not None
            await session.call_turn(
                RequestEnvelope(
                    model="scripted",
                    messages=(Message(role="user", content=(TextBlock(text="decide"),)),),
                    max_tokens=128,
                    stream=True,
                )
            )
            session.state["turns"] = int(session.state.get("turns", 0)) + 1
            return ToolCall("", {})

        async def drive(self, session: RunSession, inputs: object) -> object:
            raise NotImplementedError

    @_dataclass
    class _LetsTheGuessStage(ScriptedModel):
        """A scripted model that lets each guess reach the store buffer before moving on.

        "Staged a charge and threw it away" is only countable if the charge reached
        ``StoreBuffer.stage`` before the event that squashes it. ``bench/real_arm.py`` explains
        why a fixed delay makes that likely and a committed number needs it certain: this waits
        on the condition before every event after the first block, and stops waiting once the
        gate has closed and nothing will be staged again.
        """

        buffer: StoreBuffer | None = None
        budget: Budget | None = None
        timeouts: int = 0

        async def _settle(self) -> None:
            if self.buffer is None or self.budget is None:
                return
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not self.budget.may_speculate():
                    # The gate is shut; nothing will stage again, and waiting is not a failure.
                    return
                if self.buffer.has_staged():
                    return
                await asyncio.sleep(0)
            self.timeouts += 1

        async def stream(self, envelope):  # type: ignore[no-untyped-def]
            first = True
            async for event in super().stream(envelope):
                if not first:
                    await self._settle()
                first = False
                yield event

    # The adversarial "strong chain": a quote that carries an amount, then a charge that
    # consumes it. The amount has to be reachable from history or the drafter declines to
    # offer the prediction at all -- an unfillable guess would squash every time and its only
    # effect would be the reads its branch paid for.
    poison = [
        ToolCall("quote", {"customer_id": "cus-1", "amount": 999.0}),
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 999.0}),
    ]
    index = PatternIndex(order=2)
    index.train([poison] * 50)

    async def quote(customer_id: str, amount: float) -> JsonValue:
        return {"customer_id": customer_id, "amount": amount, "currency": "usd"}

    world = standard_world()
    registry = ToolRegistry()
    registry.register(ToolSpec(name="quote", effect=EffectClass.READ, fn=quote))
    registry.register(
        ToolSpec(
            name="lookup_customer", effect=EffectClass.READ, fn=world.lookup_customer, witness=True
        )
    )
    registry.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))

    # What the model honestly does, every turn: a quote, then a lookup. Never a charge.
    honest = (
        ("quote", {"customer_id": "cus-2", "amount": 999.0}),
        ("lookup_customer", {"customer_id": "cus-2"}),
    )
    with tempfile.TemporaryDirectory() as root:
        journal = Journal(Path(root) / "attack-77.db")
        buffer = StoreBuffer(journal=journal, run_id="")
        model = _LetsTheGuessStage(
            turns=[tool_turn(*honest, turn=i) for i in range(turns)], buffer=buffer
        )
        scheduler = Scheduler(
            graph=_TurnsGraph(),  # type: ignore[arg-type]
            registry=registry,
            journal=journal,
            buffer=buffer,
            dispatcher=Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5),
            target=JournaledModel(model, journal, provider="scripted"),
            policy=Policy(alpha_window=window, alpha_floor=0.5),
            predictor=PatternDrafter(index=index),
        )
        model.budget = scheduler.budget
        run_id = new_ulid()
        outcome = await scheduler.run(run_id, {})
        if not outcome.ok:
            raise RuntimeError(f"the run failed: {outcome.error}")
        if model.timeouts:
            # An error row, never a quietly smaller number. The staging count is only
            # meaningful if every guess reached the buffer before the block that squashes it.
            raise RuntimeError(
                f"{model.timeouts} guess(es) did not reach the store buffer within the "
                "fixture's deadline; the staging count would be an artefact of load"
            )
        disabled = [
            e
            for e in journal.read(run_id, kinds=["policy_event"])
            if e.payload.get("event") == "speculation_disabled"
        ]
        # A guess ends squashed, stalled, or confirmed-and-adopted by the branch that made it;
        # the canonical branch of each node visit is confirmed by its own turn and then retired.
        # ``counters.branches_forked`` counts both kinds of fork, so the guesses are counted
        # from how they were resolved, and a confirmation counts only if it names an adopter.
        resolved = [dict(e.payload) for e in journal.read(run_id, kinds=["branch_resolved"])]
        # Charges specifically, not effects in general: ``counters.effects_discarded`` counts
        # any tool, so on a fixture with a second write the row's name would stop matching
        # what it held.
        staged = {
            str(e.payload["effect_id"])
            for e in journal.read(run_id, kinds=["effect_staged"])
            if e.payload.get("tool") == "charge_card"
        }
        # ``effect_discarded`` carries ``effect_ids``, a list -- one entry per discarding
        # branch, not per effect.
        charges_discarded = sum(
            1
            for e in journal.read(run_id, kinds=["effect_discarded"])
            for effect_id in (e.payload.get("effect_ids") or ())
            if str(effect_id) in staged
        )
        close_all_writers()

    counters = scheduler.counters
    guesses = {
        "confirmed": sum(
            1 for r in resolved if r.get("status") == "confirmed" and "adopted_by" in r
        ),
        "squashed": sum(1 for r in resolved if r.get("status") == "squashed"),
        "stalled": sum(1 for r in resolved if r.get("status") == "stalled"),
    }
    leaked = world.mutations_by("charge_card")
    return AttackResult(
        id="7.7",
        name="drafter poisoning",
        claim=(
            "a poisoned index forks, stages and squashes until the alpha window fills with "
            "misses and the gate closes; it cannot put an effect in the world"
        ),
        measured={
            "turns_run": turns,
            "alpha_window": window,
            # confirmed + squashed + stalled == predictions_made, so the row reconciles from
            # its own fields. It used to print two of the three.
            "predictions_made": sum(guesses.values()),
            "confirmed": guesses["confirmed"],
            "squashed": guesses["squashed"],
            "stalled": guesses["stalled"],
            "charges_staged_then_discarded": charges_discarded,
            "speculation_disabled_by_alpha_gate": (
                1 if counters.speculation_disabled_reason == "alpha_below_floor" else 0
            ),
            "gate_closures_journaled": len(disabled),
            "wasted_tokens": scheduler.budget.wasted_tokens,
            "leaked_effects": len(leaked),
        },
        defeated_runtime=bool(leaked),
    )


async def attack_78_replay_under_model_drift() -> AttackResult:
    """7.8 — replay a journal after changing the system prompt, and after changing the tools.

    Expected: divergence at the first step in both cases, rather than a replay that quietly
    continues down a trajectory the recorded run never took.
    """
    import tempfile

    from specunode.core.model import (
        CallScope,
        JournaledModel,
        Message,
        RequestEnvelope,
        TextBlock,
        ToolDef,
        scoped,
    )
    from specunode.journal.replay import ReplayDivergence, ReplayModel
    from specunode.testing.models import ScriptedModel, tool_turn

    run_id = new_ulid()
    journal = Journal(Path(tempfile.mkdtemp()) / "drift.db")
    base = RequestEnvelope(
        model="scripted",
        system=(TextBlock(text="You are a support agent."),),
        messages=(Message(role="user", content=(TextBlock(text="help"),)),),
        tools=(ToolDef(name="lookup_customer", description="look up", input_schema={}),),
        max_tokens=128,
    )
    journaled = JournaledModel(ScriptedModel(turns=[tool_turn(("a", {}))]), journal)
    with scoped(CallScope(run_id=run_id, branch_id="br-1", step=0)):
        await journaled.complete(base)

    from dataclasses import replace as dc_replace

    outcomes: dict[str, int] = {}
    for label, changed in (
        ("system_prompt", dc_replace(base, system=(TextBlock(text="You are a support agent!"),))),
        ("tool_list", dc_replace(base, tools=(*base.tools, ToolDef("x", "x", {})))),
    ):
        replay = ReplayModel(journal=journal, run_id=run_id)
        step_of_first_divergence = -1
        try:
            with scoped(CallScope(run_id=run_id, step=0)):
                await replay.complete(changed)
        except ReplayDivergence as exc:
            step_of_first_divergence = exc.step
        outcomes[f"first_divergence_step_after_{label}_change"] = step_of_first_divergence

    caught_both = all(step == 0 for step in outcomes.values())
    return AttackResult(
        id="7.8",
        name="replay under model drift",
        claim="a changed prompt or tool list diverges at the first step, never silently continues",
        measured={**outcomes, "both_caught_at_step_0": 1 if caught_both else 0},
        defeated_runtime=not caught_both,
    )


ATTACKS: Sequence[tuple[str, Callable[[], Awaitable[AttackResult]]]] = (
    ("7.1", attack_71_misdeclared_tool),
    ("7.2", attack_72_read_with_side_effects),
    ("7.3", attack_73_stale_reads_under_contention),
    ("7.4", attack_74_duplicate_delivery),
    ("7.5", attack_75_return_value_laundering),
    ("7.6", attack_76_prompt_injected_tool_call),
    ("7.7", attack_77_drafter_poisoning),
    ("7.8", attack_78_replay_under_model_drift),
    ("7.9", attack_79_staging_an_irreversible_effect),
    ("7.10", attack_710_async_side_effect_behind_a_read),
)


async def run_all() -> list[AttackResult]:
    """Run every strategy. A strategy that raises becomes an error row, never a dropped one."""
    results: list[AttackResult] = []
    for attack_id, fn in ATTACKS:
        try:
            results.append(await fn())
        except Exception as exc:  # an attack that cannot run is reported as such
            results.append(
                AttackResult(
                    id=attack_id,
                    name=fn.__name__,
                    claim="(strategy failed to run)",
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-400:]}",
                )
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpecuNode adversarial suite")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = asyncio.run(run_all())
    payload = {"attacks": [asdict(r) for r in results]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("WHAT BEATS IT")
    print()
    for result in results:
        marker = "ERROR " if result.error else ("beats " if result.defeated_runtime else "held ")
        print(f"  [{marker}] {result.id:<5} {result.name}")
        print(f"            {result.claim}")
        if result.error:
            print(f"            error: {result.error.splitlines()[0]}")
        else:
            for key, value in result.measured.items():
                print(f"            {key}: {value}")
        print()
    errors = sum(1 for r in results if r.error)
    beaten = sum(1 for r in results if r.defeated_runtime)
    print(f"  {len(results)} strategies, {beaten} beat the runtime, {errors} failed to run.")
    print()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
