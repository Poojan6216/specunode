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

from specunode.canonical import canonical
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
    write that the model never asks for. Every one of those is squashed, so nothing reaches the
    world; what it costs is wasted tokens, wasted upstream reads and stalls. The measurement is
    that cost, and the assertion is that leaks stay at zero.
    """
    from specunode.core.policy import Budget, Policy
    from specunode.drafters.base import DraftContext
    from specunode.drafters.t1_pattern import PatternDrafter, PatternIndex
    from specunode.verify.gate import resolve_decision

    # The adversarial "strong chain": a lookup that carries an amount, followed by a charge
    # that consumes it. The amount has to be reachable from history or the drafter declines to
    # offer the prediction at all -- an unfillable guess would squash every time and its only
    # effect would be the reads its branch paid for.
    poison = [
        ToolCall("lookup_customer", {"customer_id": "cus-1", "amount": 999.0}),
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 999.0}),
    ]
    index = PatternIndex(order=2)
    index.train([poison] * 50)
    drafter = PatternDrafter(index=index)

    world = standard_world()
    registry = ToolRegistry()
    registry.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))
    registry.register(
        ToolSpec(
            name="lookup_customer", effect=EffectClass.READ, fn=world.lookup_customer, witness=True
        )
    )

    budget = Budget(policy=Policy(alpha_window=4, alpha_floor=0.5))
    honest = ToolCall("lookup_customer", {"customer_id": "cus-2"})
    squashed = 0
    predicted = 0
    for _ in range(8):
        candidates = await drafter.predict(
            DraftContext(
                run_id="r",
                branch_id="b",
                step_index=1,
                node_id="n",
                history=(poison[0],),
                known_tools=registry.names(),
            )
        )
        if not candidates:
            break
        predicted += 1
        guess = candidates[0].decision
        if resolve_decision(guess, honest) is BranchStatus.SQUASHED:
            squashed += 1
            budget.record_resolution(tier=1, confirmed=False, tokens=250)

    leaked = world.mutations_by("charge_card")
    return AttackResult(
        id="7.7",
        name="drafter poisoning",
        claim="a poisoned index wastes tokens and stalls; it cannot put an effect in the world",
        measured={
            "predictions_made": predicted,
            "squashed": squashed,
            "wasted_tokens": budget.wasted_tokens,
            "speculation_disabled_by_alpha_gate": 1 if not budget.may_speculate() else 0,
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
