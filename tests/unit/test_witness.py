"""Read validation at retirement (spec task 3.5, lattice rule E3).

E3 sits before E4 in the status lattice, and that ordering carries most of the integrity: a
branch whose reads went stale between speculating and being confirmed must not retire *just
because the model's decision matched*. The decision being right does not make the data the
branch computed from right.

The other half of this module is an honesty constraint rather than a safety one. A read with no
witness cannot be checked, and is reported as unwitnessed rather than counted fresh -- the
fraction of stale reads that were undetectable is the number attack 7.3 exists to publish, and
rounding it into "fresh" would erase it.
"""

from __future__ import annotations

from specunode.canonical import chash
from specunode.core.branch import Branch, BranchStatus, ReadRecord
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.testing.world import World, standard_world
from specunode.verify.witness import validate_reads


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
    return registry


async def read_record(world: World, tool: str, args: dict[str, object]) -> ReadRecord:
    value = await getattr(world, tool)(**args)
    witness = value.get("witness") if isinstance(value, dict) else None
    return ReadRecord(
        tool=tool,
        args=args,  # type: ignore[arg-type]
        args_hash=chash(args),  # type: ignore[arg-type]
        result_hash=chash(value),
        witness=witness,
        at_step=1,
        issued_while_speculative=True,
    )


async def test_a_read_that_did_not_change_is_fresh(tmp_path: object) -> None:
    world = standard_world()
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    branch.read_set.append(
        await read_record(world, "get_pipeline_status", {"pipeline_id": "etl-1"})
    )

    result = await validate_reads(branch, registry_for(world))
    assert result.fresh == 1 and result.stale == 0
    assert not result.any_stale
    assert result.probes == 1, "re-checking costs a real upstream call, and it is counted"


async def test_a_read_whose_row_moved_is_stale(tmp_path: object) -> None:
    """The case E3 exists for: the model guessed right, the world moved anyway."""
    world = standard_world()
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    branch.read_set.append(
        await read_record(world, "get_pipeline_status", {"pipeline_id": "etl-1"})
    )

    # A competing actor, attributed to <external> so it never looks like a leak.
    await world.restart_job(job_id="etl-1")

    result = await validate_reads(branch, registry_for(world))
    assert result.stale == 1 and result.fresh == 0
    assert result.any_stale


async def test_a_read_without_a_witness_is_reported_unwitnessed_not_fresh(
    tmp_path: object,
) -> None:
    """The honest number. Counting it fresh would erase the one attack 7.3 publishes."""
    world = standard_world()
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    branch.read_set.append(await read_record(world, "fetch_runbook", {"section": "restart"}))

    result = await validate_reads(branch, registry_for(world))
    assert result.unwitnessed == 1
    assert result.fresh == 0
    assert result.probes == 0, "there is nothing to re-fetch, so nothing was spent"


async def test_reads_a_confirmed_branch_issued_are_not_revalidated(tmp_path: object) -> None:
    """They happened after the decision was durable, so there is no speculation to invalidate.

    Revalidating them turns a workload with a competing writer into a livelock on the
    sequential arm, where validation means nothing at all.
    """
    world = standard_world()
    branch = Branch(id="br-1", status=BranchStatus.CONFIRMED)
    record = await read_record(world, "get_pipeline_status", {"pipeline_id": "etl-1"})
    branch.read_set.append(
        ReadRecord(
            tool=record.tool,
            args=record.args,
            args_hash=record.args_hash,
            result_hash=record.result_hash,
            witness=record.witness,
            at_step=record.at_step,
            issued_while_speculative=False,
        )
    )
    await world.restart_job(job_id="etl-1")  # would be stale if it were checked

    result = await validate_reads(branch, registry_for(world))
    assert result.total == 0
    assert result.probes == 0


async def test_an_undeclared_tool_is_unreadable_rather_than_assumed_fresh(
    tmp_path: object,
) -> None:
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    branch.read_set.append(
        ReadRecord(
            tool="nobody_declared_this",
            args={},
            args_hash=chash({}),
            result_hash="h",
            witness="v1",
            at_step=1,
        )
    )
    result = await validate_reads(branch, ToolRegistry())
    assert result.fresh == 0 and result.stale == 0
    assert result.verdicts[0].verdict == "unreadable"


async def test_the_payload_carries_the_breakdown_the_ledger_prints(tmp_path: object) -> None:
    world = standard_world()
    branch = Branch(id="br-1", status=BranchStatus.SPECULATIVE)
    branch.read_set.append(
        await read_record(world, "get_pipeline_status", {"pipeline_id": "etl-1"})
    )
    branch.read_set.append(await read_record(world, "fetch_runbook", {"section": "restart"}))

    payload = (await validate_reads(branch, registry_for(world))).payload()
    assert payload["total"] == 2
    assert payload["fresh"] == 1
    assert payload["unwitnessed"] == 1
