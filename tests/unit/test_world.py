"""The fake world is the instrument every leak claim is measured with (spec task 1.2).

If a write path could change state without appending to ``world.mutations``, the leak test
would pass while effects escaped, and every correctness claim downstream would be worthless.
These tests hold the instrument to its own standard: walk every tool the world exposes, and
assert that state moves only through the recorded funnel.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from specunode.testing.faults import Partitioned, Timeout
from specunode.testing.world import TrueEffect, World, standard_world

#: Arguments that exercise each tool. Every tool the world exposes must appear here, which
#: is asserted below -- a new tool with no coverage would otherwise slip in silently.
CALLS: dict[str, dict[str, object]] = {
    "lookup_customer": {"customer_id": "cus-1"},
    "get_ticket": {"ticket_id": "tkt-x"},
    "get_pipeline_status": {"pipeline_id": "etl-1"},
    "fetch_runbook": {"section": "restart"},
    "search_docs": {"query": "Restart"},
    "create_ticket": {"customer_id": "cus-1", "title": "broken"},
    "update_ticket": {"ticket_id": "tkt-x", "status": "closed"},
    "restart_job": {"job_id": "etl-1"},
    "post_summary": {"channel": "#ops", "text": "done"},
    "charge_card": {"customer_id": "cus-1", "amount": 10.0},
    "send_receipt": {"customer_id": "cus-1", "charge_id": "chg-1"},
    "send_email": {"to": "a@b.c", "subject": "s", "body": "b"},
    "reserve_capacity": {"job_id": "etl-1", "units": 2},
    "release_capacity": {"job_id": "etl-1", "units": 2},
    "enqueue_reindex": {"index": "docs"},
}


def test_every_tool_is_covered_by_these_tests() -> None:
    assert set(standard_world().tools()) == set(CALLS), (
        "a tool was added to the world without a call fixture here, so it would go unchecked"
    )


@pytest.mark.parametrize("name", sorted(CALLS))
async def test_each_write_appends_exactly_one_mutation_and_each_read_appends_none(
    name: str,
) -> None:
    world = standard_world()
    tool = world.tools()[name]
    before_mutations = len(world.mutations)
    before_reads = len(world.reads)

    with world.bind(branch_id="br-1", effect_key=f"key-{name}"):
        await tool.fn(**CALLS[name])

    grew_mutations = len(world.mutations) - before_mutations
    grew_reads = len(world.reads) - before_reads

    if tool.true_effect is TrueEffect.WRITE:
        assert grew_mutations == 1, f"{name} recorded {grew_mutations} mutations, expected 1"
    else:
        # A READ, or an ASYNC_WRITE whose synchronous half looks exactly like one.
        assert grew_mutations == 0, f"{name} mutated the world on what should be a read path"
        assert grew_reads == 1, f"{name} reached upstream without being recorded as a read"


@pytest.mark.parametrize("name", sorted(CALLS))
async def test_no_tool_changes_state_without_going_through_the_mutation_funnel(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break the funnel, and no state may move.

    This is the structural form of the claim: if ``_mutate`` is the only door into the
    tables, disabling it leaves them untouched whichever tool is called. A write tool must
    hit the disabled door; a read tool must not reach it at all and must still change nothing.
    """
    world = standard_world()
    before = world.snapshot()
    hits: list[str] = []

    def refuse(*args: object, **kwargs: object) -> None:
        hits.append(name)
        raise RuntimeError("mutation funnel disabled")

    monkeypatch.setattr(World, "_mutate", refuse)
    writes = world.tools()[name].true_effect is TrueEffect.WRITE
    with world.bind(branch_id="br-1", effect_key="k"):
        if writes:
            with pytest.raises(RuntimeError, match="mutation funnel disabled"):
                await world.tools()[name].fn(**CALLS[name])
        else:
            await world.tools()[name].fn(**CALLS[name])

    assert hits == ([name] if writes else []), (
        f"{name} took the wrong path: write tools must go through the funnel, reads must not"
    )
    assert world.snapshot() == before, f"{name} changed state without recording a mutation"


async def test_mutations_are_attributed_to_the_binding_branch() -> None:
    world = standard_world()
    with world.bind(branch_id="br-7", effect_key="key-a", speculative=True):
        await world.restart_job(job_id="etl-1")
    assert world.mutating_branches() == {"br-7"}
    recorded = world.mutations[-1]
    assert recorded.effect_key == "key-a"
    assert recorded.speculative is True


async def test_unbound_calls_are_attributed_to_external_not_to_a_branch() -> None:
    """Test setup and the hostile third party of attack 7.3 must never look like a leak."""
    world = standard_world()
    await world.restart_job(job_id="etl-1")
    assert world.mutating_branches() == {"<external>"}


async def test_sibling_branches_do_not_share_attribution(  # Hard Rule 6, at the instrument
) -> None:
    world = standard_world()

    async def branch(name: str) -> None:
        with world.bind(branch_id=name, effect_key=f"k-{name}"):
            await asyncio.sleep(0)
            await world.restart_job(job_id="etl-1")

    await asyncio.gather(branch("br-a"), branch("br-b"))
    assert world.mutating_branches() == {"br-a", "br-b"}
    assert {m.effect_key for m in world.mutations} == {"k-br-a", "k-br-b"}


async def test_witness_advances_on_write_and_holds_still_on_read() -> None:
    world = standard_world()
    before = world.witness_of("jobs", "etl-1")
    await world.get_pipeline_status(pipeline_id="etl-1")
    assert world.witness_of("jobs", "etl-1") == before
    with world.bind(branch_id="br-1", effect_key="k"):
        await world.restart_job(job_id="etl-1")
    assert world.witness_of("jobs", "etl-1") != before


async def test_a_witnessed_read_returns_value_and_witness_together() -> None:
    world = standard_world()
    result = await world.get_pipeline_status(pipeline_id="etl-1")
    assert isinstance(result, dict) and set(result) == {"value", "witness"}


async def test_an_unwitnessed_read_returns_a_bare_value() -> None:
    """fetch_runbook has no version to re-check, which is why it is reported as unwitnessed."""
    world = standard_world()
    result = await world.fetch_runbook(section="restart")
    assert isinstance(result, dict) and "witness" not in result


# -- faults -------------------------------------------------------------------------------


async def test_partition_at_fires_on_the_nth_attempt_and_heals() -> None:
    world = standard_world()
    world.partition(at=2)
    with world.bind(branch_id="br-1", effect_key="k1"):
        await world.restart_job(job_id="etl-1")
    with world.bind(branch_id="br-1", effect_key="k2"), pytest.raises(Partitioned):
        await world.restart_job(job_id="etl-2")
    assert len(world.mutations) == 1
    world.heal()
    with world.bind(branch_id="br-1", effect_key="k2"):
        await world.restart_job(job_id="etl-2")
    assert len(world.mutations) == 2


async def test_timeout_is_per_tool() -> None:
    world = standard_world()
    world.timeout("charge_card")
    with pytest.raises(Timeout):
        await world.charge_card(customer_id="cus-1", amount=1.0)
    await world.restart_job(job_id="etl-1")


async def test_duplicate_delivery_lands_twice_and_the_world_says_so() -> None:
    """Attack 7.4: dedupe covers the dispatcher's retries, not the network beyond it."""
    world = standard_world()
    world.duplicate_delivery("charge_card")
    with world.bind(branch_id="br-1", effect_key="k"):
        await world.charge_card(customer_id="cus-1", amount=10.0)
    assert len(world.mutations_by("charge_card")) == 2


async def test_a_truly_idempotent_tool_logs_both_deliveries_but_applies_once() -> None:
    world = standard_world()
    world.duplicate_delivery("restart_job")
    with world.bind(branch_id="br-1", effect_key="same-key"):
        result = await world.restart_job(job_id="etl-1")
    assert len(world.mutations_by("restart_job")) == 2, "both deliveries must be visible"
    assert isinstance(result, dict) and result["applied"] == 1


async def test_slow_reads_are_cancellable_so_a_squash_can_stop_one() -> None:
    """Spec task 3.7: squashing a branch cancels a read that is still in flight."""
    world = standard_world()
    world.slow("read", 2000)
    started = time.monotonic()
    task = asyncio.create_task(world.fetch_runbook(section="restart"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 1.0, "cancellation did not actually interrupt the read"
    assert world.mutations == []


async def test_a_read_that_was_cancelled_still_counts_as_having_reached_upstream() -> None:
    """Attack 7.2: a speculative read costs whatever it costs, squash or no squash."""
    world = standard_world()
    world.slow("read", 2000)
    with world.bind(branch_id="br-doomed", effect_key="", speculative=True):
        task = asyncio.create_task(world.fetch_runbook(section="restart"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(world.reads_from({"br-doomed"})) == 1


async def test_an_async_write_behind_a_read_lands_after_its_branch_is_gone() -> None:
    """Attack 7.10, measured rather than asserted away."""
    world = standard_world()
    with world.bind(branch_id="br-squashed", effect_key="", speculative=True):
        result = await world.enqueue_reindex(index="docs")
    assert isinstance(result, dict) and result["status"] == "queued"
    assert world.mutations == [], "the synchronous half looks exactly like a read"

    assert world.drain_pending_jobs() == 1
    assert world.mutating_branches() == {"br-squashed"}, (
        "the effect is attributed to the branch that caused it, even though that branch is gone"
    )
