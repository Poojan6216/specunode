"""Every bundled adapter, held to the contract in docs/adapters.md (spec task 8.4).

An adapter that quietly fails one of these does not break loudly. A tool that cannot be
cancelled makes a squashed branch keep paying for it; one that is not really idempotent turns
an ambiguous crash window into a second charge; one that reports the wrong ``sent`` bit turns a
safe retry into a duplicate or a duplicate into a dead letter. None of those show up as a
crash, so they are checked here rather than left to be discovered in a chaos run.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from specunode.buffer.dispatcher import Dispatcher, ToolDispatchError
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.testing.faults import Partitioned, Timeout
from specunode.testing.world import TrueEffect, World, standard_world

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


def tool_names() -> list[str]:
    return sorted(standard_world().tools())


def test_every_bundled_tool_is_covered_by_this_suite() -> None:
    assert set(tool_names()) == set(CALLS), "a bundled tool is not held to the adapter contract"


@pytest.mark.parametrize("name", tool_names())
async def test_a_tool_is_cancellable(name: str) -> None:
    """Squashing a branch cancels its task; a tool that blocks the loop cannot be squashed."""
    world = standard_world()
    world.slow("read", 2000)
    world.slow("write", 2000)

    task = asyncio.create_task(world.tools()[name].fn(**CALLS[name]))
    await asyncio.sleep(0.02)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 1.0, f"{name} did not stop when cancelled"


@pytest.mark.parametrize("name", tool_names())
async def test_a_tool_returns_json(name: str) -> None:
    """Results are journaled and hashed, so they must have a canonical form."""
    from specunode.canonical import canonical

    world = standard_world()
    with world.bind(branch_id="br-1", effect_key="k"):
        value = await world.tools()[name].fn(**CALLS[name])
    canonical(value)  # raises if it has no canonical form


@pytest.mark.parametrize("name", tool_names())
async def test_a_witnessed_read_returns_a_witness(name: str) -> None:
    world = standard_world()
    tool = world.tools()[name]
    if not tool.witness:
        pytest.skip(f"{name} does not claim to return a witness")
    value = await tool.fn(**CALLS[name])
    assert isinstance(value, dict) and "witness" in value and "value" in value


@pytest.mark.parametrize("name", tool_names())
async def test_a_truly_idempotent_tool_absorbs_a_second_delivery(name: str) -> None:
    """The claim a tool makes with idempotent=True, checked against what it actually does."""
    world = standard_world()
    tool = world.tools()[name]
    if not tool.truly_idempotent or tool.true_effect is not TrueEffect.WRITE:
        pytest.skip(f"{name} does not claim to absorb a repeat delivery")

    with world.bind(branch_id="br-1", effect_key="same-key"):
        await tool.fn(**CALLS[name])
        repeat = await tool.fn(**CALLS[name])
    assert isinstance(repeat, dict) and repeat["applied"] == 0, (
        f"{name} claims idempotence but applied a second delivery"
    )


@pytest.mark.parametrize("name", tool_names())
async def test_a_tool_reports_whether_its_request_left_the_process(name: str) -> None:
    """The one bit that keeps the ambiguous crash window narrow."""
    world = standard_world()
    world.partition()
    with pytest.raises(Partitioned) as caught:
        await world.tools()[name].fn(**CALLS[name])
    assert caught.value.sent == "no", "an unreachable upstream never received the request"

    healed = standard_world()
    healed.timeout(name)
    with pytest.raises(Timeout) as timed_out:
        await healed.tools()[name].fn(**CALLS[name])
    assert timed_out.value.sent == "maybe", "a timeout is ambiguous and must say so"


@pytest.mark.parametrize("name", tool_names())
async def test_a_faulting_tool_raises_something_the_dispatcher_understands(name: str) -> None:
    world = standard_world()
    world.partition()
    registry = ToolRegistry()
    registry.register(ToolSpec(name=name, effect=EffectClass.WRITE, fn=world.tools()[name].fn))
    dispatcher = Dispatcher(registry=registry, max_attempts=2, base_delay_ms=0.5, cap_delay_ms=1.0)
    outcome = await dispatcher.dispatch(
        name,
        dict(CALLS[name]),
        idempotency_key="k",
        branch_id="br-1",  # type: ignore[arg-type]
    )
    assert not outcome.ok
    assert outcome.sent == "no", "the dispatcher must learn that nothing left the process"


def test_the_dispatch_error_contract_defaults_to_the_safe_assumption() -> None:
    """Unknown means 'may already have happened', because the other default charges twice."""
    assert ToolDispatchError("x").sent == "maybe"
    assert ToolDispatchError("x", sent="no").sent == "no"


@pytest.mark.parametrize("name", tool_names())
def test_a_tool_declares_its_true_effect(name: str) -> None:
    """The fake world knows what it really does; the registry knows what was claimed.

    Keeping them separate is what makes attack 7.1 measurable at all -- if the world read its
    semantics from the declaration, a misdeclared tool would look correct by construction.
    """
    tool = standard_world().tools()[name]
    assert tool.true_effect in (TrueEffect.READ, TrueEffect.WRITE, TrueEffect.ASYNC_WRITE)
    if tool.witness:
        assert tool.true_effect is TrueEffect.READ


async def test_a_world_tool_never_mutates_outside_the_funnel() -> None:
    """Restated here so the adapter suite is self-contained for a third-party adapter."""
    world: World = standard_world()
    before = world.snapshot()
    for name in tool_names():
        tool = world.tools()[name]
        if tool.true_effect is TrueEffect.WRITE:
            continue
        with world.bind(branch_id="br-1", effect_key="k"):
            await tool.fn(**CALLS[name])
    assert world.snapshot() == before, "a read path changed state"
