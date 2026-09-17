"""The MCP proxy's staging rules (spec tasks 4.1 through 4.4).

The proxy cannot see the model, which is the whole difficulty. It sees tool calls but not the
decision that produced them, so branch resolution has to arrive out of band — and that changes
what it can safely hand back to a client.

The tests below are about that choice, because it is the one place this integration could be
wrong in a way that looks fine. Handing a placeholder to a client that does not understand it
puts the placeholder straight into the next prompt, and the proxy has no visibility of prompts
to stop it. So the mode is decided by what the client *advertised*, not by a default.
"""

from __future__ import annotations

import asyncio

import pytest

from specunode.core.decision import ToolCall
from specunode.core.effects import EffectClass, ToolRegistry, ToolSpec
from specunode.integrations.mcp_proxy import (
    DECISION_CAPABILITY,
    PROXY_TOOLS,
    ClientMode,
    ProxyState,
    mode_for,
)
from specunode.testing.world import standard_world


def registry() -> ToolRegistry:
    world = standard_world()
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="lookup_customer", effect=EffectClass.READ, fn=world.lookup_customer, witness=True
        )
    )
    reg.register(ToolSpec(name="charge_card", effect=EffectClass.WRITE, fn=world.charge_card))
    reg.register(ToolSpec(name="send_email", effect=EffectClass.IRREVERSIBLE, fn=world.send_email))
    return reg


def state(mode: ClientMode = ClientMode.HANDLES) -> ProxyState:
    return ProxyState(registry=registry(), mode=mode)


# -- classification (4.1, 4.2) ----------------------------------------------------------


def test_an_undeclared_tool_is_a_write_through_the_proxy_too() -> None:
    """Hard Rule 2 does not weaken because the call arrived over a socket."""
    assert state().classify("nobody_declared_this").effect is EffectClass.WRITE


def test_a_declared_read_keeps_its_class() -> None:
    assert state().classify("lookup_customer").effect is EffectClass.READ


def test_staging_a_write_sends_nothing_upstream() -> None:
    proxy = state()
    call = proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    assert proxy.staged == [call]
    assert proxy.dispatched == []
    assert call.handle.startswith("$specunode.handle:")


def test_a_handle_capable_client_is_told_the_write_has_not_happened() -> None:
    """The result says so in words, because a client that misreads it will prompt with it."""
    proxy = state()
    call = proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    result = proxy.result_for(call)
    payload = result["_specunode"]
    assert isinstance(payload, dict)
    assert payload["staged"] is True
    assert "has not happened" in str(payload["note"])
    assert "Do not put this handle in a prompt" in str(payload["note"])


# -- the decision boundary (4.3) --------------------------------------------------------


def test_nothing_is_released_until_a_decision_arrives() -> None:
    proxy = state()
    proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    assert proxy.dispatched == []
    assert len(proxy.staged) == 1


def test_a_matching_decision_releases_the_write() -> None:
    proxy = state()
    args = {"customer_id": "cus-1", "amount": 10.0}
    proxy.stage("charge_card", args)
    confirmed, dropped = proxy.retire(ToolCall("charge_card", args))
    assert len(confirmed) == 1 and dropped == []
    assert proxy.staged == []


def test_a_different_decision_discards_the_write_unsent() -> None:
    """The same exact-equality rule as the in-process gate, over a different transport."""
    proxy = state()
    proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    confirmed, dropped = proxy.retire(
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 11.0})
    )
    assert confirmed == [] and len(dropped) == 1
    assert proxy.dispatched == []


def test_an_integer_amount_does_not_confirm_a_float_one_over_mcp_either() -> None:
    proxy = state()
    proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10})
    confirmed, dropped = proxy.retire(
        ToolCall("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    )
    assert confirmed == [] and len(dropped) == 1


async def test_a_blocking_client_waits_rather_than_receiving_a_handle() -> None:
    """Task 4.3's timed check: without a decision, no write ever goes out."""
    proxy = state(ClientMode.BLOCKING)
    proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    assert await proxy.wait_for_decision(deadline_s=0.2) is False
    assert proxy.dispatched == []
    assert len(proxy.staged) == 1


async def test_a_decision_releases_a_waiting_blocking_client() -> None:
    """Deciding and publishing are two steps, and the order is load-bearing.

    ``retire`` resolves the held calls; ``publish`` releases the clients waiting on them. They
    used to be one step, and the forwarding happens *between* them -- so the blocked caller
    woke first, found its call in ``dispatched`` and its result still absent, and every
    confirmed write returned nothing to the client that issued it.
    """
    proxy = state(ClientMode.BLOCKING)
    args = {"customer_id": "cus-1", "amount": 10.0}
    held = proxy.stage("charge_card", args)

    async def decide_shortly() -> None:
        await asyncio.sleep(0.05)
        proxy.retire(ToolCall("charge_card", args))
        # What the real control tool does between these two lines is forward the call and
        # record what came back.
        proxy.record_result(held, [{"type": "text", "text": "charged"}])
        proxy.publish()

    task = asyncio.create_task(decide_shortly())
    assert await proxy.wait_for_decision(deadline_s=2.0) is True
    await task
    assert len(proxy.dispatched) == 1
    assert proxy.result_of(held) is not None, (
        "the client was released before its result was recorded"
    )


async def test_deciding_alone_does_not_release_a_waiting_client() -> None:
    """The half that makes the split worth having: no result yet means no release yet."""
    proxy = state(ClientMode.BLOCKING)
    args = {"customer_id": "cus-1", "amount": 10.0}
    proxy.stage("charge_card", args)

    proxy.retire(ToolCall("charge_card", args))

    assert await proxy.wait_for_decision(deadline_s=0.1) is False, (
        "retire() released the waiter before any result had been recorded"
    )


# -- the mode is asked for, not assumed -------------------------------------------------


def test_a_client_that_advertises_the_capability_gets_handles() -> None:
    assert mode_for({"experimental": {DECISION_CAPABILITY: {}}}) is ClientMode.HANDLES


@pytest.mark.parametrize("capabilities", [None, {}, {"experimental": {}}, {"roots": {}}])
def test_any_other_client_blocks(capabilities: object) -> None:
    """Handing this client a placeholder puts it in the next prompt, unseen by anything."""
    assert mode_for(capabilities) is ClientMode.BLOCKING  # type: ignore[arg-type]


def test_the_proxy_never_claims_to_have_checked_a_prompt() -> None:
    """It cannot see one. Saying 'enforced' here would be a stamp for nobody's work."""
    assert state().context_identity == "unenforced"
    assert state(ClientMode.BLOCKING).context_identity == "unenforced"


# -- the proxy's own tools (4.4) --------------------------------------------------------


def test_the_proxy_exposes_its_six_tools() -> None:
    names = {tool["name"] for tool in PROXY_TOOLS}
    assert names == {
        "specunode.status",
        "specunode.ledger",
        "specunode.stall",
        "specunode.discard",
        "specunode.retire",
        "specunode.replay_check",
    }


def test_status_says_what_is_held_and_that_it_has_not_happened() -> None:
    proxy = state()
    proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    status = proxy.status()
    assert len(status["staged"]) == 1  # type: ignore[arg-type]
    assert "have not happened" in str(status["note"])
    assert status["context_identity"] == "unenforced"


def test_discard_drops_everything_unsent() -> None:
    proxy = state()
    proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    proxy.stage("send_email", {"to": "a@b.c", "subject": "s", "body": "b"})
    assert proxy.discard_all() == 2
    assert proxy.dispatched == [] and len(proxy.discarded) == 2


def test_staged_writes_keep_their_order() -> None:
    """They are dispatched in stage order; a later one may depend on an earlier one landing."""
    proxy = state()
    first = proxy.stage("charge_card", {"customer_id": "cus-1", "amount": 10.0})
    second = proxy.stage("send_email", {"to": "a@b.c", "subject": "s", "body": "b"})
    assert [c.stage_index for c in (first, second)] == [0, 1]
