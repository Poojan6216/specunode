"""Five minutes with SpecuNode: charge a card, die at the worst moment, resume, charge nobody twice.

    python examples/quickstart.py

The process is killed after the payments API took the charge and before the reply came back --
the one moment where a retry charges the customer twice and a checkpoint cannot tell. The run is
resumed from its journal. The runtime knows the charge may already be out, because it claimed it
under a deterministic key before sending it, so it asks the payments API about that key instead
of guessing -- and carries on with the charge that was actually made.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import specunode


class ProcessDied(BaseException):
    """Pulling the plug: nothing after it runs, and nothing is cleaned up."""


class Payments:
    """A stand-in payments API that, like a real one, remembers the request key of each charge."""

    def __init__(self) -> None:
        self.charges: dict[str, dict[str, Any]] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.by_key: dict[str, str] = {}

    async def charge(self, customer_id: str, amount: float, request_key: str) -> str:
        charge_id = f"ch_{len(self.charges) + 1}"
        self.charges[charge_id] = {"customer": customer_id, "amount": amount}
        self.by_key[request_key] = charge_id
        return charge_id

    async def send_receipt(self, customer_id: str, charge_id: str, request_key: str) -> str:
        receipt_id = f"rc_{len(self.receipts) + 1}"
        self.receipts[receipt_id] = {"customer": customer_id, "charge": charge_id}
        self.by_key[request_key] = receipt_id
        return receipt_id


payments = Payments()
pull_the_plug = {"after_the_charge": True}


async def charge_was_taken(key: str, args: Mapping[str, Any]) -> dict[str, str] | None:
    """Asked after a crash: did the charge sent under this key go through?"""
    charge_id = payments.by_key.get(key)
    print(f"  asked the payments API about request {key[:12]}...: {charge_id or 'never arrived'}")
    return None if charge_id is None else {"charge_id": charge_id}


async def receipt_was_sent(key: str, args: Mapping[str, Any]) -> dict[str, str] | None:
    receipt_id = payments.by_key.get(key)
    return None if receipt_id is None else {"receipt_id": receipt_id}


@specunode.tool(effect="write", reconcile=charge_was_taken)
async def charge_card(customer_id: str, amount: float) -> dict[str, str]:
    key = specunode.current_idempotency_key()
    charge_id = await payments.charge(customer_id, amount, request_key=key)
    if pull_the_plug["after_the_charge"]:
        pull_the_plug["after_the_charge"] = False
        raise ProcessDied("killed after the charge went through, before its reply came back")
    return {"charge_id": charge_id}


@specunode.tool(effect="write", reconcile=receipt_was_sent)
async def send_receipt(customer_id: str, charge_id: str) -> dict[str, str]:
    key = specunode.current_idempotency_key()
    receipt_id = await payments.send_receipt(customer_id, charge_id, request_key=key)
    return {"receipt_id": receipt_id}


@specunode.node()
async def bill(session: specunode.RunSession) -> specunode.Decision:
    customer = str(session.state["customer_id"])
    charge = await session.call_tool("charge_card", {"customer_id": customer, "amount": 25.0})
    assert isinstance(charge, Mapping)
    await session.call_tool(
        "send_receipt", {"customer_id": customer, "charge_id": str(charge["charge_id"])}
    )
    session.state["billed"] = True
    return specunode.FreeText.of(f"billed {customer}")


def route(state: Mapping[str, Any]) -> str | None:
    return None if state.get("billed") else "bill"


async def main() -> dict[str, int]:
    with tempfile.TemporaryDirectory() as directory:
        runtime = specunode.Runtime(
            specunode.graph([bill], route),
            tools=[charge_card, send_receipt],
            journal=Path(directory) / "journal.db",
        )
        run_id = "quickstart"
        print("billing cus-1 ...")
        try:
            await runtime.run({"customer_id": "cus-1"}, run_id=run_id)
        except ProcessDied as died:
            print(f"  the process died: {died}")
        print("resuming from the journal ...")
        result = await runtime.resume(run_id)
        print(f"  finished: {result.ok}")
        charges, receipts = len(payments.charges), len(payments.receipts)
        print(f"charges made: {charges}, receipts sent: {receipts}")
        print("(a retry that did not know the charge was out would have made a second one)")
        return {"charges": charges, "receipts": receipts, "ok": int(result.ok)}


if __name__ == "__main__":
    asyncio.run(main())
