"""A write whose reply was lost is never sent again on a guess -- with or without a crash.

The dispatcher used to retry any failure, whatever the tool declared. A payment gateway that
took a charge and then timed out on the reply was charged again on the next attempt: with the
default settings and no crash at all. A retry is safe only when the failed attempt demonstrably
sent nothing, or when the tool declared a repeat harmless. Otherwise the runtime asks the
upstream (the tool's ``reconcile``), or stops for a human, and a resume does not send it again
until someone who has checked says what happened (``specunode resolve``). Found by the eighth
review.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from typer.testing import CliRunner

from specunode.buffer.dispatcher import Dispatcher, ToolDispatchError
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.cli import app
from specunode.core.decision import Decision, FreeText
from specunode.core.graph import RunSession
from specunode.core.policy import Policy
from specunode.core.scheduler import RunResult, Scheduler
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal
from specunode.journal.ledger import build_ledger
from specunode.testing.models import ScriptedModel

RUN = "01LOSTREPLIESAAAAAAAAAAAAA"


class Gateway:
    """A payment API. ``replies`` scripts each call: "ok", "lost" (took it, then timed out),
    "refused" (turned it away before taking it), or "down" (never reached)."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.taken: list[str] = []

    async def charge(self, customer_id: str, amount: float) -> JsonValue:
        reply = self.replies.pop(0) if self.replies else "ok"
        if reply == "down":
            raise ToolDispatchError("connection refused", sent="no")
        if reply == "refused":
            raise ToolDispatchError("declined before taking it", sent="no")
        self.taken.append(f"{customer_id} {amount}")
        if reply == "lost":
            raise ToolDispatchError("gateway timeout after the charge", sent="maybe")
        if reply == "raised":
            raise TimeoutError("read timed out")
        return {"charge_id": f"ch_{len(self.taken)}"}

    def charged(self, key: str) -> JsonValue | None:
        return {"charge_id": "ch_1"} if self.taken else None


def scheduler(
    journal: Journal, gateway: Gateway, *, idempotent: bool, reconcile: bool
) -> Scheduler:
    reconciler: Callable[[str, Mapping[str, JsonValue]], object] | None = None
    if reconcile:

        async def reconciler(key: str, args: Mapping[str, JsonValue]) -> JsonValue | None:  # type: ignore[no-redef]
            return gateway.charged(key)

    @tool(effect="write", idempotent=idempotent, reconcile=reconciler)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        return await gateway.charge(customer_id, amount)

    @node(name="bill")
    async def bill(session: RunSession) -> Decision:
        await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 25.0})
        session.state["billed"] = True
        return FreeText.of("billed")

    def route(state: Mapping[str, JsonValue]) -> str | None:
        return None if state.get("billed") else "bill"

    registry = registry_of([charge_card])
    return Scheduler(
        graph=PlainAdapter.of([bill], route),  # type: ignore[arg-type]
        registry=registry,
        journal=journal,
        buffer=StoreBuffer(journal=journal, run_id=""),
        dispatcher=Dispatcher(registry=registry, max_attempts=3, base_delay_ms=0.1),
        target=ScriptedModel(turns=[]),  # type: ignore[arg-type]
        policy=Policy(speculation=False),
    )


async def run(
    tmp_path: Path, gateway: Gateway, *, idempotent: bool = False, reconcile: bool = False
) -> tuple[RunResult, Journal]:
    journal = Journal(tmp_path / "journal.db")
    result = await scheduler(journal, gateway, idempotent=idempotent, reconcile=reconcile).run(
        RUN, {}
    )
    return result, journal


def dead_letters(journal: Journal) -> list[Mapping[str, JsonValue]]:
    return [entry.payload for entry in journal.read(RUN, kinds=["effect_dead_lettered"])]


async def test_a_charge_whose_reply_was_lost_is_not_sent_again(tmp_path: Path) -> None:
    gateway = Gateway("lost")
    result, journal = await run(tmp_path, gateway)
    assert not result.ok
    assert gateway.taken == ["cus-1 25.0"], "the charge was sent again after its reply was lost"
    assert [letter["sent"] for letter in dead_letters(journal)] == ["maybe"]


async def test_an_exception_the_tool_did_not_classify_is_a_lost_reply_too(tmp_path: Path) -> None:
    gateway = Gateway("raised")
    result, _journal = await run(tmp_path, gateway)
    assert not result.ok
    assert gateway.taken == ["cus-1 25.0"]


async def test_a_failure_before_anything_left_is_retried(tmp_path: Path) -> None:
    gateway = Gateway("down", "ok")
    result, _journal = await run(tmp_path, gateway)
    assert result.ok, result.error
    assert gateway.taken == ["cus-1 25.0"]


async def test_an_idempotent_write_is_redelivered_after_a_lost_reply(tmp_path: Path) -> None:
    """Declaring a repeat harmless is what permits it: the upstream takes the same key twice."""
    gateway = Gateway("lost", "ok")
    result, _journal = await run(tmp_path, gateway, idempotent=True)
    assert result.ok, result.error
    assert gateway.taken == ["cus-1 25.0", "cus-1 25.0"]


async def test_one_attempt_that_may_have_landed_makes_the_dead_letter_maybe(
    tmp_path: Path,
) -> None:
    """The last attempt was refused before it took anything; the first may have landed. The
    dead letter used to record only the last, and a resume then asked the model again."""
    gateway = Gateway("lost", "refused", "refused")
    result, journal = await run(tmp_path, gateway, idempotent=True)
    assert not result.ok
    assert [letter["sent"] for letter in dead_letters(journal)] == ["maybe"]


async def test_a_lost_reply_is_asked_about_when_the_tool_can_say(tmp_path: Path) -> None:
    gateway = Gateway("lost")
    result, journal = await run(tmp_path, gateway, reconcile=True)
    assert result.ok, result.error
    assert gateway.taken == ["cus-1 25.0"]
    [sent] = [entry.payload for entry in journal.read(RUN, kinds=["effect_dispatched"])]
    assert sent["reconciled"] is True and sent["ack"] == {"charge_id": "ch_1"}


async def test_a_lost_request_the_upstream_never_saw_is_sent_once_more(tmp_path: Path) -> None:
    """The reply was lost, but so was the request: the upstream has no record, so it is sent."""

    class LostRequest(Gateway):
        async def charge(self, customer_id: str, amount: float) -> JsonValue:
            if not self.replies:
                return await super().charge(customer_id, amount)
            self.replies.pop(0)
            raise ToolDispatchError("timeout; the request may or may not have arrived")

    gateway = LostRequest("lost")
    result, _journal = await run(tmp_path, gateway, reconcile=True)
    assert result.ok, result.error
    assert gateway.taken == ["cus-1 25.0"]


async def test_an_operator_settles_a_dead_letter_and_the_resume_goes_on(tmp_path: Path) -> None:
    """A resume does not send a charge that may have gone out. Someone checks the gateway, says
    it landed, and the resume skips it -- the ledger then shows it sent, once."""
    gateway = Gateway("lost")
    result, journal = await run(tmp_path, gateway)
    assert not result.ok

    again = await scheduler(journal, gateway, idempotent=False, reconcile=False).resume(RUN)
    assert not again.ok, "a resume sent, or skipped, a charge nobody had checked"
    assert gateway.taken == ["cus-1 25.0"]

    [row] = build_ledger(journal, RUN).rows[:1]
    cli = CliRunner().invoke(
        app,
        [
            "resolve",
            RUN,
            row.nkey[:12],
            "--landed",
            "--ack",
            '{"charge_id": "ch_1"}',
            "--journal",
            str(tmp_path / "journal.db"),
        ],
    )
    assert cli.exit_code == 0, cli.output
    finished = await scheduler(journal, gateway, idempotent=False, reconcile=False).resume(RUN)
    assert finished.ok, finished.error
    assert gateway.taken == ["cus-1 25.0"]
    assert {row.status for row in build_ledger(journal, RUN).rows} == {"DISPATCHED"}


async def test_an_operator_who_finds_nothing_landed_has_the_resume_send_it(
    tmp_path: Path,
) -> None:
    class LostRequest(Gateway):
        async def charge(self, customer_id: str, amount: float) -> JsonValue:
            if self.replies:
                self.replies.pop(0)
                raise ToolDispatchError("timeout; the request may or may not have arrived")
            return await super().charge(customer_id, amount)

    gateway = LostRequest("lost")
    result, journal = await run(tmp_path, gateway)
    assert not result.ok and gateway.taken == []

    [row] = build_ledger(journal, RUN).rows
    cli = CliRunner().invoke(
        app, ["resolve", RUN, row.nkey, "--not-sent", "--journal", str(tmp_path / "journal.db")]
    )
    assert cli.exit_code == 0, cli.output
    finished = await scheduler(journal, gateway, idempotent=False, reconcile=False).resume(RUN)
    assert finished.ok, finished.error
    assert gateway.taken == ["cus-1 25.0"]
