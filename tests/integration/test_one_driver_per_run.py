"""A run is driven by one process, and one task in it, at a time.

Two resumes of one run at once -- a job queue that delivers the resume twice, a retried HTTP
handler -- each took up the same claim. One sent the charge; the other asked the upstream while
the first request was still in flight, heard "absent", and sent it again. Found by the tenth
review. ``Journal.hold_run`` now holds the run for as long as ``run`` or ``resume`` drives it,
with a lock the operating system drops when the holder dies, so a crashed run can always be
resumed and a running one cannot be resumed twice.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from specunode.buffer.dispatcher import Dispatcher, ToolDispatchError
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText
from specunode.core.graph import RunSession
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler
from specunode.integrations.plain import PlainAdapter, node, registry_of, tool
from specunode.journal.journal import Journal, RunBusy
from specunode.testing.models import ScriptedModel

RUN = "01ONEDRIVERAAAAAAAAAAAAAAA"


def scheduler(journal: Journal, taken: list[str], *, down: list[bool]) -> Scheduler:
    async def charged(key: str, args: Mapping[str, JsonValue]) -> JsonValue | None:
        await asyncio.sleep(0.05)  # the upstream's record lags the request it is serving
        return {"charge_id": "ch_1"} if taken else None

    @tool(effect="write", idempotent=False, reconcile=charged)
    async def charge_card(customer_id: str, amount: float) -> JsonValue:
        if down[0]:
            raise ToolDispatchError("connection refused", sent="no")
        await asyncio.sleep(0.1)  # in flight long enough for a second resume to ask about it
        taken.append(f"{customer_id} {amount}")
        return {"charge_id": f"ch_{len(taken)}"}

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
        dispatcher=Dispatcher(registry=registry, max_attempts=1, base_delay_ms=0.1),
        target=ScriptedModel(turns=[]),  # type: ignore[arg-type]
        policy=Policy(speculation=False),
    )


async def test_two_resumes_of_one_run_at_once_charge_once(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    taken: list[str] = []
    down = [True]
    first = await scheduler(journal, taken, down=down).run(RUN, {})
    assert not first.ok and taken == []  # the gateway was down: a dead letter that never left

    down[0] = False
    both = await asyncio.gather(
        scheduler(journal, taken, down=down).resume(RUN),
        scheduler(journal, taken, down=down).resume(RUN),
        return_exceptions=True,
    )
    assert taken == ["cus-1 25.0"], "two resumes of one run charged twice"
    refused = [outcome for outcome in both if isinstance(outcome, RunBusy)]
    finished = [outcome for outcome in both if not isinstance(outcome, BaseException)]
    assert len(refused) == 1 and len(finished) == 1 and finished[0].ok


def test_a_run_held_by_another_process_cannot_be_driven_here(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from specunode.journal.journal import Journal\n"
            f"with Journal({str(path)!r}).hold_run({RUN!r}):\n"
            "    print('held', flush=True)\n"
            "    time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(RunBusy, match="another process"), Journal(path).hold_run(RUN):
            pass
    finally:
        holder.kill()
        holder.wait()
    # Its holder is gone -- killed, as a crash would -- so the run is free again.
    with Journal(path).hold_run(RUN):
        pass
