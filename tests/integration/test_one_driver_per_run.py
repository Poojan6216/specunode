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
import contextlib
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from specunode.buffer.dispatcher import Dispatcher, ToolDispatchError
from specunode.buffer.store_buffer import StoreBuffer
from specunode.canonical import JsonValue
from specunode.core.decision import Decision, FreeText
from specunode.core.graph import RunSession
from specunode.core.policy import Policy
from specunode.core.scheduler import Scheduler, SchedulerError
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


async def test_a_scheduler_drives_one_run(tmp_path: Path) -> None:
    """A Scheduler keeps the run it drives on itself -- its id, its counters, its buffer -- so
    a second run on it took the first over when both were going, and inherited the first's
    counters when it came after. The LangGraph wrapper shared one across every call. A second
    run, at once or later, is now refused. Found by the eleventh review."""
    journal = Journal(tmp_path / "journal.db")
    taken: list[str] = []
    driver = scheduler(journal, taken, down=[False])
    first = asyncio.create_task(driver.run(RUN, {}))
    await asyncio.sleep(0.01)  # the first run's charge is in flight
    with pytest.raises(SchedulerError, match="already driven run"):
        await driver.run("01ONEDRIVERBBBBBBBBBBBBBBB", {})
    assert (await first).ok
    with pytest.raises(SchedulerError, match="already driven run"):
        await driver.resume(RUN)
    assert taken == ["cus-1 25.0"]
    assert driver.counters.effects_dispatched == 1


async def test_a_buffer_serves_one_run_at_a_time(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    taken: list[str] = []
    one = scheduler(journal, taken, down=[False])
    other = scheduler(journal, taken, down=[False])
    other.buffer = one.buffer
    first = asyncio.create_task(one.run(RUN, {}))
    await asyncio.sleep(0.01)
    with pytest.raises(SchedulerError, match="StoreBuffer is in use by run"):
        await other.run("01ONEDRIVERBBBBBBBBBBBBBBB", {})
    assert (await first).ok
    assert taken == ["cus-1 25.0"]


async def test_a_run_is_started_once_and_resumed_after(tmp_path: Path) -> None:
    """Started again from its beginning, a run asks the model everything afresh, and a call
    that comes out different goes out under a key nothing has seen. ``resume`` knows what
    already went out; ``run`` on a run the journal holds is refused and says so."""
    journal = Journal(tmp_path / "journal.db")
    taken: list[str] = []
    down = [True]
    assert not (await scheduler(journal, taken, down=down).run(RUN, {})).ok
    down[0] = False
    with pytest.raises(SchedulerError, match="resume it"):
        await scheduler(journal, taken, down=down).run(RUN, {})
    assert taken == []
    assert (await scheduler(journal, taken, down=down).resume(RUN)).ok
    assert taken == ["cus-1 25.0"]


def hold_elsewhere(path: Path, run_id: str) -> subprocess.Popen[str]:
    """Hold ``run_id`` from another process, through ``path``, until killed."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from specunode.journal.journal import Journal\n"
            f"with Journal({str(path)!r}).hold_run({run_id!r}):\n"
            "    print('held', flush=True)\n"
            "    time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
    return holder


def test_two_paths_to_one_journal_are_one_lock(tmp_path: Path) -> None:
    """A symlink to the journal named a second lock folder beside it, and a run held through
    one path could be driven through the other, in another process or in this one. Found by
    the eleventh review."""
    path = tmp_path / "journal.db"
    Journal(path)
    alias = tmp_path / "alias.db"
    alias.symlink_to(path)
    holder = hold_elsewhere(path, RUN)
    try:
        with pytest.raises(RunBusy, match="another process"), Journal(alias).hold_run(RUN):
            pass
    finally:
        holder.kill()
        holder.wait()
    with (
        Journal(path).hold_run(RUN),
        pytest.raises(RunBusy, match="in this process"),
        Journal(alias).hold_run(RUN),
    ):
        pass


def test_run_ids_that_differ_only_in_case_are_two_runs(tmp_path: Path) -> None:
    """The lock file was named after the run id, and a filesystem that ignores case -- macOS's
    by default -- made ``run-a`` and ``RUN-A`` one file: one could not be driven while the other
    was. Found by the eleventh review."""
    journal = Journal(tmp_path / "journal.db")
    with journal.hold_run("run-a"), journal.hold_run("RUN-A"):
        pass


def test_a_lock_that_cannot_be_made_is_a_journal_error(tmp_path: Path) -> None:
    """Not an ``OSError`` from deep inside, which the CLI did not catch."""
    from specunode.journal.journal import JournalError

    path = tmp_path / "journal.db"
    journal = Journal(path)
    (tmp_path / "journal.db.locks").write_text("not a folder")
    with pytest.raises(JournalError, match="cannot take run"), journal.hold_run(RUN):
        pass


class SlowLock(Journal):
    """A journal whose run lock takes a while to take, as a Postgres one does on a slow server."""

    @contextlib.contextmanager
    def _run_lock(self, run_id: str) -> Iterator[None]:
        time.sleep(0.3)
        with super()._run_lock(run_id):
            yield


async def test_taking_the_run_lock_does_not_hold_up_other_runs(tmp_path: Path) -> None:
    """A Postgres run lock opens a connection, and it was taken on the event loop: a slow server
    held up every run in the process. Found by the twelfth review."""
    journal = SlowLock(tmp_path / "journal.db")
    ticks: list[float] = []

    async def other_work() -> None:
        for _ in range(10):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    started = time.monotonic()
    ticker = asyncio.create_task(other_work())
    async with journal.hold_run_async(RUN):
        pass
    await ticker
    assert sum(1 for tick in ticks if tick < started + 0.25) >= 5, "the loop stood still"


async def test_a_run_cancelled_while_taking_its_lock_is_not_left_held(tmp_path: Path) -> None:
    journal = SlowLock(tmp_path / "journal.db")

    async def hold() -> None:
        async with journal.hold_run_async(RUN):
            await asyncio.Event().wait()

    holding = asyncio.create_task(hold())
    await asyncio.sleep(0.05)  # the lock is being taken
    holding.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await holding
    with Journal(tmp_path / "journal.db").hold_run(RUN):
        pass
