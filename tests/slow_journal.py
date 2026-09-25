"""A pytest plugin that makes every durable journal write slower, the way a CI disk is.

``pytest -p tests.slow_journal`` with ``SPECUNODE_SLOW_JOURNAL_MS=40`` adds that much latency
before each write the runtime makes -- an entry appended, a dispatch claimed, a claim marked
unsent, an outcome settled -- reaches the writer thread. The delay is awaited on the event
loop, so concurrent writes are slowed side by side rather than queued behind one another. A
test that passes only because a write landed inside some block delay fails under it on any
machine, rather than on a CI runner on one run in twenty. The blocking ``Journal.append``,
which the runtime never calls, is left alone.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from specunode.journal.journal import Journal

#: Every coroutine on :class:`Journal` that writes to the database.
WRITES = ("append_async", "claim_dispatch", "mark_not_sent", "settle_dispatch")


def pytest_configure(config: pytest.Config) -> None:
    delay_s = float(os.environ.get("SPECUNODE_SLOW_JOURNAL_MS", "40")) / 1000.0
    for name in WRITES:
        write: Callable[..., Awaitable[Any]] = getattr(Journal, name)

        async def slow(
            self: Journal, *args: Any, _write: Callable[..., Awaitable[Any]] = write, **kwargs: Any
        ) -> Any:
            await asyncio.sleep(delay_s)
            return await _write(self, *args, **kwargs)

        setattr(Journal, name, slow)
