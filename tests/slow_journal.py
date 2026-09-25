"""A pytest plugin that makes every async journal append slower, the way a CI disk is.

``pytest -p tests.slow_journal`` with ``SPECUNODE_SLOW_JOURNAL_MS=40`` adds that much latency
before each append reaches the writer thread. A test that passes only because an append landed
inside some block delay fails under it on any machine, rather than on a CI runner on one run in
twenty. The whole suite passed under it at 40 ms when it was added, and the ``slow-disk`` CI job
runs the tests that guess under it on every push.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

import pytest

from specunode.canonical import JsonValue
from specunode.journal.journal import Journal


def pytest_configure(config: pytest.Config) -> None:
    delay_ms = float(os.environ.get("SPECUNODE_SLOW_JOURNAL_MS", "40"))
    append = Journal.append_async

    async def slow_append(
        self: Journal, run_id: str, kind: str, payload: Mapping[str, JsonValue]
    ) -> int:
        await asyncio.sleep(delay_ms / 1000.0)
        return await append(self, run_id, kind, payload)

    Journal.append_async = slow_append  # type: ignore[method-assign]
