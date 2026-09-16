"""Fault injection for the fake world.

The invariants this project claims are about behaviour under failure, so the tests need a
world that fails on demand and in a way that repeats. Every fault here is deterministic:
it fires on a counted attempt or on a named tool, never on a coin flip, so a chaos run that
found a leak can be replayed exactly.

Four faults, matching the spec's list:

``partition(at=n)``
    the *n*-th dispatch attempt onward raises :class:`Partitioned` until :meth:`Faults.heal`
``timeout(tool)``
    calls to ``tool`` raise :class:`Timeout`
``duplicate_delivery(tool)``
    the upstream receives the call twice -- the network duplicating a message *after* the
    dispatcher has done its own deduplication, which is the case dedupe cannot cover
``slow(kind, ms)``
    reads or writes take ``ms`` milliseconds, so a branch can be squashed mid-call
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["Faults", "Partitioned", "Timeout", "WorldFault"]


class WorldFault(Exception):
    """Base class for an injected upstream failure."""


class Partitioned(WorldFault):
    """The upstream is unreachable. Retryable: the dispatcher should back off."""


class Timeout(WorldFault):
    """The upstream did not answer in time.

    Retryable, and ambiguous by nature: the call may or may not have taken effect upstream.
    That ambiguity is exactly why dispatch is at-least-once with idempotent dedupe.
    """


@dataclass
class Faults:
    """A deterministic fault plan."""

    partition_at: int | None = None
    partitioned: bool = False
    timeout_tools: set[str] = field(default_factory=set)
    duplicate_tools: set[str] = field(default_factory=set)
    slow_read_ms: int = 0
    slow_write_ms: int = 0

    #: Counts every upstream attempt, so ``partition(at=2)`` means "the second one".
    attempts: int = 0

    def partition(self, *, at: int | None = None) -> None:
        """Cut the link, either now (``at=None``) or from the ``at``-th attempt onward."""
        if at is None:
            self.partitioned = True
        else:
            self.partition_at = at

    def heal(self) -> None:
        self.partitioned = False
        self.partition_at = None

    def timeout(self, tool: str) -> None:
        self.timeout_tools.add(tool)

    def duplicate_delivery(self, tool: str) -> None:
        self.duplicate_tools.add(tool)

    def slow(self, kind: Literal["read", "write"], ms: int) -> None:
        if kind == "read":
            self.slow_read_ms = ms
        else:
            self.slow_write_ms = ms

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    # -- hooks the world calls -----------------------------------------------------------

    def before_call(self, tool: str) -> None:
        """Raise if this attempt should fail. Counts the attempt either way."""
        self.attempts += 1
        reached_partition = self.partition_at is not None and self.attempts >= self.partition_at
        if self.partitioned or reached_partition:
            raise Partitioned(f"upstream unreachable on attempt {self.attempts} for {tool!r}")
        if tool in self.timeout_tools:
            raise Timeout(f"upstream timed out for {tool!r}")

    async def delay(self, *, write: bool) -> None:
        """Sleep the configured latency.

        ``asyncio.sleep`` is a cancellation point, which is what task 3.7 needs: a branch
        squashed while a slow read is in flight must actually stop.
        """
        ms = self.slow_write_ms if write else self.slow_read_ms
        if ms:
            await asyncio.sleep(ms / 1000.0)

    def deliveries(self, tool: str) -> int:
        return 2 if tool in self.duplicate_tools else 1
