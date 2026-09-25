"""Kill a process at an exact point in a run, for the kill/resume tests and the demo sweep.

A kill point is named, not timed:

* ``op:N`` -- die just before the run's N-th durable journal write: an entry appended, a
  dispatch claimed, a claim marked unsent, or an outcome settled
* ``send:N`` -- die just before the world applies its N-th mutation: the claim is on disk and
  the request never arrived -- the lost request
* ``mutation:N`` -- die just after the world has durably applied its N-th mutation, before the
  runtime has recorded that it did -- the request went out and the reply was lost

The process dies by ``os._exit``: no cleanup, no flush, no finally blocks, as if the machine
lost power, except that what the OS already holds survives.

Kill points were once delays in milliseconds, calibrated against a timed run. A delay is not a
point in the run: on a CI runner the same delay landed before the run had journaled anything on
one attempt and after it had finished on another, so a sweep missed most of the run on a slow
machine and still passed. Counting names each point at which the journal, or the effects in the
world, change on disk -- on every machine, every time.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from specunode.journal import journal as journal_module
from specunode.testing.world import World

#: Every method that changes the journal database, all run on its writer thread.
DURABLE_WRITES = ("_append", "_claim", "_mark_not_sent", "_settle")


class Counter:
    """Counts one kind of event and dies at the chosen one."""

    def __init__(self, die_at: int | None) -> None:
        self.count = 0
        self.die_at = die_at
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self.count += 1
            if self.count == self.die_at:
                os._exit(9)


@dataclass
class KillPoints:
    writes: Counter
    sends: Counter
    mutations: Counter

    def report(self) -> str:
        """What a run that finished prints, so a sweep can name every point it has."""
        return f"ops={self.writes.count} mutations={self.mutations.count}"


def is_kill_point(spec: str) -> bool:
    return spec.partition(":")[0] in ("op", "send", "mutation")


def arm(spec: str) -> KillPoints:
    """Count journal writes from now on, dying at ``spec`` if it names one. ``-1``: never."""
    kind, _, at = spec.partition(":")
    points = KillPoints(
        writes=Counter(int(at) if kind == "op" else None),
        sends=Counter(int(at) if kind == "send" else None),
        mutations=Counter(int(at) if kind == "mutation" else None),
    )
    writer = journal_module._JournalWriter
    for name in DURABLE_WRITES:
        original: Callable[..., Any] = getattr(writer, name)

        def before(
            self: object,
            *args: Any,
            _original: Callable[..., Any] = original,
            _counter: Counter = points.writes,
        ) -> Any:
            _counter.tick()
            return _original(self, *args)

        setattr(writer, name, before)
    return points


def watch(world: World, points: KillPoints) -> None:
    """Tick either side of the world log's fsync: ``send:N`` before, ``mutation:N`` after.

    Dying before the log write loses the request -- the world's in-memory change dies with the
    process, and the world a resumed process opens never had it. Dying after loses the reply.
    """
    original = world._persist_mutation

    def around(*args: Any) -> None:
        points.sends.tick()
        original(*args)
        points.mutations.tick()

    world._persist_mutation = around  # type: ignore[method-assign]
