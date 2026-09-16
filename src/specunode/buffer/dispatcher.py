"""Dispatch: at-least-once, with deterministic idempotency keys.

Not exactly-once. The docs say "at-least-once dispatch with deterministic idempotency keys",
and the vocabulary test fails the build on an unqualified "exactly-once", because the only
thing that makes a repeated delivery harmless is the *tool* honouring its key -- and whether
it does is the developer's word, measured in attack 7.4 rather than assumed.

Backoff is exponential and **unjittered by default**. Jitter is the usual advice and it is
wrong here: the chaos matrix has to reproduce a failure it found, and a random delay makes a
leak that appeared once unreproducible.

Every failure reports whether the request *left the process*. That single bit is what keeps
the ambiguous crash window narrow: a connection refused before any bytes went out is safe to
retry, while a timeout after the request was sent is the two-generals boundary and is treated
as such.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from specunode.canonical import JsonValue
from specunode.core.effects import ToolRegistry, UnknownTool
from specunode.core.model import CallScope, call_scope

__all__ = ["DispatchOutcome", "Dispatcher", "ToolDispatchError"]

Sent = Literal["no", "maybe"]


class ToolDispatchError(RuntimeError):
    """A dispatch attempt failed.

    ``sent`` is the load-bearing field. ``"no"`` means nothing left this process, so a retry
    cannot duplicate anything. ``"maybe"`` -- the default, because it is the safe assumption --
    means the upstream may already have acted.
    """

    def __init__(self, message: str, *, sent: Sent = "maybe", retriable: bool = True) -> None:
        super().__init__(message)
        self.sent: Sent = sent
        self.retriable = retriable


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    ok: bool
    ack: JsonValue = None
    attempts: int = 0
    error: str | None = None
    sent: Sent = "no"


@dataclass
class Dispatcher:
    """Sends a staged effect to the world, with bounded retries."""

    registry: ToolRegistry
    max_attempts: int = 5
    base_delay_ms: float = 50.0
    cap_delay_ms: float = 5000.0
    #: Off by default so a chaos run can reproduce what it found. Turn it on in production,
    #: where a thundering herd matters more than reproducibility.
    jitter: bool = False

    def _delay_ms(self, attempt: int) -> float:
        return float(min(self.base_delay_ms * (2.0 ** (attempt - 1)), self.cap_delay_ms))

    async def dispatch(
        self,
        tool: str,
        args: dict[str, JsonValue],
        *,
        idempotency_key: str,
        branch_id: str,
    ) -> DispatchOutcome:
        """Call ``tool``, retrying until it acks or the attempts run out.

        The call runs under a :class:`CallScope` carrying the branch and the key, which is how
        the fake world attributes every mutation -- and therefore how the leak test can state
        its invariant as a set comparison.
        """
        spec = self.registry.get(tool)
        last_error: str | None = None
        last_sent: Sent = "no"

        for attempt in range(1, self.max_attempts + 1):
            scope = CallScope(branch_id=branch_id, effect_key=idempotency_key, speculative=False)
            token = call_scope.set(scope)
            try:
                ack = await spec.fn(**args)
                return DispatchOutcome(ok=True, ack=ack, attempts=attempt, sent="maybe")
            except UnknownTool as exc:
                # Not retriable and not ambiguous: there was nothing to call.
                return DispatchOutcome(ok=False, attempts=attempt, error=str(exc), sent="no")
            except ToolDispatchError as exc:
                last_error, last_sent = str(exc), exc.sent
                if not exc.retriable:
                    break
            except asyncio.CancelledError:
                # A drain is never speculative, so a cancellation here is the process going
                # away rather than a squash. Do not swallow it.
                raise
            except Exception as exc:  # an adapter that raised something of its own
                last_error, last_sent = f"{type(exc).__name__}: {exc}", "maybe"
            finally:
                call_scope.reset(token)

            if attempt < self.max_attempts:
                await asyncio.sleep(self._delay_ms(attempt) / 1000.0)

        return DispatchOutcome(
            ok=False, attempts=self.max_attempts, error=last_error, sent=last_sent
        )
