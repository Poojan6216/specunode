"""Tier 0: issue each tool call the instant the target finishes emitting it.

The model streams its turn. A ``tool_use`` block becomes complete some time before the turn
does -- often hundreds of milliseconds before, on a turn that emits several. Tier 0 hands that
block to the scheduler as a candidate the moment it parses, so a read can be in flight while
the model is still writing the rest of its answer.

Nothing is being *guessed*. These are calls the target has already emitted, so the gate
confirms every one of them: acceptance is 1 by construction, and the ledger records tier 0
separately for exactly that reason -- mixing it into a drafter accuracy number would flatter
the ones that really predict. What is being speculated is time, not content.

The pattern is Claude Code's streaming tool executor, credited as theirs.

Hard Rule 3 is unaffected. A branch forked on a tier-0 candidate is SPECULATIVE like any other:
its reads run, its writes stage, and nothing dispatches until the turn's ``model_response``
entry is durable and the branch is confirmed. Tier 0 shortens the wait; it does not shorten the
proof.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from specunode.core.decision import Decision, ToolCall
from specunode.core.model import StreamEvent, ToolUseComplete, TurnComplete
from specunode.drafters.base import DraftContext, Prediction

__all__ = ["StreamDrafter"]


@dataclass
class StreamDrafter:
    """Serves the blocks the current turn has already emitted."""

    tier: Literal[0, 1, 2] = 0
    #: Blocks seen in the in-flight turn, in stream order. Program order, never completion
    #: order: the model asked for them in this sequence and the results must line up with it.
    _pending: list[Decision] = field(default_factory=list)
    _consumed: int = 0

    def observe(self, event: StreamEvent) -> Decision | None:
        """Feed a stream event in. Returns a newly available decision, if this event is one."""
        if isinstance(event, ToolUseComplete):
            decision = ToolCall(name=event.block.name, args=event.block.args)
            self._pending.append(decision)
            return decision
        if isinstance(event, TurnComplete):
            self.end_turn()
        return None

    def end_turn(self) -> None:
        """The turn finished. Anything unconsumed belongs to the turn that just closed."""
        self._pending.clear()
        self._consumed = 0

    @property
    def pending(self) -> int:
        """Blocks emitted by the in-flight turn that the scheduler has not taken yet."""
        return len(self._pending) - self._consumed

    async def predict(self, ctx: DraftContext) -> Sequence[Prediction]:
        """The next block this turn has already emitted, if there is one.

        At most one candidate. A turn's blocks are consumed in order -- the model asked for
        them in that order, and a branch speculating block *j+1* is a descendant of the one
        speculating block *j*, so offering several at once would fork siblings on decisions
        that are not alternatives to each other.
        """
        if ctx.partial_turn:
            index = min(self._consumed, len(ctx.partial_turn) - 1)
            if self._consumed < len(ctx.partial_turn):
                return [Prediction(decision=ctx.partial_turn[index], tier=0, score=1.0)]
            return []
        if self._consumed >= len(self._pending):
            return []
        return [Prediction(decision=self._pending[self._consumed], tier=0, score=1.0)]

    def take(self) -> Decision | None:
        """Consume the next emitted block, marking it handed to the scheduler."""
        if self._consumed >= len(self._pending):
            return None
        decision = self._pending[self._consumed]
        self._consumed += 1
        return decision
