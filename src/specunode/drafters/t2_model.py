"""Tier 2: ask a small model what comes next. Optional, and off by default.

A draft model is the only other model this runtime knows about, and Hard Rule 1 is precise
about its standing: its output "is only ever a guess". It proposes; the gate confirms by exact
canonical equality against the target's real decision, exactly as for tier 1. Nothing about
being a model gives a tier-2 prediction more authority than a Markov table's.

Off by default because a miss costs tokens as well as time. Tier 0 costs nothing and tier 1
costs almost nothing; a draft model bills for every guess, right or wrong, which makes the
break-even acceptance rate meaningfully higher for it than for the other two.

Two rules keep it honest, and both are enforced elsewhere but named here so a reader of this
file sees them:

**Its requests are journaled with ``role='draft'``.** Hard Rule 13's context-identity check
tracks target requests only. A draft request legitimately differs -- a different model at
minimum -- and folding it into that check would raise a divergence on the first call of every
tier-2 run. The obvious fix for *that* would be to loosen the comparison until it stopped
firing, which would also stop it catching real target-side divergence.

**It never sees anything the target did not.** The draft context is built from journaled
material on the branch's own lineage, so a draft model cannot be prompted with a sibling's work
and then have its guess confirmed against a turn that never saw any of it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from specunode.canonical import JsonValue
from specunode.core.decision import ToolCall
from specunode.core.model import (
    Message,
    ModelClient,
    RequestEnvelope,
    TextBlock,
    ToolDef,
)
from specunode.drafters.base import DraftContext, Prediction

__all__ = ["DEFAULT_DRAFT_MODEL", "ModelDrafter"]

#: A cheap model paired with a larger target, so a miss costs little. Both are swappable; the
#: claim this project makes is about the runtime, not about a particular pairing.
DEFAULT_DRAFT_MODEL = "claude-haiku-4-5"


@dataclass
class ModelDrafter:
    """Predicts the next tool call with a small model behind the same ModelClient protocol."""

    client: ModelClient
    model: str = DEFAULT_DRAFT_MODEL
    tier: Literal[0, 1, 2] = 2
    max_tokens: int = 512
    #: Tool definitions the draft model may choose between. Taken from the registry rather than
    #: written here, so the draft model cannot propose a tool the runtime does not know.
    tools: tuple[ToolDef, ...] = ()
    #: Predictions that could not be parsed into a ToolCall, reported rather than swallowed --
    #: a drafter silently returning nothing looks exactly like a drafter with no opinion.
    unparsable: int = field(default=0)

    async def predict(self, ctx: DraftContext) -> Sequence[Prediction]:
        """One candidate, or none. Never raises: an empty list means 'no opinion'."""
        if not ctx.known_tools:
            return []
        try:
            response = await self.client.complete(self._envelope(ctx))
        except Exception:
            # A draft model that is down, rate-limited or slow must not fail the run. The
            # sequential path is always correct; losing a speculation is not an error.
            return []

        for block in response.tool_uses:
            if block.name in ctx.known_tools:
                return [
                    Prediction(
                        decision=ToolCall(name=block.name, args=block.args), tier=2, score=0.5
                    )
                ]

        parsed = _parse_text_prediction(response.text, ctx.known_tools)
        if parsed is None:
            self.unparsable += 1
            return []
        return [Prediction(decision=parsed, tier=2, score=0.4)]

    def _envelope(self, ctx: DraftContext) -> RequestEnvelope:
        """Build the draft request from journaled history on this branch's own lineage."""
        history = "\n".join(
            f"{index}. {call.name} {json.dumps(dict(call.args), sort_keys=True)[:200]}"
            for index, call in enumerate(ctx.history)
            if isinstance(call, ToolCall)
        )
        return RequestEnvelope(
            model=self.model,
            system=(TextBlock(text=_SYSTEM),),
            messages=(
                Message(
                    role="user",
                    content=(
                        TextBlock(
                            text=(
                                f"Tools available: {', '.join(sorted(ctx.known_tools))}\n\n"
                                f"Calls so far:\n{history or '(none)'}\n\n"
                                "What is the next tool call?"
                            )
                        ),
                    ),
                ),
            ),
            tools=self.tools,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )


#: The one prompt in this package, and it lives here rather than in the control path. Hard
#: Rule 1 forbids a prompt in core/, buffer/, verify/ or journal/ -- this is a drafter, whose
#: entire output the gate must still confirm before anything it caused can reach the world.
_SYSTEM = (
    "You predict the next tool call an agent will make, from the calls it has already made. "
    "Answer with a single tool call and nothing else. If you cannot tell, say UNKNOWN."
)


def _parse_text_prediction(text: str, known: frozenset[str]) -> ToolCall | None:
    """Read a tool call out of a text answer, for a draft model with no tool-use support."""
    stripped = text.strip()
    if not stripped or stripped.upper().startswith("UNKNOWN"):
        return None
    start = stripped.find("{")
    if start >= 0:
        try:
            payload = json.loads(stripped[start:])
        except json.JSONDecodeError:
            return None
        if isinstance(payload, Mapping):
            name = payload.get("tool") or payload.get("name")
            args = payload.get("args") or payload.get("arguments") or {}
            if isinstance(name, str) and name in known and isinstance(args, Mapping):
                return ToolCall(name=name, args=dict(args))
        return None
    for candidate in sorted(known, key=len, reverse=True):
        if candidate in stripped:
            return ToolCall(name=candidate, args={})
    return None


def json_args(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if isinstance(value, Mapping) else {}
