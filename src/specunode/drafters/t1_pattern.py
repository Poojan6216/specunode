"""Tier 1: an order-*k* pattern index over tool signatures.

**This mechanism is PASTE's.** *Act While Thinking: Accelerating LLM Agents via Pattern-Aware
Speculative Tool Execution* (Sui et al., Microsoft Research, arXiv 2603.18897) characterises
agent traces and finds strong temporal locality in tool sequences -- "strong chains" and
"refinement loops" -- plus predictable data flow between one call's output and the next call's
arguments. This module re-implements that idea and is credited to them in the README.

What is different here is not the predictor. It is what happens when the predictor is wrong:
PASTE cannot speculate on a tool with side effects at all, because a wrong guess has already
changed the world. Here a wrong guess stages into a store buffer and is discarded, so the
predictor is allowed to be wrong about writes as well as reads. Its accuracy is therefore a
*performance* number rather than a safety one, and it is measured per workload rather than
claimed.

Two rules keep the index honest:

**It is mined from retired lineages only.** A squashed branch's calls are what the run decided
*not* to do; training on them teaches the index to predict the runtime's own mistakes back to
it.

**It is deterministic given the index file.** The same index and the same history always
produce the same candidates in the same order, so a replay reproduces the speculation exactly
and the benchmark's numbers are reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from specunode.canonical import JsonValue, canonical
from specunode.core.decision import Decision, ToolCall
from specunode.drafters.base import DraftContext, Prediction

__all__ = ["MAX_ORDER", "PatternDrafter", "PatternIndex", "mine_from_journal", "signature_of"]

#: Order-k, k <= 3. Longer contexts fragment the counts faster than they add information, on
#: traces of the length agent runs actually reach.
MAX_ORDER = 3

#: ASCII unit separator, joining a context window into one dict key. A character no tool name
#: contains, so a window of two signatures cannot collide with one long signature.
_SEP = chr(31)


def signature_of(decision: Decision) -> str:
    """A tool call reduced to what repeats across runs: its name and its argument *shape*.

    Values are dropped deliberately. Two ``create_ticket`` calls with different customer ids
    are the same step of the same pattern; keying on values would make every trace unique and
    the index would predict nothing.
    """
    if not isinstance(decision, ToolCall):
        return f"<{decision.kind}>"
    shape = ",".join(sorted(decision.args))
    return f"{decision.name}({shape})"


@dataclass
class PatternIndex:
    """Order-*k* transition counts over tool signatures, plus argument shapes."""

    order: int = 2
    #: joined context window -> next signature -> count
    transitions: dict[str, dict[str, int]] = field(default_factory=dict)
    #: signature -> the argument keys seen with it, for instantiating a candidate
    arg_keys: dict[str, list[str]] = field(default_factory=dict)
    #: signature -> tool name, so a signature can be turned back into a call
    tool_of: dict[str, str] = field(default_factory=dict)
    trained_on: int = 0

    @staticmethod
    def _window(history: Sequence[Decision], length: int) -> str:
        if length <= 0:
            return ""
        return _SEP.join(signature_of(d) for d in history[len(history) - length :])

    def observe(self, history: Sequence[Decision], nxt: Decision) -> None:
        """Count one transition. Shorter contexts are counted too, so ranking can back off."""
        if not isinstance(nxt, ToolCall):
            return
        signature = signature_of(nxt)
        self.tool_of[signature] = nxt.name
        self.arg_keys[signature] = sorted(nxt.args)
        for length in range(min(self.order, len(history)), -1, -1):
            key = self._window(history, length)
            counts = self.transitions.setdefault(key, {})
            counts[signature] = counts.get(signature, 0) + 1

    def train(self, traces: Sequence[Sequence[Decision]]) -> None:
        for trace in traces:
            for index in range(len(trace)):
                self.observe(trace[:index], trace[index])
            self.trained_on += 1

    def rank(self, history: Sequence[Decision]) -> list[tuple[str, float]]:
        """Candidate signatures, most likely first, backing off to shorter contexts.

        Ties break on the signature string so the order is total and reproducible. A dict's
        insertion order would make the benchmark depend on the order traces happened to be read
        in, which is not a property of the workload.
        """
        for length in range(min(self.order, len(history)), -1, -1):
            counts = self.transitions.get(self._window(history, length))
            if not counts:
                continue
            total = sum(counts.values())
            return sorted(
                ((sig, count / total) for sig, count in counts.items()),
                key=lambda pair: (-pair[1], pair[0]),
            )
        return []

    # -- persistence -------------------------------------------------------------------------

    def to_json(self) -> JsonValue:
        return {
            "order": self.order,
            "trained_on": self.trained_on,
            "transitions": {
                key: dict(sorted(counts.items()))
                for key, counts in sorted(self.transitions.items())
            },
            "arg_keys": dict(sorted(self.arg_keys.items())),
            "tool_of": dict(sorted(self.tool_of.items())),
        }

    def save(self, path: Path) -> None:
        """Write the index as canonical bytes.

        Canonical rather than pretty-printed, so two builds of the same index are byte-identical
        and the benchmark's "predictions are identical across three runs" check has something to
        stand on.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(self.to_json()))

    @classmethod
    def load(cls, path: Path) -> PatternIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(order=int(payload.get("order", 2)))
        index.trained_on = int(payload.get("trained_on", 0))
        index.transitions = {
            str(key): {str(sig): int(count) for sig, count in counts.items()}
            for key, counts in payload.get("transitions", {}).items()
        }
        index.arg_keys = {
            str(k): [str(x) for x in v] for k, v in payload.get("arg_keys", {}).items()
        }
        index.tool_of = {str(k): str(v) for k, v in payload.get("tool_of", {}).items()}
        return index


@dataclass
class PatternDrafter:
    """Predicts the next call from the index, filling its arguments from history."""

    index: PatternIndex
    tier: Literal[0, 1, 2] = 1
    top_k: int = 1
    #: A prediction whose arguments cannot be filled is not offered at all. The gate compares
    #: exact canonical arguments, so a half-filled call squashes every time, and its only
    #: lasting effect is the upstream reads its branch paid for on the way there.
    require_complete_args: bool = True

    async def predict(self, ctx: DraftContext) -> Sequence[Prediction]:
        out: list[Prediction] = []
        for signature, score in self.index.rank(ctx.history)[: max(self.top_k, 0)]:
            tool = self.index.tool_of.get(signature)
            if tool is None or (ctx.known_tools and tool not in ctx.known_tools):
                continue
            args = self._instantiate(signature, ctx)
            if args is None:
                continue
            out.append(Prediction(decision=ToolCall(name=tool, args=args), tier=1, score=score))
        return out

    def _instantiate(self, signature: str, ctx: DraftContext) -> Mapping[str, JsonValue] | None:
        """Fill a predicted call's arguments from what this branch has already seen.

        Prior *results* are searched as well as prior *arguments*, and that is the half that
        matters. The interesting chains are the ones where one call's output becomes the next
        call's input -- ``charge_card`` returns a charge id and ``send_receipt`` needs it --
        and an index that only carried argument values forward would predict the signature
        correctly and then be unable to fill it, offering nothing. This is PASTE's data-flow
        observation; the JSONPath generality is not implemented, only a keyed lookup.
        """
        keys = self.index.arg_keys.get(signature, [])
        if not keys:
            return {}

        seen: dict[str, JsonValue] = {}
        for decision in ctx.history:
            if isinstance(decision, ToolCall):
                seen.update(decision.args)
        for result in ctx.results.values():
            _collect(result, seen)

        args: dict[str, JsonValue] = {}
        for key in keys:
            if key in seen:
                args[key] = seen[key]
            elif self.require_complete_args:
                return None
        return args


#: How deep to look inside a tool result for a value a later call might name. Two levels covers
#: the shapes that actually occur -- a flat object, and one wrapped in {"value": ...} the way a
#: witnessed read returns -- without turning argument filling into a search.
_RESULT_DEPTH = 2


def _collect(value: JsonValue, into: dict[str, JsonValue], depth: int = _RESULT_DEPTH) -> None:
    """Gather named scalars out of a tool result, shallowest first."""
    if depth < 0 or not isinstance(value, Mapping):
        return
    for key, item in value.items():
        if isinstance(item, Mapping):
            _collect(item, into, depth - 1)
        elif key not in into:
            into[key] = item


def mine_from_journal(journal: object, run_id: str, order: int = 2) -> PatternIndex:
    """Build an index from a run's *retired* tool calls.

    Retired only: a squashed branch's calls are what the run decided not to do, and training on
    them teaches the index to predict the runtime's own mistakes back to it.
    """
    from specunode.journal.journal import Journal

    if not isinstance(journal, Journal):
        raise TypeError("mine_from_journal needs a Journal")

    retired = {
        str(entry.payload["branch_id"])
        for entry in journal.read(run_id, kinds=["branch_resolved"])
        if entry.payload.get("status") == "retired"
    }
    trace: list[Decision] = []
    for entry in journal.read(run_id, kinds=["tool_request"]):
        if str(entry.payload.get("branch_id")) not in retired:
            continue
        tool = entry.payload.get("tool")
        args = entry.payload.get("args")
        if isinstance(tool, str) and isinstance(args, Mapping):
            trace.append(ToolCall(name=tool, args=args))
    index = PatternIndex(order=min(order, MAX_ORDER))
    index.train([trace])
    return index
