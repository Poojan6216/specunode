"""Replay: re-run a journaled run without asking a model anything.

Hard Rule 5's second half. :class:`ReplayModel` serves the responses the journal recorded, and
refuses the moment the run it is replaying would have asked a *different question*. That
refusal is the point. A replay that quietly re-ran a divergent trajectory would look like a
successful reproduction while proving nothing, and the run it produced would share a run id
with a run that never happened.

What counts as "different" is :func:`~specunode.core.model.request_hash` -- the same envelope
projection Hard Rule 13 uses live. So changing one token of the system prompt, or adding a
tool to the registry, diverges at the first step rather than somewhere downstream where the
consequence finally shows (spec attack 7.8).

Only entries on the **retired chain** are served. A speculative run journals model responses
for branches that were later squashed; those are evidence for the ledger and for forensics,
and feeding one back as an input would replay a decision the run never actually made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field

from specunode.canonical import JsonValue
from specunode.core.model import (
    ModelResponse,
    RequestEnvelope,
    StreamEvent,
    ToolUseBlock,
    ToolUseComplete,
    TurnComplete,
    current_scope,
    project,
    request_hash,
    response_from_json,
)
from specunode.journal.journal import Journal

__all__ = ["JournaledTurn", "ReplayDivergence", "ReplayExhausted", "ReplayModel", "diff_requests"]


class ReplayDivergence(RuntimeError):
    """The run being replayed would ask the model something the journal does not record.

    Carries the step and a field-level difference, so the answer to "what changed?" is in the
    exception rather than in a diff the operator has to reconstruct.
    """

    def __init__(self, step: int, diff: Sequence[str], *, expected: str, actual: str) -> None:
        self.step = step
        self.diff = tuple(diff)
        self.expected_hash = expected
        self.actual_hash = actual
        detail = "\n  ".join(diff) if diff else "(projections differ in an unlisted field)"
        super().__init__(
            f"replay diverged at step {step}: the request this code would send does not match "
            f"the one the journal recorded.\n"
            f"  journaled request_hash {expected}\n"
            f"  this run's request_hash {actual}\n"
            f"  {detail}"
        )


class ReplayExhausted(RuntimeError):
    """The replayed run asked for a step the journal has no response for."""


@dataclass(frozen=True)
class JournaledTurn:
    step: int
    branch_id: str
    request_hash: str
    request: Mapping[str, JsonValue]
    response: ModelResponse
    speculative: bool


def _describe(value: JsonValue, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tool_names(tools: JsonValue) -> list[JsonValue]:
    if not isinstance(tools, list):
        return []
    return [t.get("name") for t in tools if isinstance(t, Mapping)]


def diff_requests(expected: Mapping[str, JsonValue], actual: Mapping[str, JsonValue]) -> list[str]:
    """Field-level differences between two request projections, most useful first."""
    notes: list[str] = []
    for key in ("model", "tool_choice", "params", "system"):
        if expected.get(key) != actual.get(key):
            notes.append(
                f"{key}: journaled {_describe(expected.get(key))} "
                f"!= now {_describe(actual.get(key))}"
            )

    want_tools = expected.get("tools")
    have_tools = actual.get("tools")
    if want_tools != have_tools:
        want_names = _tool_names(want_tools)
        have_names = _tool_names(have_tools)
        if want_names != have_names:
            added = [n for n in have_names if n not in want_names]
            removed = [n for n in want_names if n not in have_names]
            notes.append(f"tools: added {added}, removed {removed}")
        else:
            notes.append("tools: same names, different descriptions or schemas")

    want_messages = expected.get("messages")
    have_messages = actual.get("messages")
    if want_messages != have_messages:
        want_list = want_messages if isinstance(want_messages, list) else []
        have_list = have_messages if isinstance(have_messages, list) else []
        if len(want_list) != len(have_list):
            notes.append(f"messages: journaled {len(want_list)}, now {len(have_list)}")
        for index, (a, b) in enumerate(zip(want_list, have_list, strict=False)):
            if a != b:
                notes.append(f"messages[{index}]: journaled {_describe(a)} != now {_describe(b)}")
                break
    return notes


@dataclass
class ReplayModel:
    """A :class:`~specunode.core.model.ModelClient` that reads instead of asking.

    Never calls a model. Never writes to the journal it is reading.
    """

    journal: Journal
    run_id: str
    #: Branch ids that reached RETIRED. When empty, only non-speculative turns are served,
    #: which is the right answer for a journal written by a sequential run.
    retired_branches: frozenset[str] = frozenset()
    role: str = "target"
    _turns: dict[int, JournaledTurn] = field(default_factory=dict, init=False)
    _served: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        requests: dict[str, Mapping[str, JsonValue]] = {}
        for entry in self.journal.read(self.run_id, kinds=["model_request", "model_response"]):
            payload = entry.payload
            if payload.get("role") != self.role:
                continue
            if entry.kind == "model_request":
                request_id = payload.get("request_id")
                if isinstance(request_id, str):
                    requests[request_id] = payload
                continue

            request_id = payload.get("request_id")
            request = requests.get(request_id) if isinstance(request_id, str) else None
            if request is None:
                continue
            speculative = bool(payload.get("speculative", False))
            branch_id = str(payload.get("branch_id", ""))
            # The retired-chain rule: a squashed branch's model output is evidence, never an
            # input. Replaying one would reproduce a decision the run did not make.
            if speculative and branch_id not in self.retired_branches:
                continue
            step = payload.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            response_payload = payload.get("response")
            if not isinstance(response_payload, Mapping):
                continue
            projected = request.get("request")
            self._turns[step] = JournaledTurn(
                step=step,
                branch_id=branch_id,
                request_hash=str(request.get("request_hash", "")),
                request=projected if isinstance(projected, Mapping) else {},
                response=response_from_json(response_payload),
                speculative=speculative,
            )

    # -- the ModelClient surface ---------------------------------------------------------------

    def _turn_for(self, envelope: RequestEnvelope, step: int) -> JournaledTurn:
        turn = self._turns.get(step)
        if turn is None:
            raise ReplayExhausted(
                f"the journal for run {self.run_id} has no {self.role} turn at step {step}; "
                f"it records steps {sorted(self._turns)}"
            )
        actual = request_hash(envelope)
        if actual != turn.request_hash:
            raise ReplayDivergence(
                step,
                diff_requests(turn.request, _as_mapping(project(envelope))),
                expected=turn.request_hash,
                actual=actual,
            )
        self._served.append(step)
        return turn

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return self._turn_for(envelope, current_scope().step).response

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        """Re-emit the journaled turn's blocks, with no delay.

        Stream timings are a measurement, never a replay input: reproducing them would make
        replay take as long as the original run and would make its result depend on a number
        that has nothing to do with correctness.
        """
        response = self._turn_for(envelope, current_scope().step).response
        for index, block in enumerate(response.content):
            if isinstance(block, ToolUseBlock):
                yield ToolUseComplete(index=index, block=block)
        yield TurnComplete(response=response)

    # -- introspection for tests and the CLI -----------------------------------------------------

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._turns))

    @property
    def served(self) -> tuple[int, ...]:
        return tuple(self._served)


def _as_mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if isinstance(value, Mapping) else {}
