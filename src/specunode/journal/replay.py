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
from specunode.core.branch import StepCursor
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

__all__ = [
    "JournaledTurn",
    "Recovery",
    "ReplayDivergence",
    "ReplayExhausted",
    "ReplayModel",
    "attested_origins",
    "diff_requests",
    "recover",
]


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


# -- attestation ------------------------------------------------------------------------------


def attested_origins(
    journal: Journal, run_id: str, lineage: Sequence[str], upto_step: int
) -> frozenset[str]:
    """Branch ids whose journaled output may be read back as an *input*.

    Two sets, unioned: branches the journal records as RETIRED at or before ``upto_step``, and
    this branch's own lineage -- itself and its ancestors, which are unresolved but are its own
    past rather than somebody else's alternative.

    The union is necessary because ``lineage`` resets at every retirement: it is a
    key-derivation value, so at step five it names only the current extent, and a filter built
    on it alone would hide every earlier assistant turn. A rebuild under that filter matches
    nothing, every branch reports a context divergence, and the step is redone forever.

    The tempting repair, once that is seen, is to admit anything from a branch that is *not*
    squashed. That admits a sibling that has not resolved yet -- the exact channel Hard Rule 6
    exists to close -- so the test is membership in this set, never absence from a blacklist.
    """
    retired: set[str] = set()
    for entry in journal.read(run_id, kinds=["branch_resolved"]):
        payload = entry.payload
        if payload.get("status") != "retired":
            continue
        step = payload.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step > upto_step:
            continue
        branch_id = payload.get("branch_id")
        if isinstance(branch_id, str):
            retired.add(branch_id)
    return frozenset(retired | set(lineage))


# -- recovery ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recovery:
    """What a crashed run left behind, and what a resumed one may build on.

    Only the **retired chain** contributes. A branch that was confirmed but never journaled as
    retired, and a branch that was mid-drain when the process died, are both evidence rather
    than input: their state deltas are not applied and their cursors are not adopted. Resuming
    from a branch that had a durable ``model_response`` but no confirming entry would dispatch
    an effect that was never context-checked and never witness-validated, which is Hard Rule 3
    violated by a crash rather than by a bug.
    """

    run_id: str
    #: Committed state, rebuilt by applying the retired branches' deltas in offset order.
    state: dict[str, JsonValue]
    #: The program position the last retirement committed, restored verbatim from the journal.
    #: Inferring it from the highest step visible instead lands the resumed run somewhere else,
    #: so every idempotency key it derives differs from its pre-crash value, the dedupe table
    #: misses, and effects that already went out go out again.
    cursor: StepCursor
    #: The step index inside that cursor, for the status command.
    step_index: int
    retired_branches: frozenset[str]
    #: Branches with a durable confirming entry but no retirement: their drain was in flight.
    confirmed_not_retired: tuple[str, ...]
    #: Dispatch claims left in flight. Each is a two-generals case until it is resolved.
    unresolved_dispatches: tuple[Mapping[str, JsonValue], ...]
    last_offset: int
    finished: bool

    @property
    def resumable(self) -> bool:
        return not self.finished or bool(self.confirmed_not_retired)


def _cursor_from(payload: JsonValue, fallback: StepCursor) -> StepCursor:
    """Read a journaled ``cursor_after`` back into a cursor."""
    if not isinstance(payload, Mapping):
        return fallback
    step = payload.get("step_index")
    raw_visits = payload.get("visits")
    visits: list[tuple[str, int]] = []
    if isinstance(raw_visits, Sequence) and not isinstance(raw_visits, str):
        for item in raw_visits:
            if isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2:
                name, count = item
                if isinstance(name, str) and isinstance(count, int):
                    visits.append((name, count))
    return StepCursor(
        step_index=step if isinstance(step, int) and not isinstance(step, bool) else 0,
        visits=tuple(sorted(visits)),
    )


def recover(journal: Journal, run_id: str) -> Recovery:
    """Read a run's journal and work out what a resume may safely build on."""
    from specunode.core.state import apply, operation_from_json

    retired: set[str] = set()
    confirmed: set[str] = set()
    deltas: list[tuple[int, str, Sequence[JsonValue]]] = []
    cursor = StepCursor()
    last_offset = -1
    finished = False

    for entry in journal.read(run_id):
        last_offset = entry.offset
        payload = entry.payload
        if entry.kind == "run_finished":
            finished = True
        elif entry.kind == "branch_resolved":
            branch_id = payload.get("branch_id")
            status = payload.get("status")
            if isinstance(branch_id, str):
                if status == "retired":
                    retired.add(branch_id)
                    confirmed.discard(branch_id)
                    cursor = _cursor_from(payload.get("cursor_after"), cursor)
                elif status == "confirmed":
                    confirmed.add(branch_id)
        elif entry.kind == "state_delta_applied":
            branch_id = payload.get("branch_id")
            patch = payload.get("patch")
            if isinstance(branch_id, str) and isinstance(patch, Sequence):
                deltas.append((entry.offset, branch_id, patch))

    state: JsonValue = {}
    for _offset, branch_id, patch in deltas:
        if branch_id not in retired:
            continue
        operations = [operation_from_json(op) for op in patch if isinstance(op, Mapping)]
        state = apply(state, operations)

    return Recovery(
        run_id=run_id,
        state=dict(state) if isinstance(state, Mapping) else {},
        cursor=cursor,
        step_index=cursor.step_index,
        retired_branches=frozenset(retired),
        confirmed_not_retired=tuple(sorted(confirmed)),
        unresolved_dispatches=tuple(journal.unresolved_dispatches(run_id)),
        last_offset=last_offset,
        finished=finished,
    )
