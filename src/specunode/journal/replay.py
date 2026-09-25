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
    CallScope,
    Message,
    ModelResponse,
    RecordedTurn,
    RequestEnvelope,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
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
    "RecordedTurns",
    "Recovery",
    "ReplayDivergence",
    "ReplayExhausted",
    "ReplayModel",
    "attested_origins",
    "diff_requests",
    "fold_context",
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
    #: Recorded turns by (node, program position), in the order they were made. A list, not
    #: one turn: a node that runs a conversation -- ``agent_loop`` -- makes several model calls
    #: from one position, and an index that held one turn per step kept only the last, so a
    #: replay of any multi-turn node diverged on its first request. Keyed by node as well,
    #: because two nodes running side by side can each make a call from the same position.
    _turns: dict[tuple[str, int], list[JournaledTurn]] = field(default_factory=dict, init=False)
    _next: dict[tuple[str, int], int] = field(default_factory=dict, init=False)
    _served: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._load()
        self._keep_one_attempt()

    def _keep_one_attempt(self) -> None:
        """Serve each (node, position) the turns of the attempt the run kept, and no other.

        A resumed run re-runs a node that never retired from the same position, under the same
        node id, and asks the model again -- so the journal holds two attempts' turns under one
        key: the dead process's and the resumed one's. Served in journal order, replay handed
        the node the dead attempt's answer and reproduced a decision the committed run never
        made; for a conversation it compared the resumed turns against the dead ones and
        refused a faithful re-run. The attempt that retired is the one the run kept. When none
        did -- a run that ended mid-node -- the latest attempt is the nearest thing to it.
        """
        for key, turns in self._turns.items():
            attempts = list(dict.fromkeys(turn.branch_id for turn in turns))
            if len(attempts) < 2:
                continue
            kept = [a for a in attempts if a in self.retired_branches] or attempts[-1:]
            self._turns[key] = [turn for turn in turns if turn.branch_id in kept]

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
            key = (str(request.get("node_id") or ""), step)
            self._turns.setdefault(key, []).append(
                JournaledTurn(
                    step=step,
                    branch_id=branch_id,
                    request_hash=str(request.get("request_hash", "")),
                    request=projected if isinstance(projected, Mapping) else {},
                    response=response_from_json(response_payload),
                    speculative=speculative,
                )
            )

    # -- the ModelClient surface ---------------------------------------------------------------

    def _turn_for(self, envelope: RequestEnvelope, scope: CallScope) -> JournaledTurn:
        key = (scope.node_id, scope.step)
        recorded = self._turns.get(key, [])
        index = self._next.get(key, 0)
        if index >= len(recorded):
            raise ReplayExhausted(
                f"the journal for run {self.run_id} has no {self.role} turn number "
                f"{index + 1} for node {scope.node_id!r} at step {scope.step}; it records "
                f"{sorted((node, step, len(turns)) for (node, step), turns in self._turns.items())}"
            )
        turn = recorded[index]
        step = scope.step
        actual = request_hash(envelope)
        if actual != turn.request_hash:
            raise ReplayDivergence(
                step,
                diff_requests(turn.request, _as_mapping(project(envelope))),
                expected=turn.request_hash,
                actual=actual,
            )
        self._next[key] = index + 1
        self._served.append(step)
        return turn

    async def complete(self, envelope: RequestEnvelope) -> ModelResponse:
        return self._turn_for(envelope, current_scope()).response

    async def stream(self, envelope: RequestEnvelope) -> AsyncIterator[StreamEvent]:
        """Re-emit the journaled turn's blocks, with no delay.

        Stream timings are a measurement, never a replay input: reproducing them would make
        replay take as long as the original run and would make its result depend on a number
        that has nothing to do with correctness.
        """
        response = self._turn_for(envelope, current_scope()).response
        for index, block in enumerate(response.content):
            if isinstance(block, ToolUseBlock):
                yield ToolUseComplete(index=index, block=block)
        yield TurnComplete(response=response)

    # -- introspection for tests and the CLI -----------------------------------------------------

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(sorted({step for _node, step in self._turns}))

    @property
    def served(self) -> tuple[int, ...]:
        return tuple(self._served)


@dataclass
class RecordedTurns:
    """The model turns a resumed run has already been given, served rather than asked again.

    A resumed run re-runs every node whose branch never retired, from the same position and
    under the same node id, so each such node asks the model what it asked before. Asking
    again is what made "never duplicated" depend on the model answering the same way twice:
    the dead process may already have sent effects of that turn, and a different answer is a
    different call with a different idempotency key. The answer is in the journal whenever an
    effect of the turn could have been sent, because nothing is sent before the turn is
    durable; so a turn the journal holds is served, and only a turn it does not hold -- none of
    whose effects can have left -- goes back to the model.

    Only when the question is the same. A turn is served if the resumed request hashes to the
    recorded one, and from the first request at a node and position that does not, or that
    runs past what was recorded, that node and position ask the model. Speculative turns are
    never served: they were guesses, not decisions.

    For each node and position the longest recorded attempt is kept, the latest on a tie. A
    resumed attempt is served the recorded turns before it asks anything, and journals each
    one again, so every attempt's turns begin with those of the attempt before it.
    """

    journal: Journal
    run_id: str
    role: str = "target"
    _turns: dict[tuple[str, int], list[tuple[str, RecordedTurn]]] = field(
        default_factory=dict, init=False
    )
    _next: dict[tuple[str, int], int] = field(default_factory=dict, init=False)
    _asking: set[tuple[str, int]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        requests: dict[str, Mapping[str, JsonValue]] = {}
        attempts: dict[tuple[str, int], dict[str, list[tuple[str, RecordedTurn]]]] = {}
        for entry in self.journal.read(self.run_id, kinds=["model_request", "model_response"]):
            payload = entry.payload
            if payload.get("role") != self.role or payload.get("speculative"):
                continue
            request_id = payload.get("request_id")
            if not isinstance(request_id, str):
                continue
            if entry.kind == "model_request":
                requests[request_id] = payload
                continue
            request = requests.get(request_id)
            step = payload.get("step")
            response = payload.get("response")
            if (
                request is None
                or not isinstance(step, int)
                or isinstance(step, bool)
                or not isinstance(response, Mapping)
            ):
                continue
            # The entry a turn was first recorded in, however many times it has been served.
            origin = payload.get("recorded_from")
            offset = origin if isinstance(origin, int) else entry.offset
            key = (str(request.get("node_id") or ""), step)
            attempt = attempts.setdefault(key, {}).setdefault(str(payload.get("branch_id")), [])
            turn = RecordedTurn(response=response_from_json(response), offset=offset)
            attempt.append((str(payload.get("request_hash", "")), turn))
        for key, by_attempt in attempts.items():
            kept: list[tuple[str, RecordedTurn]] = []
            for turns in by_attempt.values():
                if len(turns) >= len(kept):
                    kept = turns
            self._turns[key] = kept

    def take(self, digest: str, scope: CallScope) -> RecordedTurn | None:
        if scope.run_id != self.run_id:
            return None
        key = (scope.node_id, scope.step)
        if key in self._asking:
            return None
        turns = self._turns.get(key, [])
        index = self._next.get(key, 0)
        if index >= len(turns) or turns[index][0] != digest:
            self._asking.add(key)
            return None
        self._next[key] = index + 1
        return turns[index][1]

    @property
    def recorded(self) -> int:
        """How many turns the journal holds for this run, over every node and position."""
        return sum(len(turns) for turns in self._turns.values())


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
class OpenGroup:
    """A Parallel group that some but not all of its lanes had retired from.

    Everything a resume needs to finish the group as the group it was: which lanes it named,
    under which node ids, from which position and which committed state they forked -- so a
    lane re-run after a crash derives the keys it derived before -- and what the lanes that did
    retire already wrote, so a clash with them is still refused.
    """

    group_id: str
    #: ``(name, path, node_id)`` for every lane, in the order the router named them.
    lanes: tuple[tuple[str, tuple[str, ...], str], ...]
    #: Node ids of the lanes that retired.
    retired: frozenset[str]
    fork_cursor: StepCursor
    #: Committed state when the group forked: what every lane saw, the first time.
    base_state: Mapping[str, JsonValue]
    #: The furthest position a retired lane reached. The group's cursor is past it.
    max_step: int
    #: State keys the retired lanes wrote, by the node id that wrote them.
    claimed: Mapping[str, str]


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
    #: A Parallel group the run was in the middle of, if any. A resume finishes it rather than
    #: asking the router again.
    open_group: OpenGroup | None = None

    @property
    def exists(self) -> bool:
        """Whether the journal has any record of this run at all.

        A run id nobody has seen produces a Recovery with ``last_offset == -1`` and empty
        everything -- and ``resumable`` answered True for it, because an unstarted run is
        certainly "not finished". ``Scheduler.resume`` then drove the graph from empty state
        and dispatched every write the workload contains. Since the run id is new, every
        idempotency key is new, so the dedupe table -- the thing that makes a resume safe --
        had nothing to match against. A typo in a run id sent real writes.
        """
        return self.last_offset >= 0

    @property
    def resumable(self) -> bool:
        return self.exists and (not self.finished or bool(self.confirmed_not_retired))


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
    # Parallel groups: each one's journaled decision, and which of its lanes retired.
    groups: dict[str, _GroupRecord] = {}
    last_group: str | None = None
    lane_of: dict[str, tuple[str, str]] = {}
    node_of: dict[str, str] = {}
    touched_by: dict[str, list[str]] = {}
    # The state the run started from. Every journaled delta was taken on top of it, so the
    # rebuild has to start there too: starting from nothing dropped the run's inputs from every
    # resumed run -- a node reading one failed -- and a delta that changed an input key had
    # nothing to apply to.
    inputs: dict[str, JsonValue] = {}
    started = False

    for entry in journal.read(run_id):
        last_offset = entry.offset
        payload = entry.payload
        if entry.kind == "run_started" and not started:
            started = True
            raw = payload.get("inputs")
            inputs = dict(raw) if isinstance(raw, Mapping) else {}
        elif entry.kind == "run_finished":
            finished = True
        elif entry.kind == "group_forked":
            group_id = payload.get("group_id")
            if isinstance(group_id, str):
                groups[group_id] = _GroupRecord.read(payload, entry.offset)
                last_group = group_id
        elif entry.kind == "branch_forked":
            group_id = payload.get("group_id")
            branch_id = payload.get("branch_id")
            if isinstance(branch_id, str) and payload.get("node_id"):
                node_of[branch_id] = str(payload["node_id"])
            if isinstance(group_id, str) and isinstance(branch_id, str):
                lane_of[branch_id] = (group_id, str(payload.get("node_id") or ""))
        elif entry.kind == "branch_resolved":
            branch_id = payload.get("branch_id")
            status = payload.get("status")
            if isinstance(branch_id, str):
                if status == "retired":
                    retired.add(branch_id)
                    confirmed.discard(branch_id)
                    # An earlier attempt at the same node -- the one a crash interrupted mid-drain,
                    # re-run by a resume under the same node id -- is superseded, not in flight.
                    node_id = node_of.get(branch_id)
                    if node_id is not None:
                        confirmed -= {b for b in confirmed if node_of.get(b) == node_id}
                    cursor = _cursor_from(payload.get("cursor_after"), cursor)
                    if branch_id in lane_of:
                        group_id, node_id = lane_of[branch_id]
                        if group_id in groups:
                            groups[group_id].retired[node_id] = branch_id
                            groups[group_id].steps.append(cursor.step_index)
                elif status == "confirmed":
                    confirmed.add(branch_id)
        elif entry.kind == "state_delta_applied":
            branch_id = payload.get("branch_id")
            patch = payload.get("patch")
            if isinstance(branch_id, str) and isinstance(patch, Sequence):
                deltas.append((entry.offset, branch_id, patch))
            touched = payload.get("touched_keys")
            if isinstance(branch_id, str) and isinstance(touched, Sequence):
                touched_by[branch_id] = [str(key) for key in touched]

    def state_before(offset: int | None) -> JsonValue:
        state: JsonValue = dict(inputs)
        for delta_offset, branch_id, patch in deltas:
            if branch_id not in retired or (offset is not None and delta_offset >= offset):
                continue
            operations = [operation_from_json(op) for op in patch if isinstance(op, Mapping)]
            state = apply(state, operations)
        return state

    state = state_before(None)
    open_group: OpenGroup | None = None
    record = groups.get(last_group) if last_group is not None else None
    if record is not None and set(record.retired) != {lane[2] for lane in record.lanes}:
        base = state_before(record.offset)
        max_step = max([record.fork_cursor.step_index, *record.steps])
        open_group = OpenGroup(
            group_id=record.group_id,
            lanes=tuple(record.lanes),
            retired=frozenset(record.retired),
            fork_cursor=record.fork_cursor,
            base_state=dict(base) if isinstance(base, Mapping) else {},
            max_step=max_step,
            claimed={
                key: node_id
                for node_id, branch_id in record.retired.items()
                for key in touched_by.get(branch_id, ())
            },
        )
        # Where the run is, for the status command: inside the group, past its retired lanes.
        cursor = StepCursor(step_index=max_step, visits=record.fork_cursor.visits)

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
        open_group=open_group,
    )


@dataclass
class _GroupRecord:
    """A ``group_forked`` entry as recovery reads it, and what happened to its lanes since."""

    group_id: str
    lanes: list[tuple[str, tuple[str, ...], str]]
    fork_cursor: StepCursor
    offset: int
    #: Node id -> the branch that retired it.
    retired: dict[str, str] = field(default_factory=dict)
    steps: list[int] = field(default_factory=list)

    @classmethod
    def read(cls, payload: Mapping[str, JsonValue], offset: int) -> _GroupRecord:
        lanes: list[tuple[str, tuple[str, ...], str]] = []
        raw_lanes = payload.get("lanes")
        if isinstance(raw_lanes, Sequence) and not isinstance(raw_lanes, str):
            for lane in raw_lanes:
                if not isinstance(lane, Mapping):
                    continue
                raw_path = lane.get("path")
                path = (
                    tuple(str(part) for part in raw_path)
                    if isinstance(raw_path, Sequence) and not isinstance(raw_path, str)
                    else ()
                )
                lanes.append((str(lane.get("name") or ""), path, str(lane.get("node_id") or "")))
        return cls(
            group_id=str(payload.get("group_id")),
            lanes=lanes,
            fork_cursor=_cursor_from(payload.get("fork_cursor"), StepCursor()),
            offset=offset,
        )


# -- the context fold (Hard Rule 13's rebuild) --------------------------------------------------


def fold_context(
    journal: Journal, run_id: str, lineage: Sequence[str], upto_step: int
) -> list[Message]:
    """Rebuild the message list as of ``upto_step``, from journal entries alone.

    A pure function over the journal, and the purity is the point. Hard Rule 13's check
    rebuilds each prompt a branch recorded and compares hashes; if the rebuild read the
    *branch's own* message list instead, it would compare that list to itself and pass every
    time. That is by far the most likely way to ship Rule 13 dead, and no other test would
    notice -- the leak, equivalence and kill tests all stay green while the check reports zero
    divergences forever.

    Entries are admitted by :func:`attested_origins`: the retired chain unioned with this
    branch's own lineage. An entry from a squashed, stalled or still-unresolved branch is in
    neither set and is evidence for the ledger rather than an input to anything.
    """
    origins = attested_origins(journal, run_id, lineage, upto_step)
    messages: list[Message] = []
    #: Result slots for the most recent assistant turn, by ordinal. Preallocated when the turn
    #: is read and filled by ordinal, never appended on completion -- program order is
    #: structural, and appending in completion order is the bug task 3.8 plants.
    pending: list[ToolResultBlock | None] = []
    pending_ids: list[str] = []

    def flush() -> None:
        if not pending:
            return
        filled = [block for block in pending if block is not None]
        if filled:
            messages.append(Message(role="user", content=tuple(filled)))
        pending.clear()
        pending_ids.clear()

    for entry in journal.read(run_id):
        payload = entry.payload
        branch_id = entry.branch_id
        if entry.kind == "run_started":
            inputs = payload.get("inputs")
            if isinstance(inputs, Mapping) and inputs:
                messages.append(
                    Message(role="user", content=(TextBlock(text=_canonical_text(inputs)),))
                )
            continue

        if branch_id is not None and branch_id not in origins:
            continue
        step = payload.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step > upto_step:
            continue

        if entry.kind == "model_response" and payload.get("end_of_turn"):
            if payload.get("role", "target") != "target":
                continue
            flush()
            response = payload.get("response")
            if not isinstance(response, Mapping):
                continue
            rebuilt = response_from_json(response)
            messages.append(
                Message(
                    role="assistant",
                    content=rebuilt.content,
                    origin_branch=branch_id or "",
                )
            )
            pending.extend([None] * len(rebuilt.tool_uses))
            pending_ids.extend(block.id for block in rebuilt.tool_uses)
        elif entry.kind == "tool_result":
            # NOTE: ``tool_result`` and ``effect_staged`` do not carry ``program_order`` --
            # only ``tool_request`` does -- so this is always None and every result slot stays
            # empty. Hard Rule 13's rebuild is not wired (the runtime fails closed instead of
            # comparing), so this is latent rather than harmful, and it is written down here
            # because a reader would otherwise assume the rebuild works.
            ordinal = payload.get("program_order")
            index = ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) else None
            if index is None or index >= len(pending):
                continue
            pending[index] = ToolResultBlock(
                tool_use_id=pending_ids[index],
                content=payload.get("value"),
                is_error=not bool(payload.get("ok", True)),
            )
        elif entry.kind == "effect_staged":
            # A staged effect's slot never fills: the write has not happened, and the branch
            # cannot make it happen before it retires. A turn that would have to include this
            # slot is refused by Hard Rule 13's gate rather than shown a synthetic value.
            ordinal = payload.get("program_order")
            index = ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) else None
            if index is not None and index < len(pending):
                pending[index] = None

    flush()
    return messages


def _canonical_text(value: JsonValue) -> str:
    from specunode.canonical import canonical

    return canonical(value).decode("utf-8")
