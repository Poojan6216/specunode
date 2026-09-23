"""Committed state, the copy-on-write branch fork, the JSON Patch delta, and the context chain.

Four mechanisms live here, and each exists because of the same Hard Rule read from a different
angle.

**The fork is deep, and that is the point (Hard Rule 6).** A branch may read the committed
state as of its fork point and its own staged effects, and nothing else. The way that rule is
broken is never a deliberate cross-branch read; it is a shallow copy. ``dict(parent.state)``
copies the top level and leaves every nested list and dict shared, so a sibling appending to
``state["findings"]`` is visible to the other sibling and to committed state before anything
has retired -- and it stays visible after that sibling is squashed, because there was never a
container to discard. :meth:`CommittedState.fork` and :meth:`BranchState.fork` therefore copy
through, and :class:`BranchState` copies values on write as well, so a node that keeps a
reference to something it stored cannot reach back into the branch's state later.

**The differ is ours because determinism is ours.** The delta a branch applies at retirement is
journaled as an RFC 6902 patch, and a resume re-applies the journaled patches in offset order
and compares the resulting state hash against the recorded one. That comparison is only
meaningful if the same pair of states produces the same patch bytes on every machine, every
interpreter and every run. Third-party differs do not specify their operation order, so a
library upgrade under an existing journal moves ``patch_hash`` and turns a clean resume into a
state divergence -- with nothing in the diff to explain it. So :func:`diff` is hand-written,
emits only ``add``/``remove``/``replace`` (never ``move``/``copy``, whose source-path semantics
are order-sensitive in exactly the way we are trying to avoid), and visits object keys in
sorted order. The full emission order is written down in :func:`diff`.

**Reducers merge sequential writes, never speculative ones.** Two speculative siblings are
mutually exclusive alternatives: at most one of them is right, and combining a decision the
model made with one it did not make is not a merge. The rule is enforced structurally rather
than by a check -- there is no function in this module that accepts two
:class:`BranchState` values, so there is nothing to call. Reduction happens in
:meth:`CommittedState.commit`, which takes one committed state and one patch, which is exactly
what a retiring branch and a resuming journal both have.

**The context chain makes "squashed branches' messages are discarded entirely" need no discard
step.** :class:`ContextChain` is an immutable linked list with structural sharing: forking
takes a reference, appending makes a new head, and a squashed branch's nodes are simply
unreachable from the branch that retires. A shared mutable list would give the same reads on
the happy path and leak a squashed sibling's message into a prompt the moment anything appended
in place, which Hard Rule 13 exists to prevent.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from specunode.canonical import JsonValue, canonical, chash, from_canonical
from specunode.core.model import Message

__all__ = [
    "NAMED_REDUCERS",
    "BranchState",
    "CommitResult",
    "CommittedState",
    "ContextChain",
    "OpKind",
    "Operation",
    "Patch",
    "PatchError",
    "Reducer",
    "ReducerError",
    "StateError",
    "apply",
    "diff",
    "escape_token",
    "operation_from_json",
    "patch_hash",
    "patch_payload",
    "resolve_reducer",
    "resolve_reducers",
]

OpKind: TypeAlias = Literal["add", "remove", "replace"]


class StateError(ValueError):
    """A value cannot be used as committed state."""


class PatchError(ValueError):
    """A patch is malformed, or does not apply to the document it was given."""


class ReducerError(ValueError):
    """A reducer name does not resolve, or was handed values it cannot combine."""


# -- RFC 6902 operations ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Operation:
    """One RFC 6902 operation.

    ``value`` is unused by ``remove`` and is omitted from :meth:`to_json`, so the journaled
    patch is the RFC's own shape rather than ours with a null tacked on -- a resume reading an
    older journal must not have to know which of the two a given release wrote.
    """

    op: OpKind
    path: str
    value: JsonValue = None

    def to_json(self) -> dict[str, JsonValue]:
        if self.op == "remove":
            return {"op": self.op, "path": self.path}
        return {"op": self.op, "path": self.path, "value": self.value}


def operation_from_json(payload: Mapping[str, JsonValue]) -> Operation:
    """Parse one journaled operation. Rejects every op kind this runtime does not emit."""
    op = payload.get("op")
    path = payload.get("path")
    if op not in ("add", "remove", "replace"):
        raise PatchError(
            f"unsupported operation {op!r}: this runtime emits and accepts only "
            "add, remove and replace"
        )
    if not isinstance(path, str):
        raise PatchError(f"operation path must be a string, got {path!r}")
    if op == "remove":
        return Operation(op="remove", path=path)
    if "value" not in payload:
        raise PatchError(f"{op} at {path!r} has no value member")
    return Operation(op=op, path=path, value=payload["value"])


Patch: TypeAlias = Sequence[Operation]


def patch_payload(patch: Patch) -> list[JsonValue]:
    """The patch as it is journaled: a JSON array of operation objects."""
    return [cast("JsonValue", operation.to_json()) for operation in patch]


def patch_hash(patch: Patch) -> str:
    """Content hash of the journaled patch, recorded so a resume can detect a rewritten one."""
    return chash(patch_payload(patch))


def escape_token(token: str) -> str:
    """RFC 6901 escaping. ``~`` first, or ``~1`` would be re-escaped into ``~01``."""
    return token.replace("~", "~0").replace("/", "~1")


def _unescape_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _parse_path(path: str) -> list[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise PatchError(f"JSON Pointer {path!r} must be empty or start with '/'")
    return [_unescape_token(token) for token in path[1:].split("/")]


# -- normalisation and typed equality --------------------------------------------------------


def _normalise(value: JsonValue) -> JsonValue:
    """Canonical form, as a fresh object graph.

    This both validates (a non-JSON value, a NaN or a key pair that collides under NFC raises
    :class:`specunode.canonical.CanonicalError` here rather than at retirement) and severs
    every alias to the caller's containers, which is what the fork depends on.
    """
    return from_canonical(canonical(value))


def _json_equal(left: JsonValue, right: JsonValue) -> bool:
    """Equality as the canonical form sees it, without encoding whole subtrees to compare them.

    Python's ``==`` says ``1 == 1.0`` and ``True == 1``; JSON does not, and neither does a tool
    handed ``{"amount": 1}`` instead of ``{"amount": 1.0}``. Using ``==`` here would suppress a
    real state change from the patch and leave the resumed run with different state from the
    run it is resuming.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        # Both zeros compare equal, matching the canonical form's -0.0 -> 0.0 fold. NaN cannot
        # reach here: _normalise rejects it.
        return left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return False  # an int and a float are different JSON tokens
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    return False


# -- the differ ------------------------------------------------------------------------------


def diff(before: JsonValue, after: JsonValue) -> list[Operation]:
    """The RFC 6902 patch that takes ``before`` to ``after``, deterministically.

    The emission order, which is the whole contract:

    1. depth first, parents before children;
    2. inside an object, keys in sorted order after NFC normalisation -- the same order
       :func:`specunode.canonical.canonical` sorts them in, so the patch and the state hash
       agree about what a key is;
    3. inside an array, the common prefix is recursed into by ascending index, then appends are
       emitted by ascending index, then truncations by *descending* index.

    Point 3's descending removals are not a style choice: ``remove /1`` then ``remove /2`` on a
    three-element array deletes the wrong element and then runs off the end, because the first
    removal reindexes everything after it. A global sort by path -- the obvious way to make the
    output "deterministic" -- produces exactly that patch.

    Arrays are diffed positionally, with no move, copy or longest-common-subsequence matching.
    An insertion at the front therefore rewrites the tail rather than producing one ``add``.
    That is a size cost, paid to keep the output a plain function of the two states: an LCS
    differ needs a tie-break policy between equally short edit scripts, and that policy is one
    more thing that has to stay byte-stable across releases for a journal to remain replayable.
    """
    operations: list[Operation] = []
    _diff("", _normalise(before), _normalise(after), operations)
    return operations


def _diff(path: str, before: JsonValue, after: JsonValue, out: list[Operation]) -> None:
    if _json_equal(before, after):
        return
    if isinstance(before, dict) and isinstance(after, dict):
        left = cast("dict[str, JsonValue]", before)
        right = cast("dict[str, JsonValue]", after)
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}/{escape_token(key)}"
            if key not in right:
                out.append(Operation(op="remove", path=child))
            elif key not in left:
                out.append(Operation(op="add", path=child, value=right[key]))
            else:
                _diff(child, left[key], right[key], out)
        return
    if isinstance(before, list) and isinstance(after, list):
        old = cast("list[JsonValue]", before)
        new = cast("list[JsonValue]", after)
        common = min(len(old), len(new))
        for index in range(common):
            _diff(f"{path}/{index}", old[index], new[index], out)
        for index in range(common, len(new)):
            out.append(Operation(op="add", path=f"{path}/{index}", value=new[index]))
        for index in range(len(old) - 1, common - 1, -1):
            out.append(Operation(op="remove", path=f"{path}/{index}"))
        return
    out.append(Operation(op="replace", path=path, value=after))


# -- applying a patch ------------------------------------------------------------------------


def apply(state: JsonValue, patch: Patch) -> JsonValue:
    """Apply ``patch`` to ``state`` and return the result. ``state`` is never modified.

    The document is normalised on the way in, so applying a patch to a state that reached this
    process by some other route -- read back from the journal, handed over by a node -- gives
    the same result as applying it to the state the patch was computed from.
    """
    document = _normalise(state)
    for index, operation in enumerate(patch):
        document = _apply_one(document, operation, index)
    return document


def _apply_one(document: JsonValue, operation: Operation, index: int) -> JsonValue:
    tokens = _parse_path(operation.path)
    where = f"operation {index} ({operation.op} {operation.path!r})"
    if not tokens:
        if operation.op == "remove":
            raise PatchError(f"{where}: the whole document cannot be removed")
        return _normalise(operation.value)

    parent: JsonValue = document
    for token in tokens[:-1]:
        parent = _descend(parent, token, where)
    last = tokens[-1]

    if isinstance(parent, dict):
        members = cast("dict[str, JsonValue]", parent)
        if operation.op == "remove":
            if last not in members:
                raise PatchError(f"{where}: no member {last!r} to remove")
            del members[last]
        elif operation.op == "replace" and last not in members:
            raise PatchError(f"{where}: no member {last!r} to replace")
        else:
            members[last] = _normalise(operation.value)
        return document

    if isinstance(parent, list):
        items = cast("list[JsonValue]", parent)
        if last == "-":
            if operation.op != "add":
                raise PatchError(f"{where}: '-' addresses the end of an array and only adds")
            items.append(_normalise(operation.value))
            return document
        position = _array_index(last, where)
        limit = len(items) if operation.op == "add" else len(items) - 1
        if position > limit:
            raise PatchError(f"{where}: index {position} is out of range for {len(items)} items")
        if operation.op == "add":
            items.insert(position, _normalise(operation.value))
        elif operation.op == "remove":
            del items[position]
        else:
            items[position] = _normalise(operation.value)
        return document

    raise PatchError(f"{where}: {type(parent).__name__} has no members to address")


def _descend(parent: JsonValue, token: str, where: str) -> JsonValue:
    if isinstance(parent, dict):
        members = cast("dict[str, JsonValue]", parent)
        if token not in members:
            raise PatchError(f"{where}: path segment {token!r} does not exist")
        return members[token]
    if isinstance(parent, list):
        items = cast("list[JsonValue]", parent)
        position = _array_index(token, where)
        if position >= len(items):
            raise PatchError(f"{where}: index {position} is out of range for {len(items)} items")
        return items[position]
    raise PatchError(f"{where}: path segment {token!r} descends into a {type(parent).__name__}")


def _array_index(token: str, where: str) -> int:
    # RFC 6901 forbids leading zeros, so "01" is not 1. Accepting it would let two spellings of
    # one path hash differently while addressing the same element.
    if not token.isdigit() or (len(token) > 1 and token[0] == "0"):
        raise PatchError(f"{where}: {token!r} is not an array index")
    return int(token)


# -- reducers --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reducer:
    """A named way to combine the committed value of one key with an incoming one.

    The name travels with the function because it is journaled in ``state_delta_applied``'s
    ``reducers_applied``: a ledger that says a key was reduced has to say by what.
    """

    name: str
    fn: Callable[[JsonValue, JsonValue], JsonValue]

    def __call__(self, current: JsonValue, incoming: JsonValue) -> JsonValue:
        return self.fn(current, incoming)


def _append(current: JsonValue, incoming: JsonValue) -> JsonValue:
    if current is None:
        base: list[JsonValue] = []
    elif isinstance(current, list):
        base = list(cast("list[JsonValue]", current))
    else:
        raise ReducerError(f"append needs a list or an absent key, found {type(current).__name__}")
    if isinstance(incoming, list):
        base.extend(cast("list[JsonValue]", incoming))
    else:
        base.append(incoming)
    return base


def _last_write(current: JsonValue, incoming: JsonValue) -> JsonValue:
    del current
    return incoming


def _comparable(current: JsonValue, incoming: JsonValue, name: str) -> bool:
    """Whether two values may be ordered. Numbers with numbers, strings with strings, nothing else.

    ``max`` over mixed kinds would silently pick by Python's rules -- and ``True > 0`` is one of
    them -- producing a committed value the developer never wrote and a patch nothing explains.
    """
    kinds = []
    for value in (current, incoming):
        if isinstance(value, bool) or value is None:
            kinds.append("other")
        elif isinstance(value, (int, float)):
            kinds.append("number")
        elif isinstance(value, str):
            kinds.append("string")
        else:
            kinds.append("other")
    if kinds[0] != kinds[1] or kinds[0] == "other":
        raise ReducerError(
            f"{name} cannot order {current!r} and {incoming!r}: both must be numbers, or both "
            "strings"
        )
    return True


def _is_greater(left: JsonValue, right: JsonValue, name: str) -> bool:
    """Whether ``left`` sorts after ``right``, for a pair ``_comparable`` has accepted.

    The comparison happens inside the narrowing rather than after it: returning a narrowed
    pair still hands the caller a union, and comparing a union is exactly the operation that
    silently does the wrong thing when a run mixes a number and a string in one state key.
    """
    _comparable(left, right, name)
    if isinstance(left, str) and isinstance(right, str):
        return left > right
    if isinstance(left, bool) or isinstance(right, bool):
        raise ReducerError(f"{name} does not order booleans")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) > float(right)
    raise ReducerError(f"{name} cannot order {left!r} and {right!r}")


def _max(current: JsonValue, incoming: JsonValue) -> JsonValue:
    if current is None:
        return incoming
    return incoming if _is_greater(incoming, current, "max") else current


def _min(current: JsonValue, incoming: JsonValue) -> JsonValue:
    if current is None:
        return incoming
    return incoming if _is_greater(current, incoming, "min") else current


#: The reducers that ship with the runtime. The set is closed on purpose: every name here is
#: resolvable from the journal alone, so a replay of a run that used one needs nothing but this
#: module.
NAMED_REDUCERS: Mapping[str, Reducer] = {
    "append": Reducer("append", _append),
    "last_write": Reducer("last_write", _last_write),
    "max": Reducer("max", _max),
    "min": Reducer("min", _min),
}


def resolve_reducer(name: str) -> Reducer:
    """Resolve a reducer named in config: one of :data:`NAMED_REDUCERS`, or ``module:function``.

    A ``module:function`` reference is a replay hazard and ``docs/limitations.md`` says so: the
    module can change under an existing journal, and nothing in the journal records what the
    function did, so a replay can produce a different committed state from the run it replays
    while every hash it checks still agrees. The four named reducers carry no such exposure.
    """
    if name in NAMED_REDUCERS:
        return NAMED_REDUCERS[name]
    module_name, separator, attribute = name.partition(":")
    if not separator or not module_name or not attribute:
        raise ReducerError(
            f"reducer {name!r} is not one of {sorted(NAMED_REDUCERS)} and is not a "
            "'module:function' reference"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ReducerError(f"reducer {name!r}: cannot import {module_name!r}: {exc}") from exc
    try:
        fn = getattr(module, attribute)
    except AttributeError as exc:
        raise ReducerError(f"reducer {name!r}: {module_name!r} has no {attribute!r}") from exc
    if not callable(fn):
        raise ReducerError(f"reducer {name!r}: {attribute!r} is not callable")
    return Reducer(name, cast("Callable[[JsonValue, JsonValue], JsonValue]", fn))


def resolve_reducers(declared: Mapping[str, str]) -> dict[str, Reducer]:
    """Resolve a config's ``state.reducers`` table. Raises before the run rather than mid-run."""
    return {key: resolve_reducer(name) for key, name in declared.items()}


# -- committed state and its copy-on-write fork ----------------------------------------------


@dataclass(frozen=True, slots=True)
class CommitResult:
    """What a retirement journals: the new state, the patch that produced it, and the hashes.

    ``patch`` is the *effective* patch -- recomputed after reduction -- so a resume that
    replays journaled patches in offset order lands on the same state this run committed. A
    journaled patch that reduction then overrode would replay to a state no run ever had.
    """

    state: CommittedState
    patch: tuple[Operation, ...]
    reducers_applied: tuple[tuple[str, str], ...]
    base_state_hash: str
    result_state_hash: str

    @property
    def patch_hash(self) -> str:
        return patch_hash(self.patch)


class CommittedState:
    """The run's committed state: a JSON object, immutable from the outside.

    Nothing hands out a live container. :meth:`__getitem__` and :meth:`to_dict` copy, so a node
    that reads a key and mutates what it got has changed its own copy and not the run's -- the
    one path by which committed state could otherwise be edited without a patch, without a
    journal entry, and without anything to replay.
    """

    __slots__ = ("_hash", "_values")

    def __init__(self, values: Mapping[str, JsonValue] | None = None) -> None:
        self._values = _as_state_object(values if values is not None else {})
        self._hash: str | None = None

    @classmethod
    def initial(cls, inputs: JsonValue) -> CommittedState:
        """Seed committed state from a run's inputs.

        Committed state is a dict (section 5). Inputs that are not a JSON object are refused
        here rather than wrapped under an invented key, because a key this module made up would
        appear in every patch and every state hash without appearing in the developer's graph.
        """
        if not isinstance(inputs, Mapping):
            raise StateError(
                f"committed state is a JSON object; run inputs are a {type(inputs).__name__}"
            )
        return cls(inputs)

    @property
    def hash(self) -> str:
        """Content hash, journaled as ``base_state_hash`` / ``result_state_hash``."""
        if self._hash is None:
            self._hash = chash(cast("JsonValue", self._values))
        return self._hash

    def to_dict(self) -> dict[str, JsonValue]:
        """A copy. The caller owns it and cannot reach this state through it."""
        return cast("dict[str, JsonValue]", _normalise(cast("JsonValue", self._values)))

    def keys(self) -> Iterable[str]:
        return tuple(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(self._values))

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def __getitem__(self, key: str) -> JsonValue:
        return _normalise(self._values[key])

    def get(self, key: str, default: JsonValue = None) -> JsonValue:
        if key not in self._values:
            return default
        return self[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommittedState):
            return NotImplemented
        return self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __repr__(self) -> str:
        return f"CommittedState(keys={sorted(self._values)!r}, hash={self.hash[:12]}…)"

    def fork(self) -> BranchState:
        """A branch's working copy. Shares no container with this state or with any sibling."""
        return BranchState(self._values)

    def commit(
        self, patch: Patch, *, reducers: Mapping[str, Reducer] | None = None
    ) -> CommitResult:
        """Apply a retiring branch's delta, running any declared reducer for a touched key.

        This is the only place a reducer runs, and its arguments are one committed state and
        one patch -- which is what a retiring branch has and what a resuming journal has. There
        is deliberately no overload taking two branch states: reducers combine writes from
        *sequential* nodes, and two speculative siblings are alternatives that no reducer could
        legitimately combine.

        A reducer sees ``(committed value, the value the branch's delta produces)``. A key the
        delta *removes* is removed rather than reduced: a removal is not a write to combine
        with, and inventing a combination for it would put a value back that the branch deleted.
        """
        table = reducers or {}
        before = self.to_dict()
        after = cast("dict[str, JsonValue]", apply(cast("JsonValue", before), patch))
        applied: list[tuple[str, str]] = []
        for key in sorted(_touched_keys(patch)):
            reducer = table.get(key)
            if reducer is None or key not in after:
                continue
            merged = reducer(before.get(key), after[key])
            if not _json_equal(merged, after[key]):
                after[key] = _normalise(merged)
            applied.append((key, reducer.name))
        result = CommittedState(after)
        return CommitResult(
            state=result,
            patch=tuple(diff(cast("JsonValue", before), cast("JsonValue", after))),
            reducers_applied=tuple(applied),
            base_state_hash=self.hash,
            result_state_hash=result.hash,
        )

    def commit_values(
        self,
        written: Mapping[str, JsonValue],
        removed: Iterable[str] = (),
        *,
        reducers: Mapping[str, Reducer] | None = None,
    ) -> CommitResult:
        """Commit the value a node wrote for each key it touched, instead of its patch.

        For a node in a Parallel group. Its delta was taken against the state the whole group
        forked from, and the lanes named before it have committed since, so the positions in
        its patch no longer point where they did. Replayed on top of a sibling's commit, an
        ``append`` from ``[]`` duplicated the sibling's item and a ``last_write`` of an object
        produced a value neither lane wrote. By key and by value, a reducer sees what it sees
        for a node run one after another -- the committed value and the value the node wrote --
        which is also how a fan-out's updates are combined in LangGraph. A lane's value for a
        key is taken whole: two lanes editing parts of one object combine only through a
        reducer that knows how to merge them.

        The journaled ``patch`` is the effective one, from the committed state before to the
        state after, so a resume that replays journaled patches lands on this state.
        """
        table = reducers or {}
        before = self.to_dict()
        after = dict(before)
        applied: list[tuple[str, str]] = []
        for key in sorted(removed):
            after.pop(key, None)
        for key in sorted(written):
            value = _normalise(written[key])
            reducer = table.get(key)
            if reducer is None:
                after[key] = value
                continue
            after[key] = _normalise(reducer(before.get(key), value))
            applied.append((key, reducer.name))
        result = CommittedState(after)
        return CommitResult(
            state=result,
            patch=tuple(diff(cast("JsonValue", before), cast("JsonValue", after))),
            reducers_applied=tuple(applied),
            base_state_hash=self.hash,
            result_state_hash=result.hash,
        )


def _touched_keys(patch: Patch) -> set[str]:
    """The top-level state keys a patch reaches. Reducers are declared per state key."""
    keys: set[str] = set()
    for operation in patch:
        tokens = _parse_path(operation.path)
        if tokens:
            keys.add(tokens[0])
    return keys


def _as_state_object(values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    normalised = _normalise(cast("JsonValue", values))
    if not isinstance(normalised, dict):
        raise StateError(f"committed state is a JSON object, got {type(values).__name__}")
    return cast("dict[str, JsonValue]", normalised)


class BranchState(MutableMapping[str, JsonValue]):
    """A branch's copy-on-write working state.

    Constructed only from a snapshot: :meth:`CommittedState.fork` for a branch off the
    canonical path, :meth:`fork` for a child speculating on top of its parent. Both copy
    through, so a child cannot reach a parent's or a sibling's container even by holding onto
    a nested value.

    Writes copy too. ``state["findings"] = my_list`` stores a copy, so a node that keeps
    ``my_list`` and appends to it later -- on another branch, or after this one was squashed --
    changes nothing here. Mutating through the state (``state["findings"].append(...)``) is the
    supported way to edit in place, and it stays inside this branch.

    :meth:`delta` is the patch from the fork point to now. It is computed at retirement, not
    accumulated as the branch runs: a recorded write that a later write overwrites contributes
    nothing to the delta, and a patch of what actually changed is smaller and easier to read in
    the journal than a log of everything that was assigned.
    """

    __slots__ = ("_base", "_values")

    def __init__(self, values: Mapping[str, JsonValue] | None = None) -> None:
        source = values if values is not None else {}
        # Two independent normalisation passes, deliberately: `base` and `_values` must not
        # share a container, or the first in-place mutation would move the baseline with it and
        # delta() would report that nothing changed.
        self._base = _as_state_object(source)
        self._values = _as_state_object(source)

    def fork(self) -> BranchState:
        """A child branch's state. The child's delta is measured from this fork point."""
        return BranchState(self._values)

    def __getitem__(self, key: str) -> JsonValue:
        return self._values[key]

    def __setitem__(self, key: str, value: JsonValue) -> None:
        self._values[key] = _normalise(value)

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"BranchState(keys={sorted(self._values)!r})"

    @property
    def base(self) -> CommittedState:
        """The state as of the fork point. What :meth:`delta` is measured against."""
        return CommittedState(self._base)

    @property
    def hash(self) -> str:
        return chash(cast("JsonValue", self._values))

    def snapshot(self) -> dict[str, JsonValue]:
        """A copy of the branch's current values."""
        return cast("dict[str, JsonValue]", _normalise(cast("JsonValue", self._values)))

    def delta(self) -> list[Operation]:
        """The RFC 6902 patch from the fork point to the branch's current values."""
        return diff(cast("JsonValue", self._base), cast("JsonValue", self._values))


# -- the context chain -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ContextNode:
    """One message and the chain behind it. Immutable, so any number of forks can share it."""

    message: Message
    parent: _ContextNode | None
    length: int


@dataclass(frozen=True, slots=True)
class ContextChain:
    """An immutable, structurally shared message list, backing ``Branch.context``.

    Forking takes a reference; appending makes a new head that points at the old one. Two
    branches forked from the same point share every message behind the fork and can never see
    each other's additions, because neither of them can change a node the other can reach.

    This is what makes section 5's "squashed branches' messages are discarded entirely" need no
    discard step: a squashed branch's nodes are reachable only from a head nothing holds any
    more. The alternative -- one list, forked by copying the reference and appended to in place
    -- reads identically on the happy path and puts a squashed sibling's message into the next
    prompt, which is the Hard Rule 13 fault that is hardest to see afterwards.
    """

    head: _ContextNode | None = None

    @classmethod
    def empty(cls) -> ContextChain:
        return cls(head=None)

    @classmethod
    def of(cls, messages: Iterable[Message]) -> ContextChain:
        chain = cls.empty()
        for message in messages:
            chain = chain.append(message)
        return chain

    def fork(self) -> ContextChain:
        """A branch's view of this context. The identity function, and that is the mechanism.

        Named rather than left implicit so the call site says what it is doing, and so nothing
        is tempted to copy the list "to be safe" -- a copy would be correct and would also make
        every fork cost the whole history.
        """
        return self

    def append(self, message: Message) -> ContextChain:
        """A new chain with ``message`` at the head. This chain is unchanged."""
        return ContextChain(head=_ContextNode(message, self.head, len(self) + 1))

    def extend(self, messages: Iterable[Message]) -> ContextChain:
        chain = self
        for message in messages:
            chain = chain.append(message)
        return chain

    def materialise(self) -> list[Message]:
        """The messages oldest first, as a fresh list for adapters that want one.

        Fresh every call: an adapter that sorts or trims what it is given must not be able to
        reorder the chain, since program order is the property Hard Rule 13 checks.
        """
        out: list[Message] = []
        node = self.head
        while node is not None:
            out.append(node.message)
            node = node.parent
        out.reverse()
        return out

    def __len__(self) -> int:
        return 0 if self.head is None else self.head.length

    def __bool__(self) -> bool:
        return self.head is not None

    def __iter__(self) -> Iterator[Message]:
        return iter(self.materialise())

    @property
    def last(self) -> Message | None:
        return None if self.head is None else self.head.message

    def origins(self) -> frozenset[str]:
        """The branch ids that contributed a message, for the builder's attestation check."""
        return frozenset(message.origin_branch for message in self.materialise())
