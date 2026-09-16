"""Committed state, copy-on-write forks, the JSON Patch differ and reducers (spec task 2.1).

Task 2.1's Verify names three properties, and all three are here: a mutated fork leaves
committed state untouched; committing a branch's delta produces committed-plus-patch; and two
forks of one parent never see each other. The third is the one worth being careful about --
Hard Rule 6 is violated by accident far more often than deliberately, and a lazily shared list
is the usual way it happens.

The differ is held to a stricter standard than "it works", because its output is journaled and
a resume re-applies it and compares a state hash: the same pair of states must produce
byte-identical patch JSON on every machine and every run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specunode.canonical import JsonValue, canonical
from specunode.core.model import Message, TextBlock
from specunode.core.state import (
    NAMED_REDUCERS,
    CommittedState,
    ContextChain,
    PatchError,
    ReducerError,
    apply,
    diff,
    patch_hash,
    patch_payload,
    resolve_reducer,
    resolve_reducers,
)

json_values = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**40), max_value=2**40),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=12),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4), st.dictionaries(st.text(max_size=6), children, max_size=4)
    ),
    max_leaves=18,
)
json_objects = st.dictionaries(st.text(max_size=6), json_values, max_size=5)


# -- the differ -------------------------------------------------------------------------------


@given(json_values, json_values)
@settings(max_examples=1500, deadline=None)
def test_applying_the_diff_reproduces_the_target(before: JsonValue, after: JsonValue) -> None:
    assert canonical(apply(before, diff(before, after))) == canonical(after)


@given(json_values, json_values)
@settings(max_examples=1500, deadline=None)
def test_the_diff_is_deterministic(before: JsonValue, after: JsonValue) -> None:
    """The patch is journaled and a resume re-applies it, so its bytes are part of the record."""
    assert canonical(patch_payload(diff(before, after))) == canonical(
        patch_payload(diff(before, after))
    )


@given(json_values)
@settings(max_examples=500, deadline=None)
def test_no_change_produces_an_empty_patch(value: JsonValue) -> None:
    assert diff(value, value) == []


def test_the_differ_emits_only_add_remove_and_replace() -> None:
    """No move or copy: both are valid RFC 6902 and both make a patch's meaning path-dependent."""
    before = {"a": 1, "b": [1, 2, 3], "c": {"d": True}}
    after = {"b": [3, 2, 1], "c": {"d": False, "e": None}, "f": "new"}
    kinds = {op.op for op in diff(before, after)}
    assert kinds <= {"add", "remove", "replace"}


def test_patch_bytes_are_stable_across_processes() -> None:
    """A resume in a new process must re-derive the same patch hash it journaled."""
    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from specunode.core.state import diff, patch_hash\n"
        "print(patch_hash(diff({'a': 1, 'b': [1, 2]}, {'a': 2, 'b': [1], 'c': None})))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=Path.cwd()
    )
    assert out.stdout.strip() == patch_hash(
        diff({"a": 1, "b": [1, 2]}, {"a": 2, "b": [1], "c": None})
    )


def test_a_path_with_a_slash_or_tilde_round_trips() -> None:
    """RFC 6902 escapes ~ as ~0 and / as ~1; getting it wrong silently targets the wrong key."""
    before: JsonValue = {"a/b": 1, "c~d": 2}
    after: JsonValue = {"a/b": 9, "c~d": 2}
    assert apply(before, diff(before, after)) == after


def test_applying_a_patch_to_the_wrong_document_is_refused() -> None:
    with pytest.raises(PatchError):
        apply({"a": 1}, diff({"b": {"c": 1}}, {"b": {"c": 2}}))


def test_a_patch_round_trips_through_its_journal_form() -> None:
    patch = diff({"a": 1}, {"a": 2, "b": [1]})
    text = json.dumps(patch_payload(patch))
    from specunode.core.state import operation_from_json

    rebuilt = [operation_from_json(entry) for entry in json.loads(text)]
    assert patch_hash(rebuilt) == patch_hash(patch)


# -- copy-on-write forks ------------------------------------------------------------------------


def test_mutating_a_fork_leaves_committed_state_untouched() -> None:
    committed = CommittedState({"findings": ["a"], "count": 1})
    fork = committed.fork()
    fork["count"] = 2
    values = fork["findings"]
    assert isinstance(values, list)
    values.append("b")
    assert committed.to_dict() == {"findings": ["a"], "count": 1}


def test_two_forks_of_one_parent_never_see_each_other() -> None:
    """Hard Rule 6. A shared mutable container is how this is violated by accident."""
    committed = CommittedState({"findings": ["a"], "nested": {"k": [1]}})
    left, right = committed.fork(), committed.fork()

    left["findings"] = [*_as_list(left["findings"]), "left"]
    _as_list(_as_dict(right["nested"])["k"]).append(99)
    right["only_right"] = True

    assert _as_list(left["findings"]) == ["a", "left"]
    assert _as_list(right["findings"]) == ["a"]
    assert _as_list(_as_dict(left["nested"])["k"]) == [1]
    assert "only_right" not in left
    assert committed.to_dict() == {"findings": ["a"], "nested": {"k": [1]}}


def test_committing_a_delta_gives_committed_plus_patch() -> None:
    committed = CommittedState({"count": 1})
    fork = committed.fork()
    fork["count"] = 5
    fork["added"] = "x"

    result = committed.commit(fork.delta())
    assert result.state.to_dict() == {"count": 5, "added": "x"}
    assert result.base_state_hash == committed.hash
    assert result.result_state_hash == result.state.hash
    assert apply(committed.to_dict(), fork.delta()) == result.state.to_dict()


def test_a_forks_delta_is_empty_until_it_changes_something() -> None:
    assert CommittedState({"a": 1}).fork().delta() == []


def test_committed_state_is_hashable_by_content() -> None:
    assert CommittedState({"a": 1, "b": 2}).hash == CommittedState({"b": 2, "a": 1}).hash


def _as_list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _as_dict(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


# -- reducers ------------------------------------------------------------------------------------


def test_the_named_reducers_are_a_closed_set() -> None:
    """Every name must be resolvable from the journal alone, or a replay cannot reproduce a run."""
    assert set(NAMED_REDUCERS) == {"append", "last_write", "max", "min"}


@pytest.mark.parametrize(
    ("name", "current", "incoming", "expected"),
    [
        ("append", ["a"], ["b"], ["a", "b"]),
        ("append", None, ["b"], ["b"]),
        ("last_write", "old", "new", "new"),
        ("max", 3, 7, 7),
        ("max", 7, 3, 7),
        ("min", 3, 7, 3),
        ("max", "a", "b", "b"),
    ],
)
def test_each_reducer_combines_as_named(
    name: str, current: JsonValue, incoming: JsonValue, expected: JsonValue
) -> None:
    assert resolve_reducer(name)(current, incoming) == expected


def test_ordering_reducers_refuse_a_mixed_pair() -> None:
    """Comparing a number with a string is the silent wrong answer a union comparison gives."""
    with pytest.raises(ReducerError):
        resolve_reducer("max")(1, "two")


def test_ordering_reducers_refuse_booleans() -> None:
    with pytest.raises(ReducerError):
        resolve_reducer("max")(True, False)


def test_an_unknown_reducer_name_is_refused() -> None:
    with pytest.raises(ReducerError):
        resolve_reducer("shuffle")


def test_a_reducer_runs_on_commit_for_its_declared_key_only() -> None:
    committed = CommittedState({"findings": ["a"], "summary": "old"})
    fork = committed.fork()
    fork["findings"] = ["b"]
    fork["summary"] = "new"

    result = committed.commit(
        fork.delta(), reducers=resolve_reducers({"findings": "append", "summary": "last_write"})
    )
    assert result.state["findings"] == ["a", "b"]
    assert result.state["summary"] == "new"
    assert dict(result.reducers_applied) == {"findings": "append", "summary": "last_write"}


def test_a_removal_is_not_reduced() -> None:
    """Combining with a removal would put back a value the branch deleted."""
    committed = CommittedState({"findings": ["a"]})
    fork = committed.fork()
    del fork["findings"]
    result = committed.commit(fork.delta(), reducers=resolve_reducers({"findings": "append"}))
    assert "findings" not in result.state


def test_no_public_function_merges_two_branch_states() -> None:
    """Two speculative siblings are alternatives; at most one is right, so no reducer applies.

    Structural, not a convention: if the API cannot express a merge of two branch states, the
    mistake cannot be made. A function taking two BranchState arguments would be that merge.
    """
    import inspect

    from specunode.core import state as module

    offenders: list[str] = []
    for name in module.__all__:
        obj = getattr(module, name)
        target = obj.commit if name == "CommittedState" else obj
        if not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        branch_params = [
            p for p in signature.parameters.values() if "BranchState" in str(p.annotation)
        ]
        if len(branch_params) > 1:
            offenders.append(name)
    assert not offenders, f"{offenders} can combine two speculative branches' states"


# -- the context chain -----------------------------------------------------------------------------


def _msg(text: str, origin: str = "br-1") -> Message:
    return Message(role="user", content=(TextBlock(text=text),), origin_branch=origin)


def test_appending_to_a_fork_does_not_change_the_parent() -> None:
    """Structural sharing is what makes 'a squashed branch's messages are discarded' free."""
    parent = ContextChain.of([_msg("a")])
    child = parent.fork().append(_msg("b"))
    assert [m.content for m in parent.materialise()] == [_msg("a").content]
    assert len(child) == 2
    assert len(parent) == 1


def test_two_children_of_one_chain_are_invisible_to_each_other() -> None:
    parent = ContextChain.of([_msg("a")])
    left = parent.append(_msg("left"))
    right = parent.append(_msg("right"))
    assert len(left) == len(right) == 2
    assert left.materialise()[-1].content != right.materialise()[-1].content


def test_a_chain_materialises_in_order() -> None:
    chain = ContextChain.empty().extend([_msg("a"), _msg("b"), _msg("c")])
    texts = [b.text for m in chain.materialise() for b in m.content if isinstance(b, TextBlock)]
    assert texts == ["a", "b", "c"]


def test_origins_reports_every_branch_that_contributed() -> None:
    """PromptBuilder checks each message's origin against the attested set; this is its input."""
    chain = ContextChain.of([_msg("a", "br-1"), _msg("b", "br-2")])
    assert chain.origins() == frozenset({"br-1", "br-2"})


def test_an_empty_chain_is_falsy_and_has_no_last_message() -> None:
    assert not ContextChain.empty()
    assert ContextChain.empty().last is None
