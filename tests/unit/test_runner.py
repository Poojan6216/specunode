"""Turning a config into a graph, and refusing clearly when it cannot (spec tasks 2.5, 2.6).

``resume`` and ``replay`` both re-drive a graph, and a journal does not contain one -- it
records what a graph did, not what it is. Everything here is about the refusals, because the
success path is exercised end to end by the CLI tests and the failure path is what a person
actually meets: they wrote a reference, it was wrong, and the message has to say how.
"""

from __future__ import annotations

import pytest

from specunode.config import Config
from specunode.core.effects import ToolRegistry
from specunode.runner import RunnerError, build_graph, build_target, load_reference


def test_a_reference_without_a_colon_is_refused() -> None:
    with pytest.raises(RunnerError, match="module:attribute"):
        load_reference("justamodule")


def test_a_missing_module_names_itself() -> None:
    with pytest.raises(RunnerError, match="no_such_module_anywhere"):
        load_reference("no_such_module_anywhere:build")


def test_a_missing_attribute_names_itself() -> None:
    with pytest.raises(RunnerError, match="no_such_attribute"):
        load_reference("specunode.runner:no_such_attribute")


def test_a_config_without_a_graph_says_what_to_add() -> None:
    """The message is the whole value here: the reader is the person who has to fix it."""
    with pytest.raises(RunnerError, match="graph:"):
        build_graph(Config())


def test_the_config_validates_the_reference_shape_before_anything_imports() -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        Config(graph="not-a-reference")


def test_a_builder_that_is_not_callable_is_refused() -> None:
    with pytest.raises(RunnerError, match="not callable"):
        build_graph(Config(graph="tests.unit.test_runner:NOT_CALLABLE"))


def test_a_builder_returning_the_wrong_shape_is_refused() -> None:
    with pytest.raises(RunnerError, match="must return a"):
        build_graph(Config(graph="tests.unit.test_runner:_returns_nothing_useful"))


def test_a_builder_without_a_tool_registry_is_refused() -> None:
    """Effect classes are declared out of band and never inferred, so this is not optional."""
    with pytest.raises(RunnerError, match="ToolRegistry"):
        build_graph(Config(graph="tests.unit.test_runner:_returns_no_registry"))


def test_a_builder_whose_adapter_is_not_a_graph_is_refused() -> None:
    with pytest.raises(RunnerError, match="GraphAdapter"):
        build_graph(Config(graph="tests.unit.test_runner:_returns_a_bad_adapter"))


def test_an_unwired_provider_is_refused_rather_than_guessed() -> None:
    from specunode.config import TargetConfig

    config = Config(target=TargetConfig(provider="openai"))
    with pytest.raises(RunnerError, match="no adapter in this release"):
        build_target(config)


# -- builders the tests above point at ---------------------------------------------------------

#: A module attribute that resolves fine and is not a function.
NOT_CALLABLE = "graphs are built by callables"


def _returns_nothing_useful() -> int:
    return 7


def _returns_no_registry() -> tuple[object, object]:
    return object(), {}


def _returns_a_bad_adapter() -> tuple[object, ToolRegistry]:
    return object(), ToolRegistry()
