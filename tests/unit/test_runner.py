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


# -- tool overrides --------------------------------------------------------------------------
#
# config.py calls this table "the part that matters most" and runner.py calls it "the supported
# way to say 'this one actually writes' without editing someone else's code". Using it used to
# make the tool permanently uncallable, because the override carries an effect class and no
# implementation, and registering it wholesale replaced the graph's callable with a stub that
# raises UnknownTool -- reporting that the tool "was never registered" to the one person who had
# just registered it.


def _graph_with_a_read() -> tuple[object, ToolRegistry]:
    from specunode.core.effects import EffectClass, ToolSpec

    async def get_ticket(ticket_id: str) -> dict[str, object]:
        return {"ticket_id": ticket_id}

    registry = ToolRegistry()
    registry.register(ToolSpec(name="get_ticket", effect=EffectClass.READ, fn=get_ticket))
    return _MinimalGraph(), registry


class _MinimalGraph:
    def nodes(self) -> list[object]:
        return []

    def run_node(self, node: object, session: object) -> object:
        raise NotImplementedError

    def next(self, state: object) -> object:
        raise NotImplementedError

    def capabilities(self) -> object:
        raise NotImplementedError


def test_an_override_reclassifies_without_destroying_the_implementation() -> None:
    from specunode.config import ToolOverride
    from specunode.core.effects import EffectClass

    config = Config(
        graph="tests.unit.test_runner:_graph_with_a_read",
        tools={"get_ticket": ToolOverride(effect="write")},
    )
    _adapter, registry = build_graph(config)

    spec = registry.get("get_ticket")
    assert spec.effect is EffectClass.WRITE, "the override did not reclassify the tool"
    assert spec.fn.__name__ == "get_ticket", (
        "the override replaced the graph's callable with the unknown-tool stub"
    )


def test_an_override_for_a_tool_the_graph_never_registered_is_refused() -> None:
    """Silently creating an uncallable tool is how the previous behaviour hid itself."""
    from specunode.config import ToolOverride

    config = Config(
        graph="tests.unit.test_runner:_graph_with_a_read",
        tools={"close_ticket": ToolOverride(effect="write")},
    )
    with pytest.raises(RunnerError, match="never registered a tool by that name"):
        build_graph(config)
