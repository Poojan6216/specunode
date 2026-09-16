"""Effect classes and the tool registry (spec task 1.1, Hard Rule 2).

The rule these tests defend is that no effect class is ever *inferred*. Everything here is
about what happens at the edges of that rule: the undeclared tool, the server that ships no
annotations, and the developer who knows the server is wrong.
"""

from __future__ import annotations

import logging

import pytest

from specunode.canonical import JsonValue
from specunode.core.effects import (
    EffectClass,
    ToolRegistry,
    ToolSpec,
    UnknownTool,
    forward_keys_from_template,
)


async def _noop(**kwargs: JsonValue) -> JsonValue:
    return None


def test_an_undeclared_tool_is_a_write_never_a_read() -> None:
    """Defaulting the other way would let an unknown tool be speculated on."""
    registry = ToolRegistry()
    spec = registry.get("something_nobody_declared")
    assert spec.effect is EffectClass.WRITE
    assert spec.idempotent is False
    assert spec.synthesised is True


def test_getting_an_unknown_tool_never_raises_but_running_one_does() -> None:
    registry = ToolRegistry()
    spec = registry.get("mystery")  # classification must work, so the call can be staged
    assert spec.effect.mutates
    with pytest.raises(UnknownTool):
        import asyncio

        asyncio.run(spec.fn())


def test_the_undeclared_warning_fires_once_per_tool(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolRegistry()
    with caplog.at_level(logging.WARNING, logger="specunode.effects"):
        for _ in range(5):
            registry.get("noisy")
        registry.get("other")
    messages = [r for r in caplog.records if "no declared effect class" in r.getMessage()]
    assert len(messages) == 2


def test_a_declared_tool_keeps_its_declaration() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", effect=EffectClass.READ, fn=_noop, witness=True))
    spec = registry.get("lookup")
    assert spec.effect is EffectClass.READ
    assert spec.synthesised is False


@pytest.mark.parametrize(
    ("annotations", "expected_effect", "expected_idempotent"),
    [
        ({"readOnlyHint": True}, EffectClass.READ, False),
        ({"destructiveHint": True}, EffectClass.IRREVERSIBLE, False),
        ({"destructiveHint": False, "idempotentHint": True}, EffectClass.WRITE, True),
        ({"destructiveHint": False}, EffectClass.WRITE, False),
        ({}, EffectClass.WRITE, False),
        (None, EffectClass.WRITE, False),
        # readOnlyHint wins over a contradictory destructiveHint: the table is ordered.
        ({"readOnlyHint": True, "destructiveHint": True}, EffectClass.READ, False),
    ],
)
def test_mcp_annotation_mapping(
    annotations: dict[str, JsonValue] | None,
    expected_effect: EffectClass,
    expected_idempotent: bool,
) -> None:
    spec = ToolRegistry.effect_from_mcp_annotations(annotations)
    assert spec.effect is expected_effect
    assert spec.idempotent is expected_idempotent


def test_a_server_with_no_annotations_yields_all_writes() -> None:
    """Decision Gate D5: this is the case where the override table is the only way to speed up."""
    registry = ToolRegistry.from_mcp_tools(
        [{"name": "read_file"}, {"name": "write_file"}, {"name": "search"}]
    )
    assert {name: registry.get(name).effect for name in registry.names()} == {
        "read_file": EffectClass.WRITE,
        "write_file": EffectClass.WRITE,
        "search": EffectClass.WRITE,
    }


def test_config_overrides_beat_server_annotations() -> None:
    """The developer running the agent carries the consequences, not the server author."""
    registry = ToolRegistry.from_mcp_tools(
        [{"name": "send_email", "annotations": {"readOnlyHint": True}}],
        overrides={"send_email": ToolSpec(name="", effect=EffectClass.IRREVERSIBLE)},
    )
    assert registry.get("send_email").effect is EffectClass.IRREVERSIBLE


def test_an_override_for_a_tool_the_server_did_not_list_is_still_registered() -> None:
    registry = ToolRegistry.from_mcp_tools(
        [{"name": "a"}], overrides={"b": ToolSpec(name="", effect=EffectClass.READ)}
    )
    assert registry.get("b").effect is EffectClass.READ


def test_input_schema_is_carried_through_from_mcp() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    registry = ToolRegistry.from_mcp_tools(
        [{"name": "read_file", "annotations": {"readOnlyHint": True}, "inputSchema": schema}]
    )
    assert registry.get("read_file").schema == schema


# -- declaration consistency ---------------------------------------------------------------


def test_compensable_without_a_compensator_is_refused() -> None:
    with pytest.raises(ValueError, match="names no compensator"):
        ToolSpec(name="reserve", effect=EffectClass.COMPENSABLE)


def test_a_compensator_on_a_non_compensable_tool_is_refused() -> None:
    with pytest.raises(ValueError, match="only COMPENSABLE"):
        ToolSpec(name="charge", effect=EffectClass.WRITE, compensator="refund")


def test_only_a_read_returns_a_witness() -> None:
    with pytest.raises(ValueError, match="only a READ"):
        ToolSpec(name="write_thing", effect=EffectClass.WRITE, witness=True)


@pytest.mark.parametrize(
    ("effect", "mutates"),
    [
        (EffectClass.READ, False),
        (EffectClass.WRITE, True),
        (EffectClass.COMPENSABLE, True),
        (EffectClass.IRREVERSIBLE, True),
    ],
)
def test_everything_but_read_must_never_leave_an_unretired_branch(
    effect: EffectClass, mutates: bool
) -> None:
    assert effect.mutates is mutates


# -- forward_keys templates -----------------------------------------------------------------


def test_forward_keys_template_substitutes_arguments() -> None:
    resolve = forward_keys_from_template("ticket:{args.customer_id}")
    assert resolve({"customer_id": "cus-1"}) == frozenset({"ticket:cus-1"})


def test_forward_keys_template_with_no_placeholders_is_a_constant_key() -> None:
    assert forward_keys_from_template("global:rate_limit")({}) == frozenset({"global:rate_limit"})


def test_a_template_referencing_a_missing_argument_names_no_key() -> None:
    """Naming no key means 'I cannot tell what this touches', which hazards treat as overlap."""
    resolve = forward_keys_from_template("ticket:{args.customer_id}")
    assert resolve({"other": 1}) == frozenset()


def test_a_brace_that_is_not_a_placeholder_is_refused() -> None:
    with pytest.raises(ValueError, match="only supported substitution"):
        forward_keys_from_template("ticket:{customer_id}")


def test_keys_touched_is_none_when_the_tool_did_not_declare() -> None:
    spec = ToolSpec(name="x", effect=EffectClass.WRITE)
    assert spec.keys_touched({"a": 1}) is None
