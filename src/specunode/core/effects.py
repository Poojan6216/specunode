"""Effect classes: what a tool does to the world, as declared by the developer.

Hard Rule 2: **effect classes are declared, never inferred.** No heuristic reads a tool's
name, description or arguments to decide whether it writes. No model is asked. The reason is
the one SCOPEGATE makes: exposure is not authority, and a runtime that guesses a tool is safe
because it is called ``get_something`` will eventually guess wrong about the one tool where it
mattered.

The consequences of that rule, in order of how often they bite:

* **A tool with no declaration is a WRITE.** Not a read, not an error -- a write. Defaulting
  the other way would mean an undeclared tool gets speculated on, which is precisely the
  failure the store buffer exists to prevent. :meth:`ToolRegistry.get` therefore never raises
  for an unknown name; it synthesises ``WRITE(idempotent=False)``, warns once, and moves on.
* **A tool whose upstream enqueues, schedules or triggers work asynchronously is a WRITE**,
  regardless of its synchronous response or its HTTP verb. ``{"status": "queued"}`` is not a
  read result. Attack 7.10 measures what happens when someone declares one of these READ.
* **The declaration is the developer's word, and the runtime cannot check it.** A tool
  declared READ that actually writes defeats the store buffer completely (attack 7.1). That
  is the trust boundary, it is documented rather than papered over, and the fake world models
  it by keeping its *true* semantics separate from what is declared here.

MCP annotations are a declaration too -- made by the server author rather than the client
developer -- so :meth:`ToolRegistry.from_mcp_tools` maps them, with the config's per-tool
override table taking precedence over both.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from specunode.canonical import JsonValue

__all__ = [
    "EffectClass",
    "JsonSchema",
    "ToolRegistry",
    "ToolSpec",
    "UnknownTool",
    "forward_keys_from_template",
]

logger = logging.getLogger("specunode.effects")

JsonSchema: TypeAlias = Mapping[str, JsonValue]
ToolFn: TypeAlias = Callable[..., Awaitable[JsonValue]]
ForwardKeys: TypeAlias = Callable[[Mapping[str, JsonValue]], frozenset[str]]
#: Asked, after a crash, whether a call whose outcome was lost took effect upstream. Given the
#: idempotency key the call was sent under and its arguments; returns the upstream's own
#: result if it did, ``None`` if it did not. It must answer from the upstream's record of that
#: key -- a payment's idempotency key, a unique request id, a row keyed by it.
Reconcile: TypeAlias = Callable[[str, Mapping[str, JsonValue]], Awaitable["JsonValue | None"]]


class UnknownTool(RuntimeError):
    """Raised when a tool that was never registered is actually executed.

    Classification of an unknown tool succeeds -- as WRITE, the safe answer -- so that hazard
    analysis and staging work. Running one cannot, because there is nothing to run.
    """


class EffectClass(Enum):
    """What a call does to the world, and therefore when it is allowed to happen."""

    #: Safe to execute speculatively. It still reaches upstream, and the ledger counts that.
    READ = "read"
    #: Staged in the store buffer; dispatched only by ``retire()`` on a CONFIRMED branch.
    WRITE = "write"
    #: A WRITE that declares how to undo itself *after* it has retired. Compensation is a
    #: second effect that reaches the world, which is why it is not the mechanism that makes
    #: speculation safe -- the store buffer is. Vocabulary adopted from SagaLLM.
    COMPENSABLE = "compensable"
    #: A WRITE with no undo at all. By default it is a barrier rather than something staged;
    #: ``policy.stage_irreversible`` can change that, and the docs say why the default is off.
    IRREVERSIBLE = "irreversible"

    @property
    def mutates(self) -> bool:
        """True for everything that must never leave an unretired branch (Hard Rule 3)."""
        return self is not EffectClass.READ


async def _unknown_tool(**kwargs: JsonValue) -> JsonValue:
    raise UnknownTool(
        "this tool was never registered; it was classified as WRITE so that it could be "
        "staged safely, but there is no implementation to run"
    )


@dataclass(frozen=True)
class ToolSpec:
    """A tool, as declared."""

    name: str
    effect: EffectClass
    fn: ToolFn = _unknown_tool
    #: Whether a second delivery of the same call is a no-op upstream. Claimed, not verified.
    idempotent: bool = False
    #: Name of the tool that undoes this one. Required iff ``effect`` is COMPENSABLE.
    compensator: str | None = None
    #: The resource keys a call touches, which is what lets a READ be checked against a
    #: staged WRITE in the same branch. ``None`` means "unknown", which makes any overlap
    #: undecidable and therefore a hazard.
    forward_keys: ForwardKeys | None = None
    #: Whether a READ returns ``{"value": ..., "witness": ...}`` so staleness can be
    #: re-checked at retirement. Reads without one are reported unwitnessed, never fresh.
    witness: bool = False
    schema: JsonSchema | None = None
    #: True when this spec was synthesised for a name nobody declared.
    synthesised: bool = False
    #: How to find out, after a crash, whether a call whose reply was lost took effect. Without
    #: it, a non-idempotent call in that window is dead-lettered for a human, because sending
    #: it again could do it twice and not sending it could skip it.
    reconcile: Reconcile | None = None

    def __post_init__(self) -> None:
        if self.effect is EffectClass.COMPENSABLE and not self.compensator:
            raise ValueError(
                f"tool {self.name!r} is declared COMPENSABLE but names no compensator; "
                "a compensable effect that cannot say how it is undone is just a WRITE"
            )
        if self.compensator and self.effect is not EffectClass.COMPENSABLE:
            raise ValueError(
                f"tool {self.name!r} names a compensator but is declared {self.effect.value}; "
                "only COMPENSABLE effects are compensated"
            )
        if self.witness and self.effect is not EffectClass.READ:
            raise ValueError(f"tool {self.name!r}: only a READ returns a witness")

    def keys_touched(self, args: Mapping[str, JsonValue]) -> frozenset[str] | None:
        """Resource keys this call touches, or ``None`` when the tool did not declare."""
        return self.forward_keys(args) if self.forward_keys is not None else None


def forward_keys_from_template(template: str) -> ForwardKeys:
    """Compile a config template such as ``"ticket:{args.customer_id}"`` into a callable.

    Deliberately tiny: literal text plus ``{args.<name>}`` placeholders, and nothing else.
    A template language with expressions in it would be a place for logic to hide, and the
    result of this function decides whether a read is a hazard.

    A template referencing an argument the call did not pass returns the empty set, which
    means "this call touches no keys I can name" -- and an unnameable key is treated by
    :mod:`specunode.core.hazards` as an overlap, never as an absence of one.
    """
    placeholder = re.compile(r"\{args\.([A-Za-z_][A-Za-z0-9_]*)\}")
    referenced = placeholder.findall(template)
    if not referenced and "{" in template:
        raise ValueError(
            f"forward_keys template {template!r} has a brace but no {{args.<name>}} "
            "placeholder; the only supported substitution is {args.<name>}"
        )

    def resolve(args: Mapping[str, JsonValue]) -> frozenset[str]:
        missing = [name for name in referenced if name not in args]
        if missing:
            return frozenset()
        return frozenset({placeholder.sub(lambda m: str(args[m.group(1)]), template)})

    return resolve


@dataclass
class ToolRegistry:
    """What the runtime believes about each tool."""

    _specs: dict[str, ToolSpec] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set)

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def register_all(self, specs: Iterator[ToolSpec] | list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> frozenset[str]:
        return frozenset(self._specs)

    def declared(self) -> Mapping[str, ToolSpec]:
        return dict(self._specs)

    def get(self, name: str) -> ToolSpec:
        """The declared spec, or a synthesised ``WRITE`` for a name nobody declared.

        Never raises, and never guesses READ. An undeclared tool that got classified as a
        read would be executed speculatively; one classified as a write is merely staged,
        and the worst case is a lost speculation rather than an effect that escaped.
        """
        spec = self._specs.get(name)
        if spec is not None:
            return spec
        if name not in self._warned:
            self._warned.add(name)
            logger.warning(
                "tool %r has no declared effect class; treating it as WRITE(idempotent=False). "
                "Declare it to let the runtime speculate on it (Hard Rule 2).",
                name,
            )
        return ToolSpec(name=name, effect=EffectClass.WRITE, idempotent=False, synthesised=True)

    # -- MCP ------------------------------------------------------------------------------

    @staticmethod
    def effect_from_mcp_annotations(annotations: Mapping[str, JsonValue] | None) -> ToolSpec:
        """Map MCP tool annotations to an effect class, per the table in spec section 5.

        Absent annotations map to ``WRITE(idempotent=False)``. Decision Gate D5: if the
        servers a developer actually uses ship no annotations, everything defaults to WRITE
        and the per-tool override table is the only route to any speculation at all.
        """
        hints = annotations or {}
        read_only = hints.get("readOnlyHint")
        destructive = hints.get("destructiveHint")
        idempotent = hints.get("idempotentHint")

        if read_only is True:
            return ToolSpec(name="", effect=EffectClass.READ)
        if destructive is True:
            return ToolSpec(name="", effect=EffectClass.IRREVERSIBLE)
        if destructive is False and idempotent is True:
            return ToolSpec(name="", effect=EffectClass.WRITE, idempotent=True)
        return ToolSpec(name="", effect=EffectClass.WRITE, idempotent=False)

    @classmethod
    def from_mcp_tools(
        cls,
        tools: list[Mapping[str, JsonValue]],
        overrides: Mapping[str, ToolSpec] | None = None,
    ) -> ToolRegistry:
        """Build a registry from an MCP ``tools/list`` response.

        The config's override table wins over the server's annotations, because the developer
        running the agent carries the consequences of a wrong class, not the server author.
        """
        registry = cls()
        overrides = overrides or {}
        for tool in tools:
            name = tool.get("name")
            if not isinstance(name, str):
                continue
            if name in overrides:
                registry.register(replace(overrides[name], name=name))
                continue
            raw = tool.get("annotations")
            annotations = raw if isinstance(raw, Mapping) else None
            base = cls.effect_from_mcp_annotations(annotations)
            schema = tool.get("inputSchema")
            registry.register(
                replace(
                    base,
                    name=name,
                    schema=schema if isinstance(schema, Mapping) else None,
                )
            )
        for name, spec in overrides.items():
            if name not in registry:
                registry.register(replace(spec, name=name))
        return registry
