"""Configuration: ``./specunode.yaml``, then ``$XDG_CONFIG_HOME/specunode/config.yaml``.

The file is schema-versioned so that a config written for one release fails loudly against a
later one rather than being half-understood.

The part that matters most is ``tools:``. Hard Rule 2 says effect classes are declared and
never inferred, and this table is where a developer declares them for tools they did not
write -- an MCP server whose annotations are wrong or absent, or a third-party adapter.
It takes precedence over everything else, including the server's own annotations, because
the developer running the agent carries the consequences of a wrong class.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from specunode.canonical import JsonValue
from specunode.core.effects import EffectClass, ToolSpec, forward_keys_from_template
from specunode.core.policy import Policy, StaleReadAction

__all__ = [
    "SCHEMA_VERSION",
    "Config",
    "ConfigError",
    "DrafterConfig",
    "JournalConfig",
    "PolicyConfig",
    "StateConfig",
    "TargetConfig",
    "ToolOverride",
    "find_config",
    "load_config",
]

#: Bump when a change would make an older file mean something different.
SCHEMA_VERSION = 1

DEFAULT_CONFIG_NAME = "specunode.yaml"
DEFAULT_JOURNAL_PATH = Path("./.specunode/journal.db")


class ConfigError(ValueError):
    """The configuration file cannot be used as written."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JournalConfig(_Model):
    kind: Literal["sqlite", "postgres"] = "sqlite"
    path: Path = DEFAULT_JOURNAL_PATH
    #: Used when ``kind`` is postgres; read from SPECUNODE_JOURNAL_DSN when absent.
    dsn: str | None = None

    @model_validator(mode="after")
    def _dsn_required_for_postgres(self) -> JournalConfig:
        if self.kind == "postgres" and not (self.dsn or os.environ.get("SPECUNODE_JOURNAL_DSN")):
            raise ValueError(
                "journal.kind is postgres but no dsn was given and SPECUNODE_JOURNAL_DSN is unset"
            )
        return self


class TargetConfig(_Model):
    """The model whose real output resolves every branch (Hard Rule 4)."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    #: Hard Rule 11: requests go to the endpoint the developer named, and nowhere else.
    #: Passed to the adapter by :func:`specunode.runner.build_target`.
    base_url: str | None = None
    max_tokens: int = 4096
    #: ``None`` means "do not send one". The current models reject sampling parameters, so a
    #: default of 0.0 here would put a 400 in front of anyone who read these defaults.
    temperature: float | None = None

    def envelope_defaults(self) -> dict[str, JsonValue]:
        """Defaults for application code that builds its own ``RequestEnvelope``.

        The runtime deliberately does **not** apply these. Hard Rule 13 makes the request the
        unit of identity, so a runtime that quietly rewrote a developer's envelope -- swapping
        the model, capping the tokens -- would make the journal's record of what was asked
        untrue, and a replay would then refuse against a prompt nobody wrote.

        They are here so an application has one place to read them from, and so this method is
        the honest answer to "what reads these fields?" rather than "nothing does".
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


class DrafterConfig(_Model):
    """One predictor. Tier 0 is early-issue, tier 1 is the pattern index, tier 2 is a model."""

    tier: Literal[0, 1, 2]
    #: Tier 1 only.
    index_path: Path | None = None
    order: int = 2
    #: Tier 2 only.
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None

    @model_validator(mode="after")
    def _fields_match_the_tier(self) -> DrafterConfig:
        if self.tier == 1 and not 1 <= self.order <= 3:
            raise ValueError("a tier-1 pattern index is order-k with k in 1..3")
        if self.tier == 2 and not self.model:
            raise ValueError("a tier-2 drafter needs a model")
        if self.tier != 2 and (self.provider or self.model):
            raise ValueError("provider/model belong to a tier-2 drafter")
        return self


class PolicyConfig(_Model):
    speculation: bool = True
    max_inflight_branches: int = 1
    max_speculation_depth: int = 3
    max_wasted_tokens: int = 20_000
    max_speculative_reads: int = 50
    alpha_window: int = 20
    alpha_floor: float | None = None
    stage_irreversible: bool = False
    on_stale_read: StaleReadAction = "squash"

    def to_policy(self) -> Policy:
        return Policy(**self.model_dump())


class ToolOverride(_Model):
    """A declaration for a tool whose own annotations are absent or wrong (Hard Rule 2)."""

    effect: Literal["read", "write", "compensable", "irreversible"]
    idempotent: bool = False
    compensator: str | None = None
    #: Literal text plus ``{args.<name>}``; see forward_keys_from_template.
    forward_keys: str | None = None
    witness: bool = False

    @field_validator("forward_keys")
    @classmethod
    def _template_compiles(cls, value: str | None) -> str | None:
        if value is not None:
            forward_keys_from_template(value)  # raises here rather than mid-run
        return value

    def to_tool_spec(self, name: str) -> ToolSpec:
        return ToolSpec(
            name=name,
            effect=EffectClass(self.effect),
            idempotent=self.idempotent,
            compensator=self.compensator,
            forward_keys=(
                forward_keys_from_template(self.forward_keys) if self.forward_keys else None
            ),
            witness=self.witness,
        )


class StateConfig(_Model):
    """Reducers apply only when *sequential* nodes write the same key.

    Two speculative siblings are never merged: at most one of them is right, and combining a
    decision the model made with one it did not make is not a merge, it is a fabrication.
    """

    reducers: dict[str, str] = Field(default_factory=dict)

    @field_validator("reducers")
    @classmethod
    def _known_reducers(cls, value: dict[str, str]) -> dict[str, str]:
        known = {"append", "last_write", "max", "min"}
        for key, name in value.items():
            if name not in known and ":" not in name:
                raise ValueError(
                    f"reducer {name!r} for state key {key!r} is not one of {sorted(known)} "
                    "and is not a 'module:function' reference"
                )
        return value


class Config(_Model):
    schema_version: int = SCHEMA_VERSION
    #: ``"module:attribute"`` naming a zero-argument callable that returns
    #: ``(graph_adapter, tool_registry)``.
    #:
    #: ``specunode resume`` and ``specunode replay`` re-drive a graph, and a journal does not
    #: contain one -- it records what a graph did, not what it is. Rather than guess, both
    #: commands refuse unless this says where to find it. The ``module:attribute`` form is the
    #: one already used for a custom reducer, so there is one convention rather than two.
    graph: str | None = None
    journal: JournalConfig = Field(default_factory=JournalConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    drafters: tuple[DrafterConfig, ...] = (DrafterConfig(tier=0),)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    tools: dict[str, ToolOverride] = Field(default_factory=dict)
    state: StateConfig = Field(default_factory=StateConfig)

    @field_validator("graph")
    @classmethod
    def _graph_is_a_reference(cls, value: str | None) -> str | None:
        if value is not None and value.count(":") != 1:
            raise ValueError(
                f"graph {value!r} is not a 'module:attribute' reference; it names the callable "
                "that builds the graph and its tool registry"
            )
        return value

    @field_validator("schema_version")
    @classmethod
    def _version_is_understood(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version {value} is not {SCHEMA_VERSION}; this release cannot "
                "read it, and reading it half-way would be worse than refusing"
            )
        return value

    @field_validator("drafters")
    @classmethod
    def _tiers_are_distinct(cls, value: tuple[DrafterConfig, ...]) -> tuple[DrafterConfig, ...]:
        tiers = [d.tier for d in value]
        if len(set(tiers)) != len(tiers):
            raise ValueError(f"each drafter tier may appear once; got {tiers}")
        return value

    def tool_overrides(self) -> Mapping[str, ToolSpec]:
        return {name: override.to_tool_spec(name) for name, override in self.tools.items()}

    def to_policy(self) -> Policy:
        return self.policy.to_policy()


def find_config(start: Path | None = None) -> Path | None:
    """``./specunode.yaml``, then ``$XDG_CONFIG_HOME/specunode/config.yaml``."""
    local = (start or Path.cwd()) / DEFAULT_CONFIG_NAME
    if local.is_file():
        return local
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    candidate = base / "specunode" / "config.yaml"
    return candidate if candidate.is_file() else None


def load_config(path: Path | None = None) -> Config:
    """Load a config, or return defaults when there is no file to load."""
    resolved = path or find_config()
    if resolved is None:
        return Config()
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved} must contain a mapping at the top level")
    try:
        return Config.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"{resolved}: {exc}") from exc
