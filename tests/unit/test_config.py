"""Configuration loading (spec section 5 and section 7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from specunode.config import SCHEMA_VERSION, Config, ConfigError, find_config, load_config
from specunode.core.effects import EffectClass


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "specunode.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_example_is_valid() -> None:
    """A broken example is worse than none: it is the first thing anyone copies.

    Located through the package rather than the repo root. The template used to live at the
    checkout root and be read via ``parents[2]``, which resolves to ``lib/python3.11/`` from an
    installed wheel -- so it shipped in neither distribution and ``specunode init`` fell back to
    writing eighteen unusable bytes for every user who was not working from source. This test
    passed the whole time, because it ran from source.
    """
    from importlib import resources

    example = Path(str(resources.files("specunode").joinpath("specunode.yaml.example")))
    assert example.is_file(), "the template is not inside the package, so wheels will not carry it"
    config = load_config(example)
    assert config.schema_version == SCHEMA_VERSION
    assert config.tools["send_email"].effect == "irreversible"
    assert config.tool_overrides()["send_email"].effect is EffectClass.IRREVERSIBLE


def test_defaults_apply_when_there_is_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    assert find_config() is None
    config = load_config()
    assert config == Config()
    assert config.policy.max_inflight_branches == 1
    assert config.policy.on_stale_read == "squash"
    assert config.policy.stage_irreversible is False
    assert [d.tier for d in config.drafters] == [0]


def test_an_unreadable_schema_version_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 99\n")
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(path)


def test_unknown_keys_are_refused_rather_than_ignored(tmp_path: Path) -> None:
    """A typo in a safety-relevant setting must not be silently dropped."""
    path = _write(tmp_path, "schema_version: 1\npolicy:\n  stage_irreversable: true\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_a_bad_forward_keys_template_fails_at_load_not_mid_run(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'schema_version: 1\ntools:\n  t:\n    effect: write\n    forward_keys: "x:{oops}"\n',
    )
    with pytest.raises(ConfigError, match="substitution"):
        load_config(path)


def test_a_compensable_override_without_a_compensator_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 1\ntools:\n  t:\n    effect: compensable\n")
    config = load_config(path)
    with pytest.raises(ValueError, match="names no compensator"):
        config.tool_overrides()


def test_duplicate_drafter_tiers_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 1\ndrafters:\n  - tier: 0\n  - tier: 0\n")
    with pytest.raises(ConfigError, match="once"):
        load_config(path)


def test_a_tier_two_drafter_needs_a_model(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 1\ndrafters:\n  - tier: 2\n")
    with pytest.raises(ConfigError, match="model"):
        load_config(path)


def test_postgres_without_a_dsn_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECUNODE_JOURNAL_DSN", raising=False)
    path = _write(tmp_path, "schema_version: 1\njournal:\n  kind: postgres\n")
    with pytest.raises(ConfigError, match="dsn"):
        load_config(path)


def test_an_unknown_reducer_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 1\nstate:\n  reducers:\n    k: shuffle\n")
    with pytest.raises(ConfigError, match="reducer"):
        load_config(path)


def test_a_module_function_reducer_reference_is_allowed(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 1\nstate:\n  reducers:\n    k: my.mod:merge\n")
    assert load_config(path).state.reducers["k"] == "my.mod:merge"


def test_alpha_floor_must_be_a_rate(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: 1\npolicy:\n  alpha_floor: 1.5\n")
    with pytest.raises(ValueError):
        load_config(path).to_policy()


def test_local_file_wins_over_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg"
    (xdg / "specunode").mkdir(parents=True)
    (xdg / "specunode" / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    project = tmp_path / "project"
    project.mkdir()
    assert find_config(project) == xdg / "specunode" / "config.yaml"

    local = _write(project, "schema_version: 1\n")
    assert find_config(project) == local


def test_malformed_yaml_says_which_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: [unclosed\n")
    with pytest.raises(ConfigError, match=r"specunode\.yaml"):
        load_config(path)


def test_every_policy_field_is_configurable_and_nothing_else_is() -> None:
    """The runtime grew three policy knobs the config never learned.

    ``on_unverifiable_read``, ``early_issue`` and ``speculate_writes`` existed on ``Policy`` and
    not on ``PolicyConfig``, and the config model forbids unknown keys, so a YAML file naming
    any of them was rejected rather than honoured. Held to the same field set by introspection,
    so the next knob cannot be added to one side only.
    """
    import dataclasses

    from specunode.config import PolicyConfig
    from specunode.core.policy import Policy

    runtime = {f.name for f in dataclasses.fields(Policy)}
    configurable = set(PolicyConfig.model_fields)
    assert runtime == configurable, (
        f"only in Policy: {sorted(runtime - configurable)}; "
        f"only in the config: {sorted(configurable - runtime)}"
    )
    # And the defaults agree, so an empty ``policy:`` block means the runtime's defaults.
    assert PolicyConfig().to_policy() == Policy()


def test_the_example_config_names_every_policy_field() -> None:
    """The template is what a new user copies, so a knob it omits is a knob nobody finds."""
    import dataclasses
    from importlib import resources

    import yaml

    from specunode.core.policy import Policy

    example = resources.files("specunode").joinpath("specunode.yaml.example").read_text()
    policy = yaml.safe_load(example)["policy"]
    missing = {f.name for f in dataclasses.fields(Policy)} - set(policy) - {"speculation"}
    assert not missing, f"the example config does not mention {sorted(missing)}"


def test_the_example_config_sends_nothing_its_model_rejects() -> None:
    """It set ``temperature: 0.0`` for a model that answers any temperature with a 400, so the
    first real request of everyone who copied it failed."""
    from importlib import resources

    import yaml

    example = resources.files("specunode").joinpath("specunode.yaml.example").read_text()
    target = yaml.safe_load(example)["target"]
    assert str(target["model"]).startswith(("claude-sonnet-5", "claude-opus-5"))
    assert not {"temperature", "top_p", "top_k"} & set(target), "a rejected parameter is set"
    assert target["cache"] is True
    path = Path(str(resources.files("specunode").joinpath("specunode.yaml.example")))
    loaded = load_config(path).target
    assert loaded.temperature is None and loaded.cache is True
