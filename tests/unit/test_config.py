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
    """A broken example is worse than none: it is the first thing anyone copies."""
    example = Path(__file__).resolve().parents[2] / "specunode.yaml.example"
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
