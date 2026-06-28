# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Config schema, alias handling, and validation."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_catalog.core import (
    ConfigError,
    ProviderIdCollisionWarning,
    load_config,
    parse_config,
)


def test_parses_camelcase_aliases(config_dict: dict[str, Any]) -> None:
    cfg = parse_config(config_dict)
    gw = cfg.providers[0].gateway
    assert gw.base_url == "https://gateway.example.invalid/base"
    assert gw.api_key_env == "EXAMPLEGW_API_KEY"
    assert gw.backends["anthropic"].path_template == "anthropic/{slug}"
    assert gw.backends["google"].action_map == {
        "streamGenerateContent": "customStreamGenerateContent"
    }


def test_capabilities_defaults(config_dict: dict[str, Any]) -> None:
    cfg = parse_config(config_dict)
    anthropic_model = cfg.providers[0].models[0]
    assert anthropic_model.capabilities.structured_output == "tool"
    # defaults applied for unspecified fields
    assert anthropic_model.capabilities.multi_step_tools is True
    assert anthropic_model.capabilities.grounding == []


def test_unknown_role_target_rejected(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["roles"]["fast"]["model"] = "does-not-exist"
    with pytest.raises(ConfigError, match="unknown model"):
        parse_config(bad)


def test_unknown_role_provider_rejected(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["roles"]["fast"]["provider"] = "nope"
    with pytest.raises(ConfigError, match="unknown provider"):
        parse_config(bad)


def test_backend_not_configured_rejected(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    # remove the google backend but keep a model that points at it
    del bad["providers"][0]["gateway"]["backends"]["google"]
    with pytest.raises(ConfigError, match="is not configured"):
        parse_config(bad)


def test_api_only_allowed_for_openai(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["providers"][0]["models"][0]["api"] = "chat"  # anthropic model
    with pytest.raises(ConfigError, match=r'only.*allowed when backend == "openai"'):
        parse_config(bad)


def test_duplicate_provider_id_rejected(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["providers"].append(copy.deepcopy(bad["providers"][0]))
    with pytest.raises(ConfigError, match="Duplicate provider id"):
        parse_config(bad)


def test_duplicate_model_id_rejected(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["providers"][0]["models"].append(
        copy.deepcopy(bad["providers"][0]["models"][0])
    )
    with pytest.raises(ConfigError, match="Duplicate model id"):
        parse_config(bad)


def test_extra_keys_forbidden(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["providers"][0]["bogus"] = 1
    with pytest.raises(ConfigError):
        parse_config(bad)


def test_provider_id_collision_warns(config_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(config_dict)
    bad["providers"][0]["id"] = "openai"  # collides with a LiteLLM built-in
    # fix the role references so only the collision warning fires
    for role in bad["roles"].values():
        role["provider"] = "openai"
    with pytest.warns(ProviderIdCollisionWarning, match="#23352"):
        parse_config(bad)


def test_load_config_yaml_roundtrip(
    tmp_path: Path, config_dict: dict[str, Any]
) -> None:
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.providers[0].id == "examplegw"


def test_load_config_json(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(config_dict), encoding="utf-8")
    cfg = load_config(p)
    assert set(cfg.roles) == {"fast", "reasoning", "search"}


def test_load_config_error_includes_path(
    tmp_path: Path, config_dict: dict[str, Any]
) -> None:
    bad = copy.deepcopy(config_dict)
    bad["roles"]["fast"]["model"] = "ghost"
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match=str(p.name)):
        load_config(p)


def test_example_file_loads() -> None:
    # The shipped placeholder example must always be valid.
    example = Path(__file__).resolve().parents[3] / "examples" / "catalog.example.yaml"
    cfg = load_config(example)
    assert "fast" in cfg.roles
