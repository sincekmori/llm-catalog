# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Config schema, alias handling, and validation (all through Catalog)."""

import copy
from typing import Any

import pytest

from llm_catalog.core import (
    Catalog,
    CatalogConfig,
    ConfigError,
    ProviderIdCollisionWarning,
)


def parse(data: dict[str, Any]) -> CatalogConfig:
    # Catalog validates its own input; there is no separate parse entry point.
    return Catalog(data).config


def test_parses_camelcase_aliases(config_dict: dict[str, Any]) -> None:
    cfg = parse(config_dict)
    gw = cfg.providers[0].gateway
    assert gw.base_url == "https://gateway.example.invalid/base"
    assert gw.api_key_env == "EXAMPLEGW_API_KEY"
    assert gw.backends["anthropic"].path_template == "anthropic/{slug}"
    assert gw.backends["google"].action_map == {
        "streamGenerateContent": "customStreamGenerateContent"
    }


def test_capabilities_camelcase_and_defaults(config_dict: dict[str, Any]) -> None:
    cfg = parse(config_dict)
    anthropic_model = cfg.providers[0].models[0]
    assert anthropic_model.capabilities.structured_output == "tool"
    # defaults applied for unspecified fields
    assert anthropic_model.capabilities.multi_step_tools is True
    assert anthropic_model.capabilities.grounding == []
    google_model = cfg.providers[0].models[2]
    assert google_model.capabilities.multi_step_tools is False


def test_snake_case_also_accepted(config_dict: dict[str, Any]) -> None:
    # populate_by_name: Python-side construction may use snake_case keys.
    config_dict["providers"][0]["models"][0]["capabilities"] = {
        "structured_output": "prompted"
    }
    cfg = parse(config_dict)
    assert cfg.providers[0].models[0].capabilities.structured_output == "prompted"


def test_top_level_schema_key_accepted(config_dict: dict[str, Any]) -> None:
    # Configs may point at schema.json for editor validation.
    config_dict["$schema"] = "./schema.json"
    parse(config_dict)


def test_api_key_env_defaults_to_gateway_var(config_dict: dict[str, Any]) -> None:
    del config_dict["providers"][0]["gateway"]["apiKeyEnvVarName"]
    cfg = parse(config_dict)
    assert cfg.providers[0].gateway.api_key_env == "AI_GATEWAY_API_KEY"


def test_api_key_literal_accepted(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["gateway"]["apiKey"] = "local-dummy"
    cfg = parse(config_dict)
    assert cfg.providers[0].gateway.api_key == "local-dummy"


def test_full_vendor_backend_set_accepted(config_dict: dict[str, Any]) -> None:
    # Every ai-sdk-catalog backend name validates, so a shared file passes even
    # when some backends are only driven from the TypeScript side.
    gw = config_dict["providers"][0]["gateway"]
    gw["backends"]["mistral"] = {"pathTemplate": "mistral/{slug}"}
    config_dict["providers"][0]["models"].append(
        {"id": "m-large", "backend": "mistral"}
    )
    cfg = parse(config_dict)
    assert cfg.providers[0].models[-1].backend == "mistral"


def test_unknown_backend_name_rejected(config_dict: dict[str, Any]) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["backends"]["bogus"] = {"pathTemplate": "x/{slug}"}
    with pytest.raises(ConfigError):
        parse(config_dict)


def test_unknown_role_target_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["roles"]["fast"]["model"] = "does-not-exist"
    with pytest.raises(ConfigError, match="unknown model"):
        parse(config_dict)


def test_unknown_role_provider_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["roles"]["fast"]["provider"] = "nope"
    with pytest.raises(ConfigError, match="unknown provider"):
        parse(config_dict)


def test_backend_not_configured_rejected(config_dict: dict[str, Any]) -> None:
    # remove the google backend but keep a model that points at it
    del config_dict["providers"][0]["gateway"]["backends"]["google"]
    with pytest.raises(ConfigError, match="is not configured"):
        parse(config_dict)


def test_path_template_requires_slug(config_dict: dict[str, Any]) -> None:
    backends = config_dict["providers"][0]["gateway"]["backends"]
    backends["anthropic"]["pathTemplate"] = "anthropic/fixed"
    with pytest.raises(ConfigError, match=r"\{slug\}"):
        parse(config_dict)


def test_google_path_template_requires_action(config_dict: dict[str, Any]) -> None:
    backends = config_dict["providers"][0]["gateway"]["backends"]
    backends["google"]["pathTemplate"] = "gemini/{slug}"
    with pytest.raises(ConfigError, match=r"\{action\}"):
        parse(config_dict)


def test_action_map_only_on_google(config_dict: dict[str, Any]) -> None:
    backends = config_dict["providers"][0]["gateway"]["backends"]
    backends["openai"]["actionMap"] = {"a": "b"}
    with pytest.raises(ConfigError, match="actionMap"):
        parse(config_dict)


def test_backend_name_only_on_openai_compatible(config_dict: dict[str, Any]) -> None:
    backends = config_dict["providers"][0]["gateway"]["backends"]
    backends["openai"]["name"] = "my-namespace"
    with pytest.raises(ConfigError, match='"name"'):
        parse(config_dict)


def test_duplicate_provider_id_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["providers"].append(copy.deepcopy(config_dict["providers"][0]))
    with pytest.raises(ConfigError, match="Duplicate provider id"):
        parse(config_dict)


def test_duplicate_model_id_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["models"].append(
        copy.deepcopy(config_dict["providers"][0]["models"][0])
    )
    with pytest.raises(ConfigError, match="Duplicate model id"):
        parse(config_dict)


def test_extra_keys_forbidden(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["bogus"] = 1
    with pytest.raises(ConfigError):
        parse(config_dict)


def test_all_issues_reported_at_once(config_dict: dict[str, Any]) -> None:
    # Invariant violations are collected, not raised one at a time.
    config_dict["roles"]["fast"]["model"] = "ghost"
    config_dict["providers"][0]["models"].append(
        copy.deepcopy(config_dict["providers"][0]["models"][0])
    )
    with pytest.raises(ConfigError) as excinfo:
        parse(config_dict)
    message = str(excinfo.value)
    assert "ghost" in message
    assert "Duplicate model id" in message


def test_provider_id_collision_warns(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["id"] = "openai"  # collides with a LiteLLM built-in
    # fix the role references so only the collision warning fires
    for role in config_dict["roles"].values():
        role["provider"] = "openai"
    with pytest.warns(ProviderIdCollisionWarning, match="#23352"):
        parse(config_dict)


def test_validated_config_passes_through(config_dict: dict[str, Any]) -> None:
    cfg = parse(config_dict)
    assert Catalog(cfg).config is cfg
