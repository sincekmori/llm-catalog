# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Config schema, alias handling, and validation (all through Catalog)."""

import copy
from typing import Any

import pytest

from llm_catalog.core import Catalog, CatalogConfig, ConfigError, EnvVarRef


def parse(data: dict[str, Any]) -> CatalogConfig:
    # Catalog validates its own input; there is no separate parse entry point.
    return Catalog(data).config


def test_parses_camelcase_aliases(config_dict: dict[str, Any]) -> None:
    cfg = parse(config_dict)
    gw = cfg.providers[0].gateway
    assert gw is not None
    assert gw.base_url == "https://gateway.example.invalid/base"
    assert gw.api_key == EnvVarRef(envVarName="EXAMPLEGW_API_KEY")
    assert gw.backends["anthropic"].path_template == "anthropic/{slug}"
    assert gw.backends["anthropic"].vendor == "anthropic"
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


def test_api_key_env_var_name_removed(config_dict: dict[str, Any]) -> None:
    # The 0.7 schema replaced apiKeyEnvVarName with apiKey: {"envVarName": ...};
    # strict validation makes the removed key fail loudly.
    config_dict["providers"][0]["gateway"]["apiKeyEnvVarName"] = "EXAMPLEGW_API_KEY"
    with pytest.raises(ConfigError, match="apiKeyEnvVarName"):
        parse(config_dict)


def test_api_key_literal_accepted(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["gateway"]["apiKey"] = "local-dummy"
    cfg = parse(config_dict)
    gw = cfg.providers[0].gateway
    assert gw is not None
    assert gw.api_key == "local-dummy"


def test_free_form_backend_keys(config_dict: dict[str, Any]) -> None:
    # Backends live under keys of your choice; the same vendor may appear twice.
    gw = config_dict["providers"][0]["gateway"]
    gw["backends"]["claude-eu"] = {
        "vendor": "anthropic",
        "pathTemplate": "eu/anthropic/{slug}",
    }
    config_dict["providers"][0]["models"].append(
        {"id": "eu-anthropic", "backend": "claude-eu"}
    )
    cfg = parse(config_dict)
    assert cfg.providers[0].models[-1].backend == "claude-eu"


def test_backend_requires_vendor(config_dict: dict[str, Any]) -> None:
    gw = config_dict["providers"][0]["gateway"]
    del gw["backends"]["anthropic"]["vendor"]
    with pytest.raises(ConfigError, match="vendor"):
        parse(config_dict)


def test_unknown_backend_vendor_rejected(config_dict: dict[str, Any]) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["backends"]["bogus"] = {"vendor": "bogus", "pathTemplate": "x/{slug}"}
    with pytest.raises(ConfigError):
        parse(config_dict)


def test_role_shorthand_and_object_form(config_dict: dict[str, Any]) -> None:
    # "provider:model" and {provider, model} are equivalent.
    cfg = parse(config_dict)
    assert cfg.roles["search"] == "examplegw:search-google"
    config_dict["roles"]["search"] = {
        "provider": "examplegw",
        "model": "search-google",
    }
    parse(config_dict)


def test_role_shorthand_without_colon_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["roles"]["search"] = "just-a-model-id"
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


def test_gateway_model_requires_backend(config_dict: dict[str, Any]) -> None:
    del config_dict["providers"][0]["models"][0]["backend"]
    with pytest.raises(ConfigError, match='must set a "backend"'):
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


def test_provider_id_must_not_contain_colon(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["id"] = "example:gw"
    config_dict["roles"] = {}
    with pytest.raises(ConfigError, match='must not contain ":"'):
        parse(config_dict)


def test_extra_keys_forbidden(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["bogus"] = 1
    with pytest.raises(ConfigError):
        parse(config_dict)


def test_unknown_settings_key_rejected(config_dict: dict[str, Any]) -> None:
    # settings are typed and strict, exactly like ai-sdk-catalog's ModelSettings.
    config_dict["providers"][0]["settings"]["temperatur"] = 0
    with pytest.raises(ConfigError, match="temperatur"):
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


def test_validated_config_passes_through(config_dict: dict[str, Any]) -> None:
    cfg = parse(config_dict)
    assert Catalog(cfg).config is cfg


# --- direct providers ---------------------------------------------------------


def test_direct_providers_accepted(direct_config_dict: dict[str, Any]) -> None:
    cfg = parse(direct_config_dict)
    assert cfg.providers[0].vendor is None  # defaults to the provider id
    assert cfg.providers[1].vendor == "openai"  # string shorthand
    block = cfg.providers[2].vendor
    assert not isinstance(block, str)
    assert block is not None
    assert block.id == "openai-compatible"
    assert block.api_key == EnvVarRef(envVarName="FIREWORKS_API_KEY")


def test_vendor_and_gateway_mutually_exclusive(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["vendor"] = "openai"
    with pytest.raises(ConfigError, match="either direct or gateway-routed"):
        parse(config_dict)


def test_direct_model_must_not_set_backend_or_slug(
    direct_config_dict: dict[str, Any],
) -> None:
    direct_config_dict["providers"][0]["models"][0]["backend"] = "anthropic"
    direct_config_dict["providers"][0]["models"][0]["slug"] = "claude"
    with pytest.raises(ConfigError) as excinfo:
        parse(direct_config_dict)
    message = str(excinfo.value)
    assert '"backend"' in message
    assert '"slug"' in message


def test_openai_compatible_requires_base_url(
    direct_config_dict: dict[str, Any],
) -> None:
    del direct_config_dict["providers"][2]["vendor"]["baseURL"]
    with pytest.raises(ConfigError, match="baseURL"):
        parse(direct_config_dict)


def test_api_key_placeholder_requires_api_key(
    direct_config_dict: dict[str, Any],
) -> None:
    block = direct_config_dict["providers"][2]["vendor"]
    del block["apiKey"]
    block["headers"] = {"Authorization": "Bearer {apiKey}"}
    with pytest.raises(ConfigError, match=r"\{apiKey\}"):
        parse(direct_config_dict)


# --- shared-file guarantee ------------------------------------------------------


def test_ai_sdk_catalog_advanced_example_validates(
    advanced_config_dict: dict[str, Any],
) -> None:
    # The whole point of this library: an ai-sdk-catalog config, unmodified.
    cfg = parse(advanced_config_dict)
    assert {p.id for p in cfg.providers} == {"openai", "anthropic", "fireworks", "acme"}
    assert set(cfg.roles) == {"chat", "search", "summarize", "bulk"}
