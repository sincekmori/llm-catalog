# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Catalog resolution: roles, keys, settings merge, key laziness."""

from typing import Any

import pytest

from llm_catalog.core import Catalog, ResolutionError


def test_resolve_role(config_dict: dict[str, Any]) -> None:
    rm = Catalog(config_dict).resolve_role("search")
    assert rm.backend == "google"
    assert rm.slug == "search-google"
    assert rm.path_template == "gemini/{slug}:{action}"
    assert rm.action_map == {"streamGenerateContent": "customStreamGenerateContent"}
    assert rm.base_url == "https://gateway.example.invalid/base"
    assert rm.api_key_env == "EXAMPLEGW_API_KEY"


def test_slug_defaults_to_id_then_override(config_dict: dict[str, Any]) -> None:
    cat = Catalog(config_dict)
    assert cat.resolve_role("reasoning").slug == "light-anthropic"  # default = id
    assert cat.resolve_role("fast").slug == "oai-light"  # explicit slug


def test_settings_merge_model_wins(config_dict: dict[str, Any]) -> None:
    # provider temperature=0, model (light-openai) temperature=0.5 -> model wins.
    cat = Catalog(config_dict)
    assert cat.resolve_role("fast").settings["temperature"] == 0.5
    # provider default flows through when model doesn't override
    assert cat.resolve_role("reasoning").settings["temperature"] == 0


def test_provider_options_merged_per_namespace(config_dict: dict[str, Any]) -> None:
    # Mirrors ai-sdk-catalog: a model adds/overrides individual providerOptions
    # without dropping the provider-level ones in the same namespace.
    provider = config_dict["providers"][0]
    provider["settings"]["providerOptions"] = {
        "openai": {"reasoningEffort": "low", "parallelToolCalls": False}
    }
    provider["models"][1]["settings"]["providerOptions"] = {
        "openai": {"reasoningEffort": "high"}
    }
    rm = Catalog(config_dict).resolve_role("fast")
    assert rm.settings["providerOptions"]["openai"] == {
        "reasoningEffort": "high",
        "parallelToolCalls": False,
    }


def test_resolve_key(config_dict: dict[str, Any]) -> None:
    rm = Catalog(config_dict).resolve_key("examplegw:light-openai")
    assert rm.model_id == "light-openai"
    assert rm.api == "chat"


def test_resolve_key_bad_format(config_dict: dict[str, Any]) -> None:
    with pytest.raises(ResolutionError, match="provider:model_id"):
        Catalog(config_dict).resolve_key("no-colon")


def test_unknown_role(config_dict: dict[str, Any]) -> None:
    with pytest.raises(ResolutionError, match="Unknown role"):
        Catalog(config_dict).resolve_role("ghost")


def test_roles_property(config_dict: dict[str, Any]) -> None:
    assert set(Catalog(config_dict).roles) == {"fast", "reasoning", "search"}


def test_meta_for_role_alias(config_dict: dict[str, Any]) -> None:
    cat = Catalog(config_dict)
    assert cat.meta_for_role("search").capabilities.multi_step_tools is False


def test_api_key_lazy(
    config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rm = Catalog(config_dict).resolve_role("fast")
    monkeypatch.delenv("EXAMPLEGW_API_KEY", raising=False)
    with pytest.raises(KeyError, match="EXAMPLEGW_API_KEY"):
        rm.api_key()
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "secret-123")
    assert rm.api_key() == "secret-123"


def test_api_key_literal_wins_over_env(
    config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dict["providers"][0]["gateway"]["apiKey"] = "local-dummy"
    rm = Catalog(config_dict).resolve_role("fast")
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "env-value")
    assert rm.api_key() == "local-dummy"
