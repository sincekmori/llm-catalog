# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Catalog resolution: roles, keys, settings merge, key/header laziness."""

from typing import Any

import pytest

from llm_catalog.core import Catalog, ResolutionError


def test_resolve_gateway_role(config_dict: dict[str, Any]) -> None:
    rm = Catalog(config_dict).resolve_role("search")
    assert rm.kind == "gateway"
    assert rm.backend == "google"
    assert rm.vendor == "google"
    assert rm.slug == "search-google"
    assert rm.path_template == "gemini/{slug}:{action}"
    assert rm.action_map == {"streamGenerateContent": "customStreamGenerateContent"}
    assert rm.base_url == "https://gateway.example.invalid/base"


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


def test_gateway_api_key_defaults_to_gateway_env(
    config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # No apiKey at all -> AI_GATEWAY_API_KEY is read (matching ai-sdk-catalog).
    del config_dict["providers"][0]["gateway"]["apiKey"]
    rm = Catalog(config_dict).resolve_role("fast")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-default")
    assert rm.api_key() == "gw-default"


# --- headers / query ----------------------------------------------------------


def test_gateway_and_backend_extras_merge(config_dict: dict[str, Any]) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["headers"] = {"Authorization": "Bearer {apiKey}", "x-region": "eu"}
    gw["query"] = {"api-version": "2026-01-01"}
    gw["backends"]["openai"]["headers"] = {"x-region": "us"}
    gw["backends"]["openai"]["query"] = {"api-version": "2026-07-01"}

    cat = Catalog(config_dict)
    fast = cat.resolve_role("fast")  # openai backend: backend wins per name
    assert fast.headers == {"Authorization": "Bearer {apiKey}", "x-region": "us"}
    assert fast.query == {"api-version": "2026-07-01"}
    reasoning = cat.resolve_role("reasoning")  # anthropic backend: gateway-level only
    assert reasoning.headers == {"Authorization": "Bearer {apiKey}", "x-region": "eu"}
    assert reasoning.query == {"api-version": "2026-01-01"}


def test_resolved_headers_substitutes_api_key_and_env(
    config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["headers"] = {
        "Authorization": "Bearer {apiKey}",
        "Ocp-Apim-Subscription-Key": {"envVarName": "SUBSCRIPTION_KEY"},
    }
    rm = Catalog(config_dict).resolve_role("fast")
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "secret-123")
    monkeypatch.delenv("SUBSCRIPTION_KEY", raising=False)
    with pytest.raises(KeyError, match="SUBSCRIPTION_KEY"):
        rm.resolved_headers()
    monkeypatch.setenv("SUBSCRIPTION_KEY", "sub-456")
    assert rm.resolved_headers() == {
        "Authorization": "Bearer secret-123",
        "Ocp-Apim-Subscription-Key": "sub-456",
    }


# --- direct providers -----------------------------------------------------------


def test_resolve_direct_role(direct_config_dict: dict[str, Any]) -> None:
    cat = Catalog(direct_config_dict)
    chat = cat.resolve_role("chat")
    assert chat.kind == "direct"
    assert chat.vendor == "anthropic"  # defaults to the provider id
    assert chat.base_url is None
    assert chat.path_template is None
    assert chat.slug is None
    assert chat.api_key() is None  # the vendor SDK's own default applies

    fast = cat.resolve_role("fast")
    assert fast.vendor == "openai"  # string shorthand
    assert fast.api == "chat"
    assert fast.settings["temperature"] == 0.7  # provider-level default

    bulk = cat.resolve_role("bulk")
    assert bulk.vendor == "openai-compatible"
    assert bulk.base_url == "https://api.fireworks.example.invalid/inference/v1"
    assert bulk.name == "fireworks"
    assert bulk.headers == {"x-tenant": "acme"}
    assert bulk.query == {"api-version": "2026-01-01"}


def test_direct_api_key_env_ref(
    direct_config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rm = Catalog(direct_config_dict).resolve_role("bulk")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    assert rm.api_key() == "fw-key"


def test_non_builtin_direct_vendor_resolves(
    direct_config_dict: dict[str, Any],
) -> None:
    # A provider whose id is not a known vendor still validates and resolves;
    # the adapters reject it at use time (mirroring ai-sdk-catalog, where such
    # a provider needs a code-level resolve override).
    direct_config_dict["providers"].append(
        {"id": "bedrock", "models": [{"id": "claude-opus-4-8"}]}
    )
    rm = Catalog(direct_config_dict).resolve_key("bedrock:claude-opus-4-8")
    assert rm.kind == "direct"
    assert rm.vendor == "bedrock"
