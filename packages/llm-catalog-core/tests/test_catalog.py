# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Catalog resolution: roles, keys, settings merge, env-var laziness."""

from typing import Any

import pytest

from llm_catalog.core import Catalog, ResolutionError, parse_config


def _catalog(config_dict: dict[str, Any]) -> Catalog:
    return Catalog(parse_config(config_dict))


def test_resolve_role(config_dict: dict[str, Any]) -> None:
    rm = _catalog(config_dict).resolve_role("search")
    assert rm.backend == "google"
    assert rm.slug == "search-google"
    assert rm.path_template == "gemini/{slug}:{action}"
    assert rm.action_map == {"streamGenerateContent": "customStreamGenerateContent"}
    assert rm.base_url == "https://gateway.example.invalid/base"
    assert rm.api_key_env == "EXAMPLEGW_API_KEY"


def test_slug_defaults_to_id_then_override(config_dict: dict[str, Any]) -> None:
    cat = _catalog(config_dict)
    assert cat.resolve_role("reasoning").slug == "light-anthropic"  # default = id
    assert cat.resolve_role("fast").slug == "oai-light"  # explicit slug


def test_settings_merge_model_wins(config_dict: dict[str, Any]) -> None:
    # provider temperature=0, model (light-openai) temperature=0.5 -> model wins.
    rm = _catalog(config_dict).resolve_role("fast")
    assert rm.settings["temperature"] == 0.5
    # provider default flows through when model doesn't override
    rm2 = _catalog(config_dict).resolve_role("reasoning")
    assert rm2.settings["temperature"] == 0


def test_resolve_key(config_dict: dict[str, Any]) -> None:
    rm = _catalog(config_dict).resolve_key("examplegw:light-openai")
    assert rm.model_id == "light-openai"
    assert rm.api == "chat"


def test_resolve_key_bad_format(config_dict: dict[str, Any]) -> None:
    with pytest.raises(ResolutionError, match="provider:model_id"):
        _catalog(config_dict).resolve_key("no-colon")


def test_unknown_role(config_dict: dict[str, Any]) -> None:
    with pytest.raises(ResolutionError, match="Unknown role"):
        _catalog(config_dict).resolve_role("ghost")


def test_roles_property(config_dict: dict[str, Any]) -> None:
    assert set(_catalog(config_dict).roles) == {"fast", "reasoning", "search"}


def test_meta_for_role_alias(config_dict: dict[str, Any]) -> None:
    cat = _catalog(config_dict)
    assert cat.meta_for_role("search").capabilities.multi_step_tools is False


def test_api_key_lazy(
    config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rm = _catalog(config_dict).resolve_role("fast")
    monkeypatch.delenv("EXAMPLEGW_API_KEY", raising=False)
    with pytest.raises(KeyError, match="EXAMPLEGW_API_KEY"):
        rm.api_key()
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "secret-123")
    assert rm.api_key() == "secret-123"
