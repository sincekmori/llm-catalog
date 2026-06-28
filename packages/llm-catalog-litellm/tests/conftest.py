# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the LiteLLM adapter tests."""

from typing import Any

import pytest

from llm_catalog.core import Catalog, parse_config

BASE = "https://gateway.example.invalid/base"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "test-key")


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return {
        "providers": [
            {
                "id": "examplegw",
                "gateway": {
                    "baseURL": BASE,
                    "apiKeyEnvVarName": "EXAMPLEGW_API_KEY",
                    "backends": {
                        "anthropic": {"pathTemplate": "anthropic/{slug}"},
                        "openai": {"pathTemplate": "gpt/{slug}"},
                    },
                },
                "models": [
                    {"id": "light-anthropic", "backend": "anthropic"},
                    {
                        "id": "light-openai",
                        "backend": "openai",
                        "slug": "oai-light",
                        "api": "chat",
                    },
                ],
            }
        ],
        "roles": {
            "fast": {"provider": "examplegw", "model": "light-openai"},
            "reasoning": {"provider": "examplegw", "model": "light-anthropic"},
        },
    }


@pytest.fixture
def registered(config_dict: dict[str, Any]):
    """Wire the module-level handler to a test catalog and register it.

    Exercises the real in-process path: set config -> register() ->
    litellm.completion("examplegw/...").
    """
    import llm_catalog.litellm as mod

    mod.handler.set_catalog(Catalog(parse_config(config_dict)))
    mod._registered.clear()
    mod.register()
    return mod
