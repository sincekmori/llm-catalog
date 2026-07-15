# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the LiteLLM adapter tests."""

from typing import Any

import pytest

from llm_catalog.core import Catalog

BASE = "https://gateway.example.invalid/base"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "test-key")
    # direct providers fall back to the vendor SDK's own env vars
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return {
        "providers": [
            {
                "id": "examplegw",
                "gateway": {
                    "baseURL": BASE,
                    "apiKey": {"envVarName": "EXAMPLEGW_API_KEY"},
                    "backends": {
                        "anthropic": {
                            "vendor": "anthropic",
                            "pathTemplate": "anthropic/{slug}",
                        },
                        "openai": {
                            "vendor": "openai",
                            "pathTemplate": "gpt/{slug}",
                        },
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
            },
            {
                # direct provider; vendor defaults to the provider id ("claude"
                # is not a vendor, so spell it out with the string shorthand)
                "id": "claude-direct",
                "vendor": "anthropic",
                "models": [{"id": "claude-sonnet-5"}],
            },
        ],
        "roles": {
            "fast": {"provider": "examplegw", "model": "light-openai"},
            "reasoning": "examplegw:light-anthropic",
            "chat": "claude-direct:claude-sonnet-5",
        },
    }


@pytest.fixture
def registered(config_dict: dict[str, Any]):
    """Wire the module-level handler to a test catalog and register it.

    Exercises the real in-process path: set config -> register() ->
    litellm.completion("examplegw/...").
    """
    import llm_catalog.litellm as mod

    mod.handler.set_catalog(Catalog(config_dict))
    mod._registered.clear()
    mod.register()
    return mod
