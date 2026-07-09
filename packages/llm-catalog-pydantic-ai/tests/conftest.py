# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the Pydantic AI adapter tests."""

from typing import Any

import pytest

from llm_catalog.core import Catalog
from llm_catalog.pydantic_ai import PydanticAICatalog

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
                        "google": {
                            "pathTemplate": "gemini/{slug}:{action}",
                            "actionMap": {
                                "streamGenerateContent": "customStreamGenerateContent"
                            },
                        },
                    },
                },
                "models": [
                    {
                        "id": "light-anthropic",
                        "backend": "anthropic",
                        "capabilities": {"structured_output": "tool"},
                    },
                    {
                        "id": "light-openai",
                        "backend": "openai",
                        "slug": "oai-light",
                        "api": "chat",
                        "capabilities": {"structured_output": "native"},
                    },
                    {
                        "id": "resp-openai",
                        "backend": "openai",
                        "api": "responses",
                        "capabilities": {"structured_output": "prompted"},
                    },
                    {
                        "id": "search-google",
                        "backend": "google",
                        "capabilities": {"grounding": ["web_search", "code_execution"]},
                    },
                ],
            }
        ],
        "roles": {
            "fast": {"provider": "examplegw", "model": "light-openai"},
            "respond": {"provider": "examplegw", "model": "resp-openai"},
            "reasoning": {"provider": "examplegw", "model": "light-anthropic"},
            "search": {"provider": "examplegw", "model": "search-google"},
        },
    }


@pytest.fixture
def pac(config_dict: dict[str, Any]) -> PydanticAICatalog:
    return PydanticAICatalog(Catalog(config_dict))
