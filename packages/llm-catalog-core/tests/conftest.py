# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for core tests."""

from typing import Any

import pytest


@pytest.fixture
def config_dict() -> dict[str, Any]:
    """A minimal, valid catalog in its file form (camelCase keys, as JSON)."""
    return {
        "providers": [
            {
                "id": "examplegw",
                "gateway": {
                    "baseURL": "https://gateway.example.invalid/base",
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
                "settings": {"temperature": 0},
                "models": [
                    {
                        "id": "light-anthropic",
                        "backend": "anthropic",
                        "capabilities": {"structuredOutput": "tool"},
                    },
                    {
                        "id": "light-openai",
                        "backend": "openai",
                        "slug": "oai-light",
                        "api": "chat",
                        "settings": {"temperature": 0.5},
                    },
                    {
                        "id": "search-google",
                        "backend": "google",
                        "capabilities": {
                            "multiStepTools": False,
                            "grounding": ["web_search"],
                        },
                    },
                ],
            }
        ],
        "roles": {
            "fast": {"provider": "examplegw", "model": "light-openai"},
            "reasoning": {"provider": "examplegw", "model": "light-anthropic"},
            "search": {"provider": "examplegw", "model": "search-google"},
        },
    }
