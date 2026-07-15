# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for core tests."""

from typing import Any

import pytest


@pytest.fixture
def config_dict() -> dict[str, Any]:
    """A minimal, valid gateway catalog in its file form (camelCase, as JSON)."""
    return {
        "providers": [
            {
                "id": "examplegw",
                "gateway": {
                    "baseURL": "https://gateway.example.invalid/base",
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
                        "google": {
                            "vendor": "google",
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
            "search": "examplegw:search-google",
        },
    }


@pytest.fixture
def direct_config_dict() -> dict[str, Any]:
    """Direct providers only: bare vendor, string shorthand, and a full block."""
    return {
        "providers": [
            {
                # vendor defaults to the provider id
                "id": "anthropic",
                "models": [{"id": "claude-sonnet-5"}],
            },
            {
                # string shorthand
                "id": "oai",
                "vendor": "openai",
                "settings": {"temperature": 0.7},
                "models": [{"id": "gpt-5.6"}, {"id": "gpt-5.6-mini", "api": "chat"}],
            },
            {
                # full vendor block (openai-compatible requires a baseURL)
                "id": "fireworks",
                "vendor": {
                    "id": "openai-compatible",
                    "baseURL": "https://api.fireworks.example.invalid/inference/v1",
                    "apiKey": {"envVarName": "FIREWORKS_API_KEY"},
                    "name": "fireworks",
                    "headers": {"x-tenant": "acme"},
                    "query": {"api-version": "2026-01-01"},
                },
                "models": [{"id": "accounts/fireworks/models/gpt-oss-120b"}],
            },
        ],
        "roles": {
            "chat": "anthropic:claude-sonnet-5",
            "fast": "oai:gpt-5.6-mini",
            "bulk": {
                "provider": "fireworks",
                "model": "accounts/fireworks/models/gpt-oss-120b",
            },
        },
    }


@pytest.fixture
def advanced_config_dict() -> dict[str, Any]:
    """A verbatim copy of ai-sdk-catalog's ``examples/ai-sdk-catalog.advanced.json``.

    The hard requirement of this library is that a config written for
    ai-sdk-catalog works here unmodified; this fixture locks that in.
    (Only the "$schema" pointer is dropped — it is a relative path into the
    other repository.)
    """
    return {
        "providers": [
            {
                "id": "openai",
                "settings": {
                    "temperature": 0.7,
                    "maxOutputTokens": 128000,
                    "providerOptions": {"openai": {"reasoningEffort": "low"}},
                },
                "models": [
                    {"id": "gpt-5.6"},
                    {
                        "id": "gpt-5.6-luna",
                        "settings": {
                            "temperature": 0.2,
                            "providerOptions": {"openai": {"parallelToolCalls": False}},
                        },
                    },
                ],
            },
            {
                "id": "anthropic",
                "models": [{"id": "claude-sonnet-5", "settings": {"temperature": 1}}],
            },
            {
                "id": "fireworks",
                "vendor": {
                    "id": "openai-compatible",
                    "baseURL": "https://api.fireworks.ai/inference/v1",
                    "apiKey": {"envVarName": "FIREWORKS_API_KEY"},
                    "name": "fireworks",
                },
                "models": [{"id": "accounts/fireworks/models/gpt-oss-120b"}],
            },
            {
                "id": "acme",
                "gateway": {
                    "baseURL": "https://gateway.example.com/v1",
                    "apiKey": {"envVarName": "ACME_API_KEY"},
                    "headers": {
                        "Authorization": "Bearer {apiKey}",
                        "Ocp-Apim-Subscription-Key": {
                            "envVarName": "ACME_SUBSCRIPTION_KEY"
                        },
                    },
                    "query": {"api-version": "2026-01-01"},
                    "backends": {
                        "claude-eu": {
                            "vendor": "anthropic",
                            "pathTemplate": "eu/anthropic/{slug}",
                        },
                        "claude-us": {
                            "vendor": "anthropic",
                            "pathTemplate": "us/anthropic/{slug}",
                        },
                        "gemini": {
                            "vendor": "google",
                            "pathTemplate": "google/{slug}:{action}",
                            "actionMap": {
                                "streamGenerateContent": "customStreamGenerateContent"
                            },
                        },
                    },
                },
                "models": [
                    {"id": "claude-opus-4-8", "backend": "claude-eu"},
                    {"id": "gemini-3.5-flash", "backend": "gemini", "slug": "flash"},
                ],
            },
        ],
        "roles": {
            "chat": "anthropic:claude-sonnet-5",
            "search": "acme:gemini-3.5-flash",
            "summarize": "openai:gpt-5.6-luna",
            "bulk": "fireworks:accounts/fireworks/models/gpt-oss-120b",
        },
    }
