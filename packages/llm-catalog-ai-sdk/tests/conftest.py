# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the AI SDK for Python adapter tests."""

from typing import Any

import httpx2
import pytest

BASE = "https://gateway.example.invalid/base"
COMPAT_BASE = "https://compat.example.invalid/v1"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "test-key")
    # direct providers fall back to the vendor's own key env vars
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-test-key")
    # `ai` honours these for its direct providers; keep the tests hermetic.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


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
                "settings": {"temperature": 0},
                "models": [
                    {"id": "light-anthropic", "backend": "anthropic"},
                    {
                        "id": "light-openai",
                        "backend": "openai",
                        "slug": "oai-light",
                        "api": "chat",
                        "settings": {
                            "temperature": 0.5,
                            "maxOutputTokens": 256,
                            "topP": 0.9,
                            "seed": 7,
                        },
                    },
                    {
                        # api omitted -> the OpenAI vendor default (Responses API)
                        "id": "resp-openai",
                        "backend": "openai",
                    },
                ],
            },
            {
                # direct provider; vendor defaults to the provider id
                "id": "anthropic",
                "models": [{"id": "claude-sonnet-5-5"}],
            },
            {
                "id": "openai",
                "models": [{"id": "gpt-6-astra"}, {"id": "gpt-6-luna", "api": "chat"}],
            },
            {
                # direct openai-compatible provider with a vendor block
                "id": "fireworks",
                "vendor": {
                    "id": "openai-compatible",
                    "baseURL": COMPAT_BASE,
                    "apiKey": {"envVarName": "FIREWORKS_API_KEY"},
                    "name": "fireworks",
                    "headers": {"x-tenant": "acme"},
                },
                "models": [{"id": "gpt-oss-120b"}],
            },
        ],
        "roles": {
            "fast": {"provider": "examplegw", "model": "light-openai"},
            "respond": "examplegw:resp-openai",
            "reasoning": "examplegw:light-anthropic",
            "chat": "anthropic:claude-sonnet-5-5",
            "direct-responses": "openai:gpt-6-astra",
            "direct-chat": "openai:gpt-6-luna",
            "bulk": "fireworks:gpt-oss-120b",
        },
    }


class Wire:
    """Stands in for the network: records each request, answers with an error.

    ``transport`` is a ``transport_factory``: the mock sits *inside*
    GatewayTransport, so it observes the final request — after the path
    rewrite, headers, query and hooks — exactly what the gateway (or vendor
    endpoint) would receive. A non-2xx reply makes ``ai`` raise before it tries
    to parse a stream, so a test only has to get the request out and inspect it.
    """

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._handler)

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(418, json={"error": {"message": "mock"}})


@pytest.fixture
def wire() -> Wire:
    return Wire()
