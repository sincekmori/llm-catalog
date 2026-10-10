# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the Pydantic AI adapter tests."""

from typing import Any

import httpx2
import pytest

from llm_catalog.core import Catalog
from llm_catalog.pydantic_ai import PydanticAICatalog

BASE = "https://gateway.example.invalid/base"
COMPAT_BASE = "https://compat.example.invalid/v1"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLEGW_API_KEY", "test-key")
    # direct providers fall back to the vendor SDK's own env vars
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-test-key")


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
                        "google": {
                            "vendor": "google",
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
                        # api omitted -> the OpenAI vendor default (Responses API)
                        "id": "resp-openai",
                        "backend": "openai",
                        "capabilities": {"structured_output": "prompted"},
                    },
                    {
                        "id": "search-google",
                        "backend": "google",
                        "capabilities": {"grounding": ["web_search", "code_execution"]},
                    },
                ],
            },
            {
                # direct provider; vendor defaults to the provider id
                "id": "anthropic",
                "models": [{"id": "claude-sonnet-5-5"}],
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
            "reasoning": {"provider": "examplegw", "model": "light-anthropic"},
            "search": "examplegw:search-google",
            "chat": "anthropic:claude-sonnet-5-5",
            "bulk": "fireworks:gpt-oss-120b",
        },
    }


class Wire:
    """Stands in for the network: records each request, answers with ``json``.

    ``transport`` is a ``transport_factory``: the mock sits *inside*
    GatewayTransport, so it observes the final request — after the path
    rewrite, headers, query and hooks — exactly what the gateway (or vendor
    endpoint) would receive.
    """

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.json: dict[str, Any] = {}

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._handler)

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(200, json=self.json)


@pytest.fixture
def wire() -> Wire:
    return Wire()


@pytest.fixture
def pac(config_dict: dict[str, Any], wire: Wire) -> PydanticAICatalog:
    return PydanticAICatalog(Catalog(config_dict), transport_factory=wire.transport)
