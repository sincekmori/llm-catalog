# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""PydanticAICatalog: model/provider kinds (gateway and direct), output modes,
grounding, and that requests reach the right URL (mock endpoints via respx)."""

import contextlib

import httpx
import pytest
import respx
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from llm_catalog.core import Catalog
from llm_catalog.core.errors import LLMCatalogError
from llm_catalog.pydantic_ai import PydanticAICatalog

BASE = "https://gateway.example.invalid/base"
COMPAT_BASE = "https://compat.example.invalid/v1"

# Minimal vendor responses, just enough for the SDK to parse a reply.
_ANTHROPIC_RESP = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "light-anthropic",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_OPENAI_CHAT_RESP = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "oai-light",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


# --- model / provider kinds -------------------------------------------------


def test_gateway_model_kinds(pac: PydanticAICatalog) -> None:
    assert isinstance(pac.model_for_role("reasoning"), AnthropicModel)
    assert isinstance(pac.model_for_role("fast"), OpenAIChatModel)  # api=chat
    # api omitted -> the OpenAI vendor default is the Responses API (TS parity)
    assert isinstance(pac.model_for_role("respond"), OpenAIResponsesModel)
    assert isinstance(pac.model_for_role("search"), GoogleModel)


def test_direct_model_kinds(pac: PydanticAICatalog) -> None:
    assert isinstance(pac.model_for_role("chat"), AnthropicModel)
    # openai-compatible speaks Chat Completions
    assert isinstance(pac.model_for_role("bulk"), OpenAIChatModel)


def test_model_by_key(pac: PydanticAICatalog) -> None:
    assert isinstance(pac.model("examplegw:light-anthropic"), AnthropicModel)


def test_accepts_plain_config_mapping(config_dict: dict) -> None:
    # The wrapper validates raw config itself, like the core Catalog.
    pac = PydanticAICatalog(config_dict)
    assert isinstance(pac.model_for_role("reasoning"), AnthropicModel)


def test_unsupported_vendor_raises_at_use(config_dict: dict) -> None:
    # A shared config may route some vendors only from the TypeScript side;
    # they validate fine and fail here only when actually used.
    gw = config_dict["providers"][0]["gateway"]
    gw["backends"]["mistral"] = {
        "vendor": "mistral",
        "pathTemplate": "mistral/{slug}",
    }
    config_dict["providers"][0]["models"].append(
        {"id": "m-large", "backend": "mistral"}
    )
    pac = PydanticAICatalog(config_dict)
    with pytest.raises(LLMCatalogError, match="not supported by the Pydantic AI"):
        pac.model("examplegw:m-large")


def test_completion_api_raises(config_dict: dict) -> None:
    config_dict["providers"][0]["models"][1]["api"] = "completion"
    pac = PydanticAICatalog(config_dict)
    with pytest.raises(LLMCatalogError, match="Completions API"):
        pac.model_for_role("fast")


def test_openai_compatible_responses_raises(config_dict: dict) -> None:
    config_dict["providers"][2]["models"][0]["api"] = "responses"
    pac = PydanticAICatalog(config_dict)
    with pytest.raises(LLMCatalogError, match="Chat Completions"):
        pac.model_for_role("bulk")


# --- output modes -----------------------------------------------------------


class _Schema(BaseModel):
    value: int


def test_output_for_modes(pac: PydanticAICatalog) -> None:
    Schema = _Schema
    assert isinstance(pac.output_for("fast", Schema), NativeOutput)  # native
    assert isinstance(pac.output_for("reasoning", Schema), ToolOutput)  # tool
    assert isinstance(pac.output_for("respond", Schema), PromptedOutput)  # prompted


# --- grounding --------------------------------------------------------------


def test_grounding_tools(pac: PydanticAICatalog) -> None:
    tools = pac.grounding_tools("search")
    assert len(tools) == 2  # web_search + code_execution


def test_grounding_unknown_raises(config_dict: dict) -> None:
    config_dict["providers"][0]["models"][3]["capabilities"]["grounding"] = ["bogus"]
    bad = PydanticAICatalog(Catalog(config_dict))
    with pytest.raises(LLMCatalogError, match="Unknown grounding tool"):
        bad.grounding_tools("search")


# --- reaches the gateway at the rewritten URL -------------------------------


async def test_anthropic_reaches_gateway_url(pac: PydanticAICatalog) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/anthropic/light-anthropic").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        agent = Agent(pac.model_for_role("reasoning"))
        with contextlib.suppress(Exception):
            await agent.run("hi")
    assert route.called


async def test_openai_chat_reaches_gateway_url(pac: PydanticAICatalog) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/gpt/oai-light").mock(
            return_value=httpx.Response(200, json=_OPENAI_CHAT_RESP)
        )
        agent = Agent(pac.model_for_role("fast"))
        with contextlib.suppress(Exception):
            await agent.run("hi")
    assert route.called


async def test_gateway_headers_and_query_reach_the_wire(config_dict: dict) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["headers"] = {"Authorization": "Bearer {apiKey}"}
    gw["query"] = {"api-version": "2026-01-01"}
    pac = PydanticAICatalog(Catalog(config_dict))
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/anthropic/light-anthropic").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        agent = Agent(pac.model_for_role("reasoning"))
        with contextlib.suppress(Exception):
            await agent.run("hi")
    assert route.called
    request = route.calls[0].request
    # {apiKey} resolved from EXAMPLEGW_API_KEY, overriding the SDK's own auth
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.url.params.get("api-version") == "2026-01-01"


# --- direct providers hit the vendor endpoint (no rewrite) --------------------


async def test_direct_anthropic_reaches_vendor_url(pac: PydanticAICatalog) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        agent = Agent(pac.model_for_role("chat"))
        with contextlib.suppress(Exception):
            await agent.run("hi")
    assert route.called
    # the vendor SDK's own key env var applies (ANTHROPIC_API_KEY)
    assert route.calls[0].request.headers["x-api-key"] == "anthropic-test-key"


async def test_direct_openai_compatible_url_and_headers(
    pac: PydanticAICatalog,
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{COMPAT_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=_OPENAI_CHAT_RESP)
        )
        agent = Agent(pac.model_for_role("bulk"))
        with contextlib.suppress(Exception):
            await agent.run("hi")
    assert route.called
    request = route.calls[0].request
    assert request.headers["x-tenant"] == "acme"  # declarative vendor header
    assert request.headers["authorization"] == "Bearer fireworks-test-key"


async def test_rewrite_hooks_reach_the_wire(config_dict: dict) -> None:
    # header_rewrite / body_rewrite passed to the catalog end up applied to the
    # outgoing gateway request.
    def add_header(headers: httpx.Headers) -> None:
        headers["x-gateway-extra"] = "on"

    hooked = PydanticAICatalog(
        Catalog(config_dict),
        header_rewrite=add_header,
        body_rewrite=lambda _request: b'{"replaced": true}',
    )
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/anthropic/light-anthropic").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        agent = Agent(hooked.model_for_role("reasoning"))
        with contextlib.suppress(Exception):
            await agent.run("hi")
    assert route.called
    request = route.calls[0].request
    assert request.headers["x-gateway-extra"] == "on"
    assert request.content == b'{"replaced": true}'
