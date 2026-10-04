# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""AISDKCatalog: provider/protocol selection (gateway and direct), the
vendor/api combinations that must raise, request params, client lifecycle, and
that requests reach the right URL (a recording mock transport in place of the
network; see the ``wire`` fixture).

These double as contract tests against the beta ``ai`` package: every one that
goes through ``ai.stream`` fails if ``ai`` stops honouring the injected client,
base URL, key or protocol.
"""

import contextlib
import json

import ai
import httpx2
import pytest
from ai.providers.anthropic import AnthropicMessagesProtocol
from ai.providers.openai import OpenAIChatCompletionsProtocol, OpenAIResponsesProtocol

from llm_catalog.ai_sdk import AISDKCatalog
from llm_catalog.core import Catalog
from llm_catalog.core.errors import LLMCatalogError

BASE = "https://gateway.example.invalid/base"
COMPAT_BASE = "https://compat.example.invalid/v1"


async def _send(model: ai.Model, params: ai.InferenceRequestParams | None = None):
    # The mock answers 418; only the outgoing request matters.
    with contextlib.suppress(ai.AIError):
        async with ai.stream(model, [ai.user_message("hi")], params=params) as stream:
            async for _ in stream:
                pass


# --- provider / protocol selection ------------------------------------------


def test_gateway_protocols(config_dict: dict) -> None:
    cat = AISDKCatalog(config_dict)
    assert isinstance(
        cat.model_for_role("reasoning").protocol, AnthropicMessagesProtocol
    )
    # an explicit chat surface
    assert isinstance(
        cat.model_for_role("fast").protocol, OpenAIChatCompletionsProtocol
    )
    # api omitted -> the OpenAI vendor default is the Responses API (TS parity)
    assert isinstance(cat.model_for_role("respond").protocol, OpenAIResponsesProtocol)


def test_direct_protocols(config_dict: dict) -> None:
    cat = AISDKCatalog(config_dict)
    assert isinstance(cat.model_for_role("chat").protocol, AnthropicMessagesProtocol)
    assert isinstance(
        cat.model_for_role("direct-responses").protocol, OpenAIResponsesProtocol
    )
    assert isinstance(
        cat.model_for_role("direct-chat").protocol, OpenAIChatCompletionsProtocol
    )
    # openai-compatible speaks Chat Completions, never ai's "openai" default
    assert isinstance(
        cat.model_for_role("bulk").protocol, OpenAIChatCompletionsProtocol
    )


def test_model_id_and_key_lookup(config_dict: dict) -> None:
    cat = AISDKCatalog(Catalog(config_dict))
    model = cat.model("examplegw:light-openai")
    assert isinstance(model, ai.Model)
    # the body carries the model id; the slug only appears in the gateway path
    assert model.id == "light-openai"


def test_anthropic_accepts_chat_as_alias(config_dict: dict) -> None:
    # ai-sdk-catalog: a single-surface vendor exposes "chat" as that surface.
    config_dict["providers"][1]["models"][0]["api"] = "chat"
    cat = AISDKCatalog(config_dict)
    assert isinstance(cat.model_for_role("chat").protocol, AnthropicMessagesProtocol)


# --- combinations that must raise instead of being rerouted -------------------


@pytest.mark.parametrize("vendor", ["google", "mistral", "groq", "xai", "deepseek"])
def test_unsupported_vendor_raises_at_use(config_dict: dict, vendor: str) -> None:
    # A shared config may route some vendors only from the TypeScript side;
    # they validate fine and fail here only when actually used.
    gw = config_dict["providers"][0]["gateway"]
    template = "x/{slug}:{action}" if vendor == "google" else "x/{slug}"
    gw["backends"]["other"] = {"vendor": vendor, "pathTemplate": template}
    config_dict["providers"][0]["models"].append({"id": "m", "backend": "other"})
    cat = AISDKCatalog(config_dict)
    with pytest.raises(LLMCatalogError, match="not supported by the AI SDK"):
        cat.model("examplegw:m")


@pytest.mark.parametrize(
    ("provider", "model", "api"),
    [
        (0, 0, "responses"),  # anthropic backend
        (0, 0, "completion"),
        (0, 1, "completion"),  # openai backend
        (3, 0, "responses"),  # openai-compatible vendor
        (3, 0, "completion"),
    ],
)
def test_unsupported_api_raises(
    config_dict: dict, provider: int, model: int, api: str
) -> None:
    entry = config_dict["providers"][provider]["models"][model]
    entry["api"] = api
    cat = AISDKCatalog(config_dict)
    key = f"{config_dict['providers'][provider]['id']}:{entry['id']}"
    with pytest.raises(LLMCatalogError, match=f'api="{api}"'):
        cat.model(key)
    # nothing was opened for a model that could not be built
    assert cat._clients == []


# --- request params -----------------------------------------------------------


def test_params_translate_merged_settings(config_dict: dict) -> None:
    cat = AISDKCatalog(config_dict)
    params = cat.params_for_role("fast")
    # model settings win over the provider's (temperature 0 -> 0.5)
    assert params == ai.InferenceRequestParams(
        sampling={
            ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(temperature=0.5),
            ai.TopPSamplerParams: ai.TopPSamplerParams(top_p=0.9),
            ai.SeedSamplerParams: ai.SeedSamplerParams(seed=7),
        },
        output=ai.OutputParams(max_tokens=256),
    )
    # the provider default flows through when the model does not override it
    assert cat.params("examplegw:light-anthropic") == ai.InferenceRequestParams(
        sampling={
            ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(temperature=0)
        }
    )


def test_params_default_when_no_settings(config_dict: dict) -> None:
    cat = AISDKCatalog(config_dict)
    assert cat.params_for_role("chat") == ai.InferenceRequestParams()


def test_penalties_and_top_k(config_dict: dict) -> None:
    config_dict["providers"][1]["models"][0]["settings"] = {
        "topK": 40,
        "presencePenalty": 0.1,
        "frequencyPenalty": 0.2,
        "providerOptions": {"anthropic": {"speed": "fast"}},  # skipped
    }
    assert AISDKCatalog(config_dict).params_for_role(
        "chat"
    ) == ai.InferenceRequestParams(
        sampling={
            ai.TopKSamplerParams: ai.TopKSamplerParams(top_k=40),
            ai.RepetitionPenaltyParams: ai.RepetitionPenaltyParams(
                presence_penalty=0.1, frequency_penalty=0.2
            ),
        }
    )


def test_stop_sequences_use_each_protocols_own_field(config_dict: dict) -> None:
    stop = {"stopSequences": ["END"]}
    config_dict["providers"][0]["models"][0]["settings"] = stop  # anthropic
    config_dict["providers"][0]["models"][1]["settings"] = stop  # openai chat
    config_dict["providers"][0]["models"][2]["settings"] = stop  # openai responses
    cat = AISDKCatalog(config_dict)
    assert cat.params_for_role("reasoning").extra_body == {"stop_sequences": ["END"]}
    assert cat.params_for_role("fast").extra_body == {"stop": ["END"]}
    with pytest.raises(LLMCatalogError, match="no stop-sequence parameter"):
        cat.params_for_role("respond")


async def test_params_reach_the_request_body(config_dict: dict, wire) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("fast"), cat.params_for_role("fast"))
    body = json.loads(wire.requests[0].content)
    assert body["model"] == "light-openai"
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    assert body["seed"] == 7
    assert body["max_completion_tokens"] == 256


# --- client lifecycle -----------------------------------------------------------


async def test_models_are_reused_and_clients_closed(config_dict: dict, wire) -> None:
    cat = AISDKCatalog(config_dict, transport_factory=wire.transport)
    first = cat.model_for_role("fast")
    assert cat.model("examplegw:light-openai") is first  # one client per model
    cat.model_for_role("reasoning")
    clients = list(cat._clients)
    assert len(clients) == 2
    await cat.aclose()
    assert all(client.is_closed for client in clients)
    assert cat._clients == []
    # usable again after closing: models are rebuilt on demand
    assert cat.model_for_role("fast") is not first
    await cat.aclose()


# --- reaches the gateway at the rewritten URL ---------------------------------


async def test_anthropic_reaches_gateway_url(config_dict: dict, wire) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("reasoning"))
    request = wire.requests[0]
    assert str(request.url.copy_with(query=None)) == f"{BASE}/anthropic/light-anthropic"
    assert request.headers["x-api-key"] == "test-key"  # the gateway key


async def test_openai_chat_reaches_gateway_url(config_dict: dict, wire) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("fast"))
    request = wire.requests[0]
    # the configured slug fills the path; the body keeps the model id
    assert str(request.url) == f"{BASE}/gpt/oai-light"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content)["model"] == "light-openai"


async def test_openai_responses_reaches_gateway_url(config_dict: dict, wire) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("respond"))
    request = wire.requests[0]
    assert str(request.url) == f"{BASE}/gpt/resp-openai"
    # a Responses API payload ("input"), not Chat Completions ("messages")
    body = json.loads(request.content)
    assert "input" in body
    assert "messages" not in body


async def test_gateway_headers_and_query_reach_the_wire(
    config_dict: dict, wire
) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["headers"] = {"Authorization": "Bearer {apiKey}"}
    gw["query"] = {"api-version": "2026-01-01"}
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("reasoning"))
    request = wire.requests[0]
    assert request.url.path == "/base/anthropic/light-anthropic"
    # {apiKey} resolved from EXAMPLEGW_API_KEY, on top of the SDK's own auth
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.url.params.get("api-version") == "2026-01-01"


# --- direct providers hit the vendor endpoint (no rewrite) --------------------


async def test_direct_anthropic_reaches_vendor_url(config_dict: dict, wire) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("chat"))
    request = wire.requests[0]
    assert request.url.host == "api.anthropic.com"
    assert request.url.path == "/v1/messages"
    # the vendor's own key env var applies (ANTHROPIC_API_KEY)
    assert request.headers["x-api-key"] == "anthropic-test-key"


async def test_direct_openai_surfaces(config_dict: dict, wire) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("direct-responses"))
        await _send(cat.model_for_role("direct-chat"))
    responses, chat = wire.requests
    assert str(responses.url) == "https://api.openai.com/v1/responses"
    assert str(chat.url) == "https://api.openai.com/v1/chat/completions"
    assert responses.headers["authorization"] == "Bearer openai-test-key"


async def test_direct_openai_compatible_url_and_headers(
    config_dict: dict, wire
) -> None:
    async with AISDKCatalog(config_dict, transport_factory=wire.transport) as cat:
        await _send(cat.model_for_role("bulk"))
    request = wire.requests[0]
    assert str(request.url) == f"{COMPAT_BASE}/chat/completions"
    assert request.headers["x-tenant"] == "acme"  # declarative vendor header
    assert request.headers["authorization"] == "Bearer fireworks-test-key"


async def test_rewrite_hooks_reach_the_wire(config_dict: dict, wire) -> None:
    # header_rewrite / body_rewrite passed to the catalog end up applied to the
    # outgoing gateway request.
    def add_header(headers: httpx2.Headers) -> None:
        headers["x-gateway-extra"] = "on"

    hooked = AISDKCatalog(
        config_dict,
        header_rewrite=add_header,
        body_rewrite=lambda _request: b'{"replaced": true}',
        transport_factory=wire.transport,
    )
    async with hooked:
        await _send(hooked.model_for_role("reasoning"))
    request = wire.requests[0]
    assert request.url.path == "/base/anthropic/light-anthropic"
    assert request.headers["x-gateway-extra"] == "on"
    assert request.content == b'{"replaced": true}'
