# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM adapter: in-process register(), OpenAI-form responses, model
resolution, and self-resolution (no gateway details in litellm_params)."""

import httpx
import litellm
import pytest
import respx

from llm_catalog.core import Catalog

BASE = "https://gateway.example.invalid/base"

_ANTHROPIC_RESP = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "light-anthropic",
    "content": [{"type": "text", "text": "ok-anthropic"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_OPENAI_RESP = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "oai-light",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok-openai"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_OPENAI_STREAM = (
    'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"he"},"finish_reason":null}]}\n\n'
    'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}\n\n'
    'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_in_process_completion_openai_form(registered) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/anthropic/light-anthropic").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        resp = litellm.completion(
            model="examplegw/reasoning",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert route.called
    # OpenAI-form response
    assert resp.choices[0].message.content == "ok-anthropic"
    assert resp.object == "chat.completion"


async def test_in_process_acompletion(registered) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/gpt/oai-light").mock(
            return_value=httpx.Response(200, json=_OPENAI_RESP)
        )
        resp = await litellm.acompletion(
            model="examplegw/fast",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert route.called
    assert resp.choices[0].message.content == "ok-openai"


async def test_streaming_yields_openai_form(registered) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.post(f"{BASE}/gpt/oai-light").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_OPENAI_STREAM.encode(),
            )
        )
        collected = []
        resp = await litellm.acompletion(
            model="examplegw/fast",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for chunk in resp:
            piece = chunk.choices[0].delta.content
            if piece:
                collected.append(piece)
    assert "".join(collected) == "hello"


def test_resolves_role_and_provider_model_key(registered) -> None:
    # A non-role model id under the provider must also resolve.
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/anthropic/light-anthropic").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        litellm.completion(
            model="examplegw/light-anthropic",  # model id, not a role
            messages=[{"role": "user", "content": "hi"}],
        )
    assert route.called


def test_register_is_idempotent(registered) -> None:
    before = [
        e for e in litellm.custom_provider_map if e.get("provider") == "examplegw"
    ]
    registered.register()
    registered.register()
    after = [e for e in litellm.custom_provider_map if e.get("provider") == "examplegw"]
    assert len(before) == 1
    assert len(after) == 1


def test_self_resolves_without_litellm_params(config_dict) -> None:
    # The handler must resolve purely from the catalog (issue #18216 mitigation):
    # given only the model name + provider, it produces a full ResolvedModel.
    from llm_catalog.core import EnvVarRef
    from llm_catalog.litellm import ChatCatalogLLM

    h = ChatCatalogLLM(catalog=Catalog(config_dict))
    rm = h._resolve("fast", {"custom_llm_provider": "examplegw"})
    assert rm.kind == "gateway"
    assert rm.backend == "openai"
    assert rm.vendor == "openai"
    assert rm.base_url == BASE
    assert rm.api_key_config == EnvVarRef(envVarName="EXAMPLEGW_API_KEY")


def test_direct_provider_reaches_vendor_url(registered) -> None:
    # A direct provider skips the path rewrite and calls the vendor endpoint,
    # with the key coming from the vendor's own env var (ANTHROPIC_API_KEY).
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        resp = litellm.completion(
            model="claude-direct/chat",  # role defined on the direct provider
            messages=[{"role": "user", "content": "hi"}],
        )
    assert route.called
    assert resp.choices[0].message.content == "ok-anthropic"


def test_register_warns_on_builtin_provider_id_collision(config_dict) -> None:
    import llm_catalog.litellm as mod
    from llm_catalog.litellm import ChatCatalogLLM, ProviderIdCollisionWarning

    config_dict["providers"][1]["id"] = "anthropic"  # collides with a built-in
    config_dict["roles"]["chat"] = "anthropic:claude-sonnet-5"
    original_handler = mod.handler
    mod.handler = ChatCatalogLLM(catalog=Catalog(config_dict))
    try:
        with pytest.warns(ProviderIdCollisionWarning, match="#23352"):
            mod.register()
    finally:
        # Undo the global registration so other tests see a clean LiteLLM state.
        litellm.custom_provider_map = [
            e for e in litellm.custom_provider_map if e.get("provider") != "anthropic"
        ]
        mod._registered.discard("anthropic")
        mod.handler = original_handler


def test_rewrite_hooks_reach_the_wire(config_dict) -> None:
    # header_rewrite / body_rewrite on a handler end up applied to the outgoing
    # gateway request (the module-level `handler` stays hook-free).
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    from llm_catalog.litellm import ChatCatalogLLM

    def add_header(headers: httpx.Headers) -> None:
        headers["x-gateway-extra"] = "on"

    h = ChatCatalogLLM(
        catalog=Catalog(config_dict),
        header_rewrite=add_header,
        body_rewrite=lambda _request: b'{"replaced": true}',
    )
    rm = h._resolve("reasoning", {"custom_llm_provider": "examplegw"})
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE}/anthropic/light-anthropic").mock(
            return_value=httpx.Response(200, json=_ANTHROPIC_RESP)
        )
        http = h._sync_client(rm)
        assert isinstance(http, HTTPHandler)  # anthropic backend -> HTTPHandler
        http.client.post(f"{BASE}/v1/messages", json={"model": "light-anthropic"})
    assert route.called
    request = route.calls[0].request
    assert request.headers["x-gateway-extra"] == "on"
    assert request.content == b'{"replaced": true}'


def test_get_catalog_reads_config_json(tmp_path, config_dict) -> None:
    import json

    from llm_catalog.litellm import ChatCatalogLLM

    path = tmp_path / "llm-catalog.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    h = ChatCatalogLLM(config_path=path)
    assert set(h.get_catalog().roles) == {"fast", "reasoning", "chat"}


def test_get_catalog_rejects_non_json(tmp_path) -> None:
    from llm_catalog.core import ConfigError
    from llm_catalog.litellm import ChatCatalogLLM

    path = tmp_path / "llm-catalog.json"
    path.write_text("providers: []", encoding="utf-8")  # YAML, not JSON
    h = ChatCatalogLLM(config_path=path)
    with pytest.raises(ConfigError, match="not valid JSON"):
        h.get_catalog()
