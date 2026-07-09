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
    from llm_catalog.litellm import ChatCatalogLLM

    h = ChatCatalogLLM(catalog=Catalog(config_dict))
    rm = h._resolve("fast", {"custom_llm_provider": "examplegw"})
    assert rm.backend == "openai"
    assert rm.base_url == BASE
    assert rm.api_key_env == "EXAMPLEGW_API_KEY"


def test_get_catalog_reads_config_json(tmp_path, config_dict) -> None:
    import json

    from llm_catalog.litellm import ChatCatalogLLM

    path = tmp_path / "llm-catalog.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    h = ChatCatalogLLM(config_path=path)
    assert set(h.get_catalog().roles) == {"fast", "reasoning"}


def test_get_catalog_rejects_non_json(tmp_path) -> None:
    from llm_catalog.core import ConfigError
    from llm_catalog.litellm import ChatCatalogLLM

    path = tmp_path / "llm-catalog.json"
    path.write_text("providers: []", encoding="utf-8")  # YAML, not JSON
    h = ChatCatalogLLM(config_path=path)
    with pytest.raises(ConfigError, match="not valid JSON"):
        h.get_catalog()
