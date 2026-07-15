# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""GatewayTransport: path rewriting, slug/action extraction, declarative
headers/query injection, and the header/body rewrite hooks.

Tests use ``httpx.MockTransport`` as the inner transport: the GatewayTransport
rewrites ``request.url`` *before* delegating, so the mock handler observes the
final, rewritten request — exactly what the gateway would receive.
"""

import json

import httpx

from llm_catalog.core import GatewayTransport, GatewayTransportSync

BASE = "https://gateway.example.invalid/base"


def _capturer() -> tuple[list[httpx.Request], httpx.MockTransport]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    return seen, httpx.MockTransport(handler)


async def test_anthropic_slug_from_body() -> None:
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="anthropic/{slug}", vendor="anthropic"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1/messages", json={"model": "light-anthropic", "max_tokens": 1}
        )
    assert str(seen[0].url) == f"{BASE}/anthropic/light-anthropic"


async def test_openai_slug_from_body() -> None:
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="gpt/{slug}", vendor="openai"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/chat/completions", json={"model": "light-openai"})
    assert str(seen[0].url) == f"{BASE}/gpt/light-openai"


async def test_any_non_google_vendor_reads_body_model() -> None:
    # Every vendor except google carries the model in the body (matching
    # ai-sdk-catalog's fixed-path handling), so e.g. mistral rewrites too.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="mistral/{slug}", vendor="mistral"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/v1/chat/completions", json={"model": "m-large"})
    assert str(seen[0].url) == f"{BASE}/mistral/m-large"


async def test_explicit_slug_overrides_body() -> None:
    # The configured slug may differ from the model id carried in the body.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="gpt/{slug}",
        vendor="openai",
        slug="oai-light",
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/chat/completions", json={"model": "light-openai"})
    assert str(seen[0].url) == f"{BASE}/gpt/oai-light"
    # body is left untouched (model id stays even though the path uses the slug)
    assert b'"light-openai"' in seen[0].content


async def test_google_slug_and_action_from_url() -> None:
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="gemini/{slug}:{action}",
        vendor="google",
        action_map={"streamGenerateContent": "customStreamGenerateContent"},
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1beta/models/search-google:streamGenerateContent?alt=sse",
            json={"contents": []},
        )
    url = seen[0].url
    assert url.path == "/base/gemini/search-google:customStreamGenerateContent"
    # query string preserved
    assert url.params.get("alt") == "sse"


async def test_google_action_passthrough_when_not_mapped() -> None:
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="gemini/{slug}:{action}", vendor="google"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1beta/models/search-google:generateContent", json={}
        )
    assert seen[0].url.path == "/base/gemini/search-google:generateContent"


async def test_declarative_headers_win_over_request_headers() -> None:
    # Config headers are merged over the vendor SDK's own (same-name wins).
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="anthropic/{slug}",
        vendor="anthropic",
        headers={"x-api-key": "gateway-key", "x-tenant": "acme"},
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1/messages",
            json={"model": "m"},
            headers={"x-api-key": "sdk-key"},
        )
    assert seen[0].headers["x-api-key"] == "gateway-key"
    assert seen[0].headers["x-tenant"] == "acme"


async def test_declarative_query_lands_on_final_url() -> None:
    # Query params are appended after the path rewrite; an existing parameter
    # is overridden so the config value wins.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="gemini/{slug}:{action}",
        vendor="google",
        query={"api-version": "2026-01-01"},
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1beta/models/g:generateContent?alt=sse&api-version=old",
            json={},
        )
    url = seen[0].url
    assert url.path == "/base/gemini/g:generateContent"
    assert url.params.get("alt") == "sse"
    assert url.params.get("api-version") == "2026-01-01"


async def test_no_path_rewrite_for_direct_use() -> None:
    # With no path_template the transport only injects headers/query — what the
    # adapters use for direct providers.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        headers={"x-tenant": "acme"},
        query={"api-version": "2026-01-01"},
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            "https://api.vendor.example.invalid/v1/messages", json={"model": "m"}
        )
    url = seen[0].url
    assert url.path == "/v1/messages"
    assert url.params.get("api-version") == "2026-01-01"
    assert seen[0].headers["x-tenant"] == "acme"


async def test_header_rewrite_invoked() -> None:
    seen, inner = _capturer()

    def rewrite(headers: httpx.Headers) -> None:
        headers["x-gateway-auth"] = "rewritten"

    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="anthropic/{slug}",
        vendor="anthropic",
        header_rewrite=rewrite,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/v1/messages", json={"model": "m"})
    assert seen[0].headers["x-gateway-auth"] == "rewritten"


async def test_body_rewrite_replaces_body_and_content_length() -> None:
    seen, inner = _capturer()

    def rewrite(request: httpx.Request) -> bytes | None:
        body = json.loads(request.content)
        body.pop("unwanted", None)
        return json.dumps(body).encode()

    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="gemini/{slug}:{action}",
        vendor="google",
        body_rewrite=rewrite,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1beta/models/search-google:generateContent",
            json={"contents": [], "unwanted": "x"},
        )
    assert json.loads(seen[0].content) == {"contents": []}
    # Content-Length matches the rewritten body, not the original one.
    assert seen[0].headers["content-length"] == str(len(seen[0].content))


async def test_body_rewrite_sees_the_rewritten_url() -> None:
    # The hook runs after the URL rewrite so it can target gateway paths.
    _, inner = _capturer()
    urls: list[str] = []

    def rewrite(request: httpx.Request) -> bytes | None:
        urls.append(str(request.url))
        return None

    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="gemini/{slug}:{action}",
        vendor="google",
        body_rewrite=rewrite,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1beta/models/search-google:generateContent", json={}
        )
    assert urls == [f"{BASE}/gemini/search-google:generateContent"]


async def test_body_rewrite_none_leaves_request_untouched() -> None:
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="anthropic/{slug}",
        vendor="anthropic",
        body_rewrite=lambda _request: None,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/v1/messages", json={"model": "m", "keep": True})
    assert json.loads(seen[0].content) == {"model": "m", "keep": True}


def test_sync_body_rewrite() -> None:
    seen, inner = _capturer()
    transport = GatewayTransportSync(
        inner,
        base_url=BASE,
        path_template="anthropic/{slug}",
        vendor="anthropic",
        body_rewrite=lambda _request: b'{"replaced": true}',
    )
    with httpx.Client(transport=transport) as client:
        client.post(f"{BASE}/v1/messages", json={"model": "m"})
    assert json.loads(seen[0].content) == {"replaced": True}
    assert seen[0].headers["content-length"] == str(len(seen[0].content))


async def test_no_rewrite_when_model_absent() -> None:
    # No body model and no explicit slug -> request passes through unchanged.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="anthropic/{slug}", vendor="anthropic"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/v1/messages", json={"no_model": True})
    assert seen[0].url.path == "/base/v1/messages"


def test_sync_transport() -> None:
    seen, _ = _capturer()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    inner = httpx.MockTransport(handler)
    transport = GatewayTransportSync(
        inner,
        base_url=BASE,
        path_template="anthropic/{slug}",
        vendor="anthropic",
        headers={"x-tenant": "acme"},
        query={"api-version": "2026-01-01"},
    )
    with httpx.Client(transport=transport) as client:
        client.post(f"{BASE}/v1/messages", json={"model": "light-anthropic"})
    assert seen[0].url.path == "/base/anthropic/light-anthropic"
    assert seen[0].url.params.get("api-version") == "2026-01-01"
    assert seen[0].headers["x-tenant"] == "acme"
