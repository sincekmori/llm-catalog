# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""GatewayTransport path rewriting, slug/action extraction, and header rewrite.

Tests use ``httpx.MockTransport`` as the inner transport: the GatewayTransport
rewrites ``request.url`` *before* delegating, so the mock handler observes the
final, rewritten request — exactly what the gateway would receive.
"""

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
        inner, base_url=BASE, path_template="anthropic/{slug}", backend="anthropic"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1/messages", json={"model": "light-anthropic", "max_tokens": 1}
        )
    assert str(seen[0].url) == f"{BASE}/anthropic/light-anthropic"


async def test_openai_slug_from_body() -> None:
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="gpt/{slug}", backend="openai"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/chat/completions", json={"model": "light-openai"})
    assert str(seen[0].url) == f"{BASE}/gpt/light-openai"


async def test_explicit_slug_overrides_body() -> None:
    # The configured slug may differ from the model id carried in the body.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="gpt/{slug}",
        backend="openai",
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
        backend="google",
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
        inner, base_url=BASE, path_template="gemini/{slug}:{action}", backend="google"
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(
            f"{BASE}/v1beta/models/search-google:generateContent", json={}
        )
    assert seen[0].url.path == "/base/gemini/search-google:generateContent"


async def test_header_rewrite_invoked() -> None:
    seen, inner = _capturer()

    def rewrite(headers: httpx.Headers) -> None:
        headers["x-gateway-auth"] = "rewritten"

    transport = GatewayTransport(
        inner,
        base_url=BASE,
        path_template="anthropic/{slug}",
        backend="anthropic",
        header_rewrite=rewrite,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(f"{BASE}/v1/messages", json={"model": "m"})
    assert seen[0].headers["x-gateway-auth"] == "rewritten"


async def test_no_rewrite_when_model_absent() -> None:
    # No body model and no explicit slug -> request passes through unchanged.
    seen, inner = _capturer()
    transport = GatewayTransport(
        inner, base_url=BASE, path_template="anthropic/{slug}", backend="anthropic"
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
        inner, base_url=BASE, path_template="anthropic/{slug}", backend="anthropic"
    )
    with httpx.Client(transport=transport) as client:
        client.post(f"{BASE}/v1/messages", json={"model": "light-anthropic"})
    assert str(seen[0].url) == f"{BASE}/anthropic/light-anthropic"
