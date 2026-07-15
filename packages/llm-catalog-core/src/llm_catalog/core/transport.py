# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""httpx transport that adapts vendor SDKs to a gateway layout.

Each vendor SDK (anthropic / openai / google-genai / ...) builds its own fixed
request path (``/v1/messages``, ``/chat/completions``,
``/v1beta/models/{m}:{action}``). A gateway instead expects a ``path_template``
such as ``anthropic/{slug}``.

:class:`GatewayTransport` sits in the vendor SDK's ``httpx`` client and rewrites
each outgoing request's path to the gateway's, leaving everything else (method,
body, query string) intact. It also injects the config-declared transport
extras — extra request ``headers`` (same-name wins over the SDK's own) and
``query`` parameters appended to the final URL — mirroring ai-sdk-catalog's
``headers``/``query``. It lives in *core* so both adapters reuse the exact same
behaviour.

Path rewriting is optional: with no ``path_template`` the transport only
applies headers/query and the rewrite hooks — that is what the adapters use for
**direct** providers, whose vendor SDK already builds the right URL.

Slug source
-----------
The ``{slug}`` placeholder is filled from, in order:

* an explicit ``slug`` passed to the transport (what the adapters do — they know
  the resolved model's slug, which may differ from the model id carried in the
  request), or
* the model identifier extracted from the request itself (the URL for
  ``google``, whose model travels in the path; the body ``model`` field for
  every other vendor) when no explicit slug is given.

Extraction is what keeps the transport generic and is exercised directly by the
tests; the explicit-slug path is what lets a configured ``slug`` differ from the
model id sent upstream in the body.
"""

import json
import re
from collections.abc import Callable

import httpx

__all__ = ["BodyRewrite", "GatewayTransport", "GatewayTransportSync", "HeaderRewrite"]

HeaderRewrite = Callable[[httpx.Headers], None]

# Receives the request *after* the URL/header rewrites (so the gateway URL is
# inspectable) and returns a replacement body, or ``None`` to leave it as-is.
# The escape hatch for gateways that reject part of a vendor SDK's payload —
# e.g. an extra field in replayed tool-call history that a strict endpoint
# refuses. The transport takes care of rebuilding the request (Content-Length
# included); the hook only transforms bytes.
BodyRewrite = Callable[[httpx.Request], "bytes | None"]

# Matches the model id and operation in a google-genai URL path, e.g.
# "/v1beta/models/gemini-2.5-pro:streamGenerateContent".
_GOOGLE_PATH = re.compile(r"/models/(?P<model>[^:/]+):(?P<action>[A-Za-z]+)")


def _extract_slug_action(
    request: httpx.Request, vendor: str
) -> tuple[str | None, str | None]:
    """Pull ``(slug, action)`` out of a request for the given vendor.

    For ``google`` the model and operation live in the URL; for every other
    vendor the model travels in the JSON request body. Returns ``(None, None)``
    when nothing can be extracted (the caller then falls back to the configured
    slug, or leaves the request unrewritten).
    """
    if vendor == "google":
        match = _GOOGLE_PATH.search(request.url.path)
        if match is None:
            return None, None
        return match.group("model"), match.group("action")

    try:
        raw = request.content
    except httpx.RequestNotRead:  # streaming body not materialised — uncommon here
        return None, None
    if not raw:
        return None, None
    try:
        body = json.loads(raw)
    except ValueError:  # JSONDecodeError and UnicodeDecodeError both subclass it
        return None, None
    model = body.get("model") if isinstance(body, dict) else None
    return (model if isinstance(model, str) and model else None), None


def _build_url(
    request: httpx.Request,
    *,
    base_url: str,
    path_template: str,
    action_map: dict[str, str],
    vendor: str,
    slug: str | None,
) -> httpx.URL | None:
    """Compute the rewritten gateway URL, or ``None`` to leave the request as-is."""
    extracted_slug, action = _extract_slug_action(request, vendor)
    final_slug = slug if slug is not None else extracted_slug
    if final_slug is None:
        # Nothing to substitute — don't touch the request.
        return None

    mapped_action = action_map.get(action, action) if action is not None else ""
    new_path = path_template.format(slug=final_slug, action=mapped_action)

    base = base_url.rstrip("/")
    path = new_path.lstrip("/")
    # Plain concatenation (not httpx.URL.join) so a gateway base path like
    # ".../base" is preserved rather than treated as a sibling to replace.
    url = httpx.URL(f"{base}/{path}")
    # Re-attach the original query string (e.g. google's ?alt=sse).
    return url.copy_merge_params(request.url.params)


def _apply_body_rewrite(request: httpx.Request, rewrite: BodyRewrite) -> httpx.Request:
    """Run the body hook and rebuild the request when it returns a new body.

    Rebuilding (rather than mutating) keeps ``Content-Length`` correct: the
    stale header is dropped and httpx recomputes it for the new content.
    Streaming uploads (body not materialised) are left untouched.
    """
    try:
        _ = request.content
    except httpx.RequestNotRead:  # pragma: no cover - unusual for JSON APIs
        return request
    new_body = rewrite(request)
    if new_body is None:
        return request
    headers = httpx.Headers(
        [(k, v) for k, v in request.headers.raw if k.lower() != b"content-length"]
    )
    return httpx.Request(request.method, request.url, headers=headers, content=new_body)


class _RewriteMixin:
    """The shared request rewrite, identical for the async and sync transports."""

    base_url: "str | None"
    path_template: "str | None"
    vendor: "str | None"
    action_map: dict[str, str]
    slug: "str | None"
    headers: dict[str, str]
    query: dict[str, str]
    header_rewrite: "HeaderRewrite | None"
    body_rewrite: "BodyRewrite | None"

    def _init(
        self,
        base_url: "str | None",
        path_template: "str | None",
        vendor: "str | None",
        action_map: "dict[str, str] | None",
        slug: "str | None",
        headers: "dict[str, str] | None",
        query: "dict[str, str] | None",
        header_rewrite: "HeaderRewrite | None",
        body_rewrite: "BodyRewrite | None",
    ) -> None:
        self.base_url = base_url
        self.path_template = path_template
        self.vendor = vendor
        self.action_map = action_map or {}
        self.slug = slug
        self.headers = headers or {}
        self.query = query or {}
        self.header_rewrite = header_rewrite
        self.body_rewrite = body_rewrite

    def _rewrite(self, request: httpx.Request) -> httpx.Request:
        # 1. Path rewrite to the gateway layout (gateway providers only).
        if (
            self.base_url is not None
            and self.path_template is not None
            and self.vendor is not None
        ):
            new_url = _build_url(
                request,
                base_url=self.base_url,
                path_template=self.path_template,
                action_map=self.action_map,
                vendor=self.vendor,
                slug=self.slug,
            )
            if new_url is not None:
                request.url = new_url
        # 2. Config-declared query params land on the final URL (a parameter
        #    already present is overridden, so the config value wins).
        if self.query:
            request.url = request.url.copy_merge_params(self.query)
        # 3. Config-declared headers win over the vendor SDK's own.
        for name, value in self.headers.items():
            request.headers[name] = value
        # 4. Code-level escape hatches run last, seeing the final request.
        if self.header_rewrite is not None:
            self.header_rewrite(request.headers)
        if self.body_rewrite is not None:
            request = _apply_body_rewrite(request, self.body_rewrite)
        return request


class GatewayTransport(_RewriteMixin, httpx.AsyncBaseTransport):
    """Async transport that adapts vendor requests to the configured layout."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        base_url: str | None = None,
        path_template: str | None = None,
        vendor: str | None = None,
        action_map: dict[str, str] | None = None,
        slug: str | None = None,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
        header_rewrite: HeaderRewrite | None = None,
        body_rewrite: BodyRewrite | None = None,
    ) -> None:
        self.inner = inner
        self._init(
            base_url,
            path_template,
            vendor,
            action_map,
            slug,
            headers,
            query,
            header_rewrite,
            body_rewrite,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Apply the rewrites, then delegate to the inner transport."""
        return await self.inner.handle_async_request(self._rewrite(request))

    async def aclose(self) -> None:
        """Close the wrapped inner transport."""
        await self.inner.aclose()


class GatewayTransportSync(_RewriteMixin, httpx.BaseTransport):
    """Synchronous counterpart of :class:`GatewayTransport`."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        base_url: str | None = None,
        path_template: str | None = None,
        vendor: str | None = None,
        action_map: dict[str, str] | None = None,
        slug: str | None = None,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
        header_rewrite: HeaderRewrite | None = None,
        body_rewrite: BodyRewrite | None = None,
    ) -> None:
        self.inner = inner
        self._init(
            base_url,
            path_template,
            vendor,
            action_map,
            slug,
            headers,
            query,
            header_rewrite,
            body_rewrite,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Apply the rewrites, then delegate to the inner transport."""
        return self.inner.handle_request(self._rewrite(request))

    def close(self) -> None:
        """Close the wrapped inner transport."""
        self.inner.close()
