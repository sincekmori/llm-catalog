# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Path-rewriting httpx transport that adapts vendor SDKs to a gateway layout.

Each vendor SDK (anthropic / openai / google-genai) builds its own fixed request
path (``/v1/messages``, ``/chat/completions``, ``/v1beta/models/{m}:{action}``).
A gateway instead expects a ``path_template`` such as ``anthropic/{slug}``.

:class:`GatewayTransport` sits in the vendor SDK's ``httpx`` client and rewrites
each outgoing request's path to the gateway's, leaving everything else (method,
headers, body, query string) intact. It lives in *core* so both adapters reuse
the exact same rewriting.

Slug source
-----------
The ``{slug}`` placeholder is filled from, in order:

* an explicit ``slug`` passed to the transport (what the adapters do — they know
  the resolved model's slug, which may differ from the model id carried in the
  request), or
* the model identifier extracted from the request itself (body ``model`` for
  fixed-path backends, the URL for google) when no explicit slug is given.

Extraction is what keeps the transport generic and is exercised directly by the
tests; the explicit-slug path is what lets a configured ``slug`` differ from the
model id sent upstream in the body.
"""

import json
import re
from collections.abc import Callable

import httpx

__all__ = ["GatewayTransport", "GatewayTransportSync", "HeaderRewrite"]

HeaderRewrite = Callable[[httpx.Headers], None]

# Matches the model id and operation in a google-genai URL path, e.g.
# "/v1beta/models/gemini-2.5-pro:streamGenerateContent".
_GOOGLE_PATH = re.compile(r"/models/(?P<model>[^:/]+):(?P<action>[A-Za-z]+)")

# Backends whose model lives in the JSON request body (not the URL).
_BODY_BACKENDS = frozenset({"anthropic", "openai"})


def _extract_slug_action(
    request: httpx.Request, backend: str
) -> tuple[str | None, str | None]:
    """Pull ``(slug, action)`` out of a request for the given backend.

    Returns ``(None, None)`` when nothing can be extracted (the caller then falls
    back to the configured slug, or leaves the request unrewritten).
    """
    if backend == "google":
        match = _GOOGLE_PATH.search(request.url.path)
        if match is None:
            return None, None
        return match.group("model"), match.group("action")

    if backend in _BODY_BACKENDS:
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

    return None, None


def _build_url(
    request: httpx.Request,
    *,
    base_url: str,
    path_template: str,
    action_map: dict[str, str],
    backend: str,
    slug: str | None,
) -> httpx.URL | None:
    """Compute the rewritten gateway URL, or ``None`` to leave the request as-is."""
    extracted_slug, action = _extract_slug_action(request, backend)
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


class GatewayTransport(httpx.AsyncBaseTransport):
    """Async transport that rewrites vendor paths to the gateway's layout."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        base_url: str,
        path_template: str,
        backend: str,
        action_map: dict[str, str] | None = None,
        slug: str | None = None,
        header_rewrite: HeaderRewrite | None = None,
    ) -> None:
        self.inner = inner
        self.base_url = base_url
        self.path_template = path_template
        self.backend = backend
        self.action_map = action_map or {}
        self.slug = slug
        self.header_rewrite = header_rewrite

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Rewrite the request path to the gateway layout, then delegate."""
        new_url = _build_url(
            request,
            base_url=self.base_url,
            path_template=self.path_template,
            action_map=self.action_map,
            backend=self.backend,
            slug=self.slug,
        )
        if new_url is not None:
            request.url = new_url
        if self.header_rewrite is not None:
            self.header_rewrite(request.headers)
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the wrapped inner transport."""
        await self.inner.aclose()


class GatewayTransportSync(httpx.BaseTransport):
    """Synchronous counterpart of :class:`GatewayTransport`."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        base_url: str,
        path_template: str,
        backend: str,
        action_map: dict[str, str] | None = None,
        slug: str | None = None,
        header_rewrite: HeaderRewrite | None = None,
    ) -> None:
        self.inner = inner
        self.base_url = base_url
        self.path_template = path_template
        self.backend = backend
        self.action_map = action_map or {}
        self.slug = slug
        self.header_rewrite = header_rewrite

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Rewrite the request path to the gateway layout, then delegate."""
        new_url = _build_url(
            request,
            base_url=self.base_url,
            path_template=self.path_template,
            action_map=self.action_map,
            backend=self.backend,
            slug=self.slug,
        )
        if new_url is not None:
            request.url = new_url
        if self.header_rewrite is not None:
            self.header_rewrite(request.headers)
        return self.inner.handle_request(request)

    def close(self) -> None:
        """Close the wrapped inner transport."""
        self.inner.close()
