# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The ``httpx`` flavour of the gateway transport.

For runtimes still built on ``httpx`` (e.g. LiteLLM). SDKs that moved to
``httpx2`` (``openai>=3``, ``anthropic>=1``) are typed for ``httpx2`` clients,
and some reject ``httpx`` ones outright — use the identical classes in
:mod:`llm_catalog.core.transport2` for those.

What the transport rewrites, and how the ``{slug}`` placeholder is filled, is
described once in :mod:`llm_catalog.core._rewrite`, which both flavours share.
"""

from collections.abc import Callable

import httpx

from ._rewrite import RewriteMixin

__all__ = ["BodyRewrite", "GatewayTransport", "GatewayTransportSync", "HeaderRewrite"]

HeaderRewrite = Callable[[httpx.Headers], None]

# Receives the request *after* the URL/header rewrites (so the gateway URL is
# inspectable) and returns a replacement body, or ``None`` to leave it as-is.
# The escape hatch for gateways that reject part of a vendor SDK's payload —
# e.g. an extra field in replayed tool-call history that a strict endpoint
# refuses. The transport takes care of rebuilding the request (Content-Length
# included); the hook only transforms bytes.
BodyRewrite = Callable[[httpx.Request], "bytes | None"]


class GatewayTransport(
    RewriteMixin[httpx.AsyncBaseTransport, httpx.Headers, httpx.Request],
    httpx.AsyncBaseTransport,
):
    """Async transport that adapts vendor requests to the configured layout."""

    _http = httpx

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Apply the rewrites, then delegate to the inner transport."""
        return await self.inner.handle_async_request(self._rewrite(request))

    async def aclose(self) -> None:
        """Close the wrapped inner transport."""
        await self.inner.aclose()


class GatewayTransportSync(
    RewriteMixin[httpx.BaseTransport, httpx.Headers, httpx.Request],
    httpx.BaseTransport,
):
    """Synchronous counterpart of :class:`GatewayTransport`."""

    _http = httpx

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Apply the rewrites, then delegate to the inner transport."""
        return self.inner.handle_request(self._rewrite(request))

    def close(self) -> None:
        """Close the wrapped inner transport."""
        self.inner.close()
