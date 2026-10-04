# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The ``httpx2`` flavour of the gateway transport.

For vendor SDKs built on ``httpx2`` (``openai>=3``, ``anthropic>=1``, Pydantic
AI, the AI SDK for Python). It needs the optional dependency: install
``llm-catalog-core[httpx2]``. The classes mirror
:mod:`llm_catalog.core.transport` (the ``httpx`` flavour) one-to-one.

What the transport rewrites, and how the ``{slug}`` placeholder is filled, is
described once in :mod:`llm_catalog.core._rewrite`, which both flavours share.
"""

from collections.abc import Callable

try:
    import httpx2
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the extra
    raise ModuleNotFoundError(
        "llm_catalog.core.transport2 requires httpx2; install it with "
        '`pip install "llm-catalog-core[httpx2]"`.'
    ) from exc

from ._rewrite import RewriteMixin

__all__ = ["BodyRewrite", "GatewayTransport", "GatewayTransportSync", "HeaderRewrite"]

HeaderRewrite = Callable[[httpx2.Headers], None]

# Receives the request *after* the URL/header rewrites (so the gateway URL is
# inspectable) and returns a replacement body, or ``None`` to leave it as-is.
# The escape hatch for gateways that reject part of a vendor SDK's payload —
# e.g. an extra field in replayed tool-call history that a strict endpoint
# refuses. The transport takes care of rebuilding the request (Content-Length
# included); the hook only transforms bytes.
BodyRewrite = Callable[[httpx2.Request], "bytes | None"]


class GatewayTransport(
    RewriteMixin[httpx2.AsyncBaseTransport, httpx2.Headers, httpx2.Request],
    httpx2.AsyncBaseTransport,
):
    """Async transport that adapts vendor requests to the configured layout."""

    _http = httpx2

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Apply the rewrites, then delegate to the inner transport."""
        return await self.inner.handle_async_request(self._rewrite(request))

    async def aclose(self) -> None:
        """Close the wrapped inner transport."""
        await self.inner.aclose()


class GatewayTransportSync(
    RewriteMixin[httpx2.BaseTransport, httpx2.Headers, httpx2.Request],
    httpx2.BaseTransport,
):
    """Synchronous counterpart of :class:`GatewayTransport`."""

    _http = httpx2

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        """Apply the rewrites, then delegate to the inner transport."""
        return self.inner.handle_request(self._rewrite(request))

    def close(self) -> None:
        """Close the wrapped inner transport."""
        self.inner.close()
