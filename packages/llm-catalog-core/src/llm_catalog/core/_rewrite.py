# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The request rewrite shared by the ``httpx`` and ``httpx2`` transports.

Each vendor SDK (anthropic / openai / google-genai / ...) builds its own fixed
request path (``/v1/messages``, ``/chat/completions``,
``/v1beta/models/{m}:{action}``). A gateway instead expects a ``path_template``
such as ``anthropic/{slug}``.

A ``GatewayTransport`` sits in the vendor SDK's HTTP client and rewrites each
outgoing request's path to the gateway's, leaving everything else (method,
body, query string) intact. It also injects the config-declared transport
extras — extra request ``headers`` (same-name wins over the SDK's own) and
``query`` parameters appended to the final URL — mirroring ai-sdk-catalog's
``headers``/``query``. It lives in *core* so every adapter reuses the exact same
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

Two HTTP libraries
------------------
``httpx2`` is the Pydantic-maintained continuation of ``httpx``: the same API
under a different import name, with its own (incompatible) ``Request`` / ``URL``
/ ``Headers`` classes. The vendor SDKs are split between the two, so the
rewrite is written once here against the API both share, and each transport
module (:mod:`~llm_catalog.core.transport` for ``httpx``,
:mod:`~llm_catalog.core.transport2` for ``httpx2``) binds it to its library:
the type parameters of :class:`RewriteMixin` give the public constructor its
precise types, and :attr:`RewriteMixin._http` hands the rewrite the imported
module. Inside this module requests and the library handle are typed ``Any``.
"""

import json
import re
from collections.abc import Callable
from typing import Any, ClassVar, Self

from .resolve import ResolvedModel

__all__ = ["RewriteMixin"]

# Matches the model id and operation in a google-genai URL path, e.g.
# "/v1beta/models/gemini-3.8-flash:streamGenerateContent".
_GOOGLE_PATH = re.compile(r"/models/(?P<model>[^:/]+):(?P<action>[A-Za-z]+)")


def _extract_slug_action(
    http: Any, request: Any, vendor: str
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
    except http.RequestNotRead:  # streaming body not materialised — uncommon here
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
    http: Any,
    request: Any,
    *,
    base_url: str,
    path_template: str,
    action_map: dict[str, str],
    vendor: str,
    slug: str | None,
) -> Any:
    """Compute the rewritten gateway URL, or ``None`` to leave the request as-is."""
    if slug is not None and vendor != "google":
        # The body would only yield a slug, and the configured one wins anyway:
        # skip parsing a potentially large payload on every request.
        final_slug, action = slug, None
    else:
        extracted_slug, action = _extract_slug_action(http, request, vendor)
        final_slug = slug if slug is not None else extracted_slug
    if final_slug is None:
        # Nothing to substitute — don't touch the request.
        return None

    mapped_action = action_map.get(action, action) if action is not None else ""
    new_path = path_template.format(slug=final_slug, action=mapped_action)

    base = base_url.rstrip("/")
    path = new_path.lstrip("/")
    # Plain concatenation (not URL.join) so a gateway base path like
    # ".../base" is preserved rather than treated as a sibling to replace.
    url = http.URL(f"{base}/{path}")
    # Re-attach the original query string (e.g. google's ?alt=sse).
    return url.copy_merge_params(request.url.params)


def _apply_body_rewrite(
    http: Any, request: Any, rewrite: Callable[[Any], bytes | None]
) -> Any:
    """Run the body hook and rebuild the request when it returns a new body.

    Rebuilding (rather than mutating) keeps ``Content-Length`` correct: the
    stale header is dropped and the library recomputes it for the new content.
    Streaming uploads (body not materialised) are left untouched.
    """
    try:
        _ = request.content
    except http.RequestNotRead:  # pragma: no cover - unusual for JSON APIs
        return request
    new_body = rewrite(request)
    if new_body is None:
        return request
    headers = http.Headers(
        [(k, v) for k, v in request.headers.raw if k.lower() != b"content-length"]
    )
    return http.Request(request.method, request.url, headers=headers, content=new_body)


class RewriteMixin[InnerT, HeadersT, RequestT]:
    """The constructor and request rewrite shared by every transport flavour.

    The type parameters are the bound library's inner transport, ``Headers``
    and ``Request`` classes; a concrete transport fixes them and sets
    :attr:`_http` to the library module.
    """

    # The HTTP library the concrete transport is built on (httpx or httpx2).
    _http: ClassVar[Any]

    def __init__(
        self,
        inner: InnerT,
        base_url: str | None = None,
        path_template: str | None = None,
        vendor: str | None = None,
        action_map: dict[str, str] | None = None,
        slug: str | None = None,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
        header_rewrite: Callable[[HeadersT], None] | None = None,
        body_rewrite: Callable[[RequestT], bytes | None] | None = None,
    ) -> None:
        self.inner = inner
        self.base_url = base_url
        self.path_template = path_template
        self.vendor = vendor
        self.action_map = action_map or {}
        self.slug = slug
        self.headers = headers or {}
        self.query = query or {}
        self.header_rewrite = header_rewrite
        self.body_rewrite = body_rewrite

    @classmethod
    def for_model(
        cls,
        inner: InnerT,
        model: ResolvedModel,
        *,
        header_rewrite: Callable[[HeadersT], None] | None = None,
        body_rewrite: Callable[[RequestT], bytes | None] | None = None,
    ) -> Self:
        """Build the transport for a resolved model.

        A gateway model gets the path rewrite; a direct model
        (``path_template`` is ``None``) only the declarative headers/query and
        the code-level hooks. Header values are resolved here, so this reads
        the environment (see :meth:`ResolvedModel.resolved_headers`).
        """
        return cls(
            inner,
            base_url=model.base_url,
            path_template=model.path_template,
            vendor=model.vendor,
            action_map=model.action_map,
            slug=model.slug,
            headers=model.resolved_headers(),
            query=model.query,
            header_rewrite=header_rewrite,
            body_rewrite=body_rewrite,
        )

    def _rewrite(self, request: Any) -> Any:
        # 1. Path rewrite to the gateway layout (gateway providers only).
        if (
            self.base_url is not None
            and self.path_template is not None
            and self.vendor is not None
        ):
            new_url = _build_url(
                self._http,
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
            request = _apply_body_rewrite(self._http, request, self.body_rewrite)
        return request
