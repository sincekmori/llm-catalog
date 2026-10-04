# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Build native AI SDK for Python models from a catalog.

:class:`AISDKCatalog` wraps the core :class:`~llm_catalog.core.Catalog` and, for
a given role/key, constructs an ``ai.Model`` bound to the right provider and
wire protocol. A **gateway** model gets an ``httpx2.AsyncClient`` whose
transport is the core :class:`~llm_catalog.core.transport2.GatewayTransport`
(path rewriting plus the declarative ``headers``/``query``); a **direct** model
calls the vendor's own endpoint (or the vendor block's ``baseURL``), with the
same transport applying only the declarative extras. Nothing gateway-specific
is hardcoded — every quirk comes from the catalog config.

The AI SDK for Python (``ai``) is in public beta, so everything that touches it
lives in this one module and uses only its documented surface:
``ai.get_provider`` (with ``base_url`` / ``api_key`` / ``client``), ``ai.Model``
(with an explicit ``protocol``), the three exported wire protocols, and
``ai.InferenceRequestParams``. The official OpenAI / Anthropic SDK clients are
built by ``ai`` itself from the ``httpx2`` client handed to it; this module
never imports those SDKs.

Vendors and call surfaces
-------------------------
``ai`` speaks three wire protocols directly — Anthropic Messages, OpenAI
Responses and OpenAI Chat Completions — which covers the ``anthropic``,
``openai`` and ``openai-compatible`` vendors (see ``_SURFACES``). Any other
vendor, and any ``api`` a vendor does not offer here, raises
:class:`~llm_catalog.core.errors.LLMCatalogError` when the model is built;
nothing is silently routed to a different API. The other vendors of the shared
schema (``google``, ``mistral``, ``groq``, ...) have no native protocol in
``ai``; declare an endpoint that speaks Chat Completions as an
``openai-compatible`` vendor (or backend) instead.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import ai
import httpx2
from ai.providers.anthropic import AnthropicMessagesProtocol
from ai.providers.openai import OpenAIChatCompletionsProtocol, OpenAIResponsesProtocol

from llm_catalog.core import Catalog, CatalogConfig, ResolvedModel
from llm_catalog.core.errors import LLMCatalogError
from llm_catalog.core.transport2 import BodyRewrite, GatewayTransport, HeaderRewrite

__all__ = ["AISDKCatalog"]


@dataclass(frozen=True)
class _Surface:
    """One of ``ai``'s wire protocols and how this adapter drives it."""

    # Which official SDK `ai` wraps ("openai" / "anthropic"). It only selects
    # the SDK: the endpoint, key and transport come from the catalog.
    provider_id: str
    protocol: type[ai.ProviderProtocol[Any]]
    # The API's own name for stop sequences (`ai` has no typed parameter, so
    # they travel as an extra body field); None when the API has none.
    stop_field: str | None


_MESSAGES = _Surface("anthropic", AnthropicMessagesProtocol, "stop_sequences")
_RESPONSES = _Surface("openai", OpenAIResponsesProtocol, None)
_CHAT = _Surface("openai", OpenAIChatCompletionsProtocol, "stop")

# vendor -> `api` (None when omitted) -> surface. The defaults and aliases
# follow ai-sdk-catalog: OpenAI defaults to the Responses API, an
# OpenAI-compatible server only speaks Chat Completions, and a single-surface
# vendor (Anthropic) exposes "chat" as an alias of its one surface.
_SURFACES: dict[str, dict[str | None, _Surface]] = {
    "anthropic": {None: _MESSAGES, "chat": _MESSAGES},
    "openai": {None: _RESPONSES, "responses": _RESPONSES, "chat": _CHAT},
    "openai-compatible": {None: _CHAT, "chat": _CHAT},
}


def _surface(rm: ResolvedModel) -> _Surface:
    """Pick the wire protocol for a resolved model, or raise."""
    by_api = _SURFACES.get(rm.vendor)
    if by_api is None:
        # The config schema accepts every ai-sdk-catalog vendor so a shared
        # file validates as-is; this adapter can only drive the ones `ai` has
        # a native wire protocol for.
        raise LLMCatalogError(
            f'Vendor "{rm.vendor}" (model "{rm.key}") is not supported by the AI '
            f"SDK for Python adapter. Supported vendors: {sorted(_SURFACES)}. An "
            "endpoint that speaks Chat Completions can be declared as an "
            '"openai-compatible" vendor instead.'
        )
    surface = by_api.get(rm.api)
    if surface is None:
        offered = sorted(api for api in by_api if api is not None)
        raise LLMCatalogError(
            f'Model "{rm.key}" sets api="{rm.api}", which the "{rm.vendor}" vendor '
            "does not offer in the AI SDK for Python adapter (omit it, or use one "
            f"of {offered})."
        )
    return surface


def _to_params(rm: ResolvedModel) -> ai.InferenceRequestParams:
    """Translate the merged catalog settings into ``ai`` request params.

    ``providerOptions`` is skipped: its entries are options of the TypeScript
    AI SDK providers and have no counterpart in ``ai``'s wire protocols.
    """
    settings = rm.settings
    samplers: list[Any] = []
    if "temperature" in settings:
        samplers.append(
            ai.TemperatureSamplerParams(temperature=settings["temperature"])
        )
    if "topP" in settings:
        samplers.append(ai.TopPSamplerParams(top_p=settings["topP"]))
    if "topK" in settings:
        samplers.append(ai.TopKSamplerParams(top_k=settings["topK"]))
    penalties = {
        name: settings[key]
        for key, name in (
            ("presencePenalty", "presence_penalty"),
            ("frequencyPenalty", "frequency_penalty"),
        )
        if key in settings
    }
    if penalties:
        samplers.append(ai.RepetitionPenaltyParams(**penalties))
    if "seed" in settings:
        samplers.append(ai.SeedSamplerParams(seed=settings["seed"]))

    kwargs: dict[str, Any] = {}
    if samplers:
        kwargs["sampling"] = {type(sampler): sampler for sampler in samplers}
    if "maxOutputTokens" in settings:
        kwargs["output"] = ai.OutputParams(max_tokens=settings["maxOutputTokens"])
    if "stopSequences" in settings:
        stop_field = _surface(rm).stop_field
        if stop_field is None:
            raise LLMCatalogError(
                f'Model "{rm.key}" sets "stopSequences", but the OpenAI Responses '
                "API has no stop-sequence parameter. Drop the setting or use "
                'api="chat".'
            )
        kwargs["extra_body"] = {stop_field: settings["stopSequences"]}
    return ai.InferenceRequestParams(**kwargs)


class AISDKCatalog:
    """A thin AI SDK for Python layer over the core :class:`Catalog`.

    Accepts a ready :class:`Catalog`, or anything :class:`Catalog` itself
    accepts — the mapping you parsed from your config JSON, or a validated
    :class:`CatalogConfig`:

        config = json.loads(Path("ai-sdk-catalog.json").read_text("utf-8"))
        async with AISDKCatalog(config) as cat:
            model = cat.model_for_role("fast")
            params = cat.params_for_role("fast")
            async with ai.stream(model, messages, params=params) as stream:
                ...

    ``ai.Model`` carries no default call settings, so the catalog's merged
    ``settings`` come back separately from :meth:`params_for_role` /
    :meth:`params` as an ``ai.InferenceRequestParams`` to pass (or refine) per
    call.

    Each model owns one ``httpx2`` client; models are built once and reused, and
    :meth:`aclose` (or the ``async with`` form) closes every client this catalog
    opened.

    ``header_rewrite`` / ``body_rewrite`` are passed through to every
    :class:`~llm_catalog.core.transport2.GatewayTransport` this catalog builds
    — the escape hatch for gateways that need a header tweaked or part of a
    vendor payload adjusted. Both hooks run after the URL rewrite, so they can
    target a specific gateway path via ``request.url``. They receive
    ``httpx2`` objects (``httpx2.Headers`` / ``httpx2.Request``).

    ``transport_factory`` builds the network transport each
    ``GatewayTransport`` wraps (default: ``httpx2.AsyncHTTPTransport``) — pass
    one to route through a proxy, set mTLS or connection limits, or to
    substitute ``httpx2.MockTransport`` in tests.
    """

    def __init__(
        self,
        catalog: Catalog | CatalogConfig | Mapping[str, Any],
        *,
        header_rewrite: HeaderRewrite | None = None,
        body_rewrite: BodyRewrite | None = None,
        transport_factory: Callable[[], httpx2.AsyncBaseTransport] | None = None,
    ) -> None:
        self._catalog = catalog if isinstance(catalog, Catalog) else Catalog(catalog)
        self._header_rewrite = header_rewrite
        self._body_rewrite = body_rewrite
        self._transport_factory = transport_factory or httpx2.AsyncHTTPTransport
        self._models: dict[str, ai.Model] = {}
        self._clients: list[httpx2.AsyncClient] = []

    @property
    def catalog(self) -> Catalog:
        """The underlying runtime-agnostic catalog (for metadata queries)."""
        return self._catalog

    def model_for_role(self, role: str) -> ai.Model:
        """Return the ``ai.Model`` for a role."""
        return self._model(self._catalog.resolve_role(role))

    def model(self, key: str) -> ai.Model:
        """Return the ``ai.Model`` for a ``provider:model_id`` key."""
        return self._model(self._catalog.resolve_key(key))

    def params_for_role(self, role: str) -> ai.InferenceRequestParams:
        """Return the role's default call settings as ``ai`` request params."""
        return _to_params(self._catalog.resolve_role(role))

    def params(self, key: str) -> ai.InferenceRequestParams:
        """Return a ``provider:model_id`` key's default call settings."""
        return _to_params(self._catalog.resolve_key(key))

    async def aclose(self) -> None:
        """Close every HTTP client this catalog opened and forget its models."""
        clients, self._clients = self._clients, []
        self._models.clear()
        for client in clients:
            await client.aclose()

    async def __aenter__(self) -> Self:
        """Enter the context; the catalog is usable without it too."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the HTTP clients on exit."""
        await self.aclose()

    def _model(self, rm: ResolvedModel) -> ai.Model:
        model = self._models.get(rm.key)
        if model is None:
            model = self._models[rm.key] = self._build(rm)
        return model

    def _build(self, rm: ResolvedModel) -> ai.Model:
        # Resolve everything that can fail first, so an unsupported vendor/api
        # or a missing key never leaves an opened client behind.
        surface = _surface(rm)
        # `base_url`/`api_key` are omitted when unset, so a direct provider
        # falls back to `ai`'s own defaults (the vendor endpoint, its key env
        # var — e.g. OPENAI_API_KEY). For a gateway model both are always
        # present (the key defaults to AI_GATEWAY_API_KEY).
        kwargs: dict[str, Any] = {}
        if rm.base_url is not None:
            kwargs["base_url"] = rm.base_url
        api_key = rm.api_key()
        if api_key is not None:
            kwargs["api_key"] = api_key
        transport = GatewayTransport.for_model(
            self._transport_factory(),
            rm,
            header_rewrite=self._header_rewrite,
            body_rewrite=self._body_rewrite,
        )
        client = httpx2.AsyncClient(transport=transport)
        self._clients.append(client)
        provider = ai.get_provider(surface.provider_id, client=client, **kwargs)
        return ai.Model(id=rm.model_id, provider=provider, protocol=surface.protocol())
