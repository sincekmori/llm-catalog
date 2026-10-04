# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Build native Pydantic AI models from a catalog.

:class:`PydanticAICatalog` wraps the core :class:`~llm_catalog.core.Catalog`
and, for a given role/key, constructs the right Pydantic AI ``Model`` +
``Provider``. A **gateway** model gets an ``httpx2.AsyncClient`` whose transport
is the core :class:`~llm_catalog.core.transport2.GatewayTransport` (path
rewriting plus the declarative ``headers``/``query``); a **direct** model calls
the vendor's own endpoint (or the vendor block's ``baseURL``), with the same
transport applying only the declarative extras. Nothing gateway-specific is
hardcoded — every quirk comes from the catalog config.

The HTTP client is ``httpx2`` (the Pydantic-maintained continuation of
``httpx``): Pydantic AI and the vendor SDKs it wraps (``openai>=3``,
``anthropic>=1``) are built on it, and the Anthropic SDK rejects an ``httpx``
client outright.
"""

from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Self, cast

import httpx2

from llm_catalog.core import Catalog, CatalogConfig, ResolvedModel
from llm_catalog.core.errors import LLMCatalogError
from llm_catalog.core.transport2 import BodyRewrite, GatewayTransport, HeaderRewrite
from pydantic_ai import (
    CodeExecutionTool,
    NativeOutput,
    PromptedOutput,
    ToolOutput,
    WebSearchTool,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

__all__ = ["PydanticAICatalog"]

# catalog capability grounding name -> Pydantic AI builtin tool factory.
#
# NOTE (§9, requires verification against the real gateway): whether Pydantic
# AI's builtin tool emits the *exact* tool variant your gateway expects is not
# guaranteed (standard web search vs an enterprise variant may differ). If it
# doesn't match, drive the tool through the raw vendor client on the provider.
_GROUNDING_TOOLS: dict[str, type[AbstractNativeTool]] = {
    "web_search": WebSearchTool,
    "url_context": WebSearchTool,  # closest builtin; verify the gateway's variant
    "code_execution": CodeExecutionTool,
}

# Translate ai-sdk-catalog camelCase / common setting names to Pydantic AI's
# ModelSettings keys. Unlisted keys pass through untouched (ModelSettings is a
# TypedDict, so extra keys are harmless at runtime).
_SETTINGS_ALIASES: dict[str, str] = {
    "maxOutputTokens": "max_tokens",
    "max_output_tokens": "max_tokens",
    "topP": "top_p",
    "topK": "top_k",
    "presencePenalty": "presence_penalty",
    "frequencyPenalty": "frequency_penalty",
    "stopSequences": "stop_sequences",
}


def _to_model_settings(settings: dict[str, Any]) -> ModelSettings | None:
    if not settings:
        return None
    translated: dict[str, Any] = {}
    for key, value in settings.items():
        # providerOptions has no ModelSettings equivalent; skip it here.
        if key in {"providerOptions", "provider_options"}:
            continue
        translated[_SETTINGS_ALIASES.get(key, key)] = value
    if not translated:
        return None
    return cast("ModelSettings", translated)


class PydanticAICatalog:
    """A thin Pydantic AI layer over the core :class:`Catalog`.

    Accepts a ready :class:`Catalog`, or anything :class:`Catalog` itself
    accepts — the mapping you parsed from your config JSON, or a validated
    :class:`CatalogConfig`:

        config = json.loads(Path("ai-sdk-catalog.json").read_text("utf-8"))
        async with PydanticAICatalog(config) as cat:
            agent = Agent(cat.model_for_role("fast"))
            ...

    Each model owns one ``httpx2`` client; models are built once and reused
    (so repeated lookups share a connection pool), and :meth:`aclose` (or the
    ``async with`` form) closes every client this catalog opened.

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
        self._models: dict[str, Model] = {}
        self._clients: list[httpx2.AsyncClient] = []

    @property
    def catalog(self) -> Catalog:
        """The underlying runtime-agnostic catalog (for capability queries)."""
        return self._catalog

    def model_for_role(self, role: str) -> Model:
        """Return the native Pydantic AI ``Model`` for a role."""
        return self._model(self._catalog.resolve_role(role))

    def model(self, key: str) -> Model:
        """Return the native Pydantic AI ``Model`` for a ``provider:model_id`` key."""
        return self._model(self._catalog.resolve_key(key))

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

    def output_for(
        self, role: str, schema: Any
    ) -> NativeOutput[Any] | ToolOutput[Any] | PromptedOutput[Any]:
        """Pick the output mode that matches the model's ``structured_output``.

        ``native`` -> ``NativeOutput``; ``tool`` -> ``ToolOutput``; ``prompted``
        -> ``PromptedOutput``. This lets a backend whose gateway policy rejects
        native structured output fall back to tool-mode by config alone.
        """
        mode = self._catalog.meta_for_role(role).capabilities.structured_output
        if mode == "native":
            return NativeOutput(schema)
        if mode == "tool":
            return ToolOutput(schema)
        return PromptedOutput(schema)

    def grounding_tools(self, role: str) -> list[AbstractNativeTool]:
        """Map the role's allowed grounding names to Pydantic AI builtin tools."""
        tools: list[AbstractNativeTool] = []
        for name in self._catalog.meta_for_role(role).capabilities.grounding:
            factory = _GROUNDING_TOOLS.get(name)
            if factory is None:
                raise LLMCatalogError(
                    f'Unknown grounding tool "{name}" for role "{role}". '
                    f"Known: {sorted(_GROUNDING_TOOLS)}."
                )
            tools.append(factory())
        return tools

    def _model(self, rm: ResolvedModel) -> Model:
        model = self._models.get(rm.key)
        if model is None:
            model = self._models[rm.key] = self._build(rm)
        return model

    def _build(self, rm: ResolvedModel) -> Model:
        settings = _to_model_settings(rm.settings)
        if rm.vendor == "anthropic":
            return self._anthropic(rm, settings)
        if rm.vendor == "openai":
            return self._openai(rm, settings)
        if rm.vendor == "openai-compatible":
            return self._openai_compatible(rm, settings)
        if rm.vendor == "google":
            return self._google(rm, settings)
        # The config schema accepts every ai-sdk-catalog vendor so a shared
        # file validates as-is; this adapter can only drive these four.
        raise LLMCatalogError(
            f'Vendor "{rm.vendor}" (model "{rm.key}") is '
            "not supported by the Pydantic AI adapter. Supported vendors: "
            '"anthropic", "openai", "openai-compatible", "google".'
        )

    def _client(self, rm: ResolvedModel) -> httpx2.AsyncClient:
        """Build the httpx2 client; one transport serves both provider kinds.

        A gateway model gets the path rewrite; a direct model only the
        declarative headers/query (and the code-level hooks).
        """
        transport = GatewayTransport.for_model(
            self._transport_factory(),
            rm,
            header_rewrite=self._header_rewrite,
            body_rewrite=self._body_rewrite,
        )
        client = httpx2.AsyncClient(transport=transport)
        self._clients.append(client)
        return client

    def _provider_kwargs(self, rm: ResolvedModel) -> dict[str, Any]:
        """Build the common Pydantic AI provider kwargs.

        ``base_url``/``api_key`` are omitted when unset, so a direct provider
        falls back to the vendor SDK's own defaults (its endpoint, its key env
        var — e.g. ``OPENAI_API_KEY``). For a gateway model both are always
        present (the key defaults to ``AI_GATEWAY_API_KEY``).
        """
        kwargs: dict[str, Any] = {}
        if rm.base_url is not None:
            kwargs["base_url"] = rm.base_url
        api_key = rm.api_key()
        if api_key is not None:
            kwargs["api_key"] = api_key
        # Opened last: a missing key env var must not leave a client behind.
        kwargs["http_client"] = self._client(rm)
        return kwargs

    def _anthropic(
        self, rm: ResolvedModel, settings: ModelSettings | None
    ) -> AnthropicModel:
        provider = AnthropicProvider(**self._provider_kwargs(rm))
        return AnthropicModel(rm.model_id, provider=provider, settings=settings)

    def _openai(self, rm: ResolvedModel, settings: ModelSettings | None) -> Model:
        if rm.api == "completion":
            raise LLMCatalogError(
                f'Model "{rm.key}" sets api="completion"; '
                "the legacy Completions API is not supported by the Pydantic AI "
                "adapter."
            )
        provider = OpenAIProvider(**self._provider_kwargs(rm))
        if rm.api == "chat":
            return OpenAIChatModel(rm.model_id, provider=provider, settings=settings)
        # The OpenAI vendor's default surface is the Responses API, matching
        # ai-sdk-catalog; set api="chat" for a gateway that only speaks Chat
        # Completions.
        return OpenAIResponsesModel(rm.model_id, provider=provider, settings=settings)

    def _openai_compatible(
        self, rm: ResolvedModel, settings: ModelSettings | None
    ) -> Model:
        if rm.api in {"responses", "completion"}:
            raise LLMCatalogError(
                f'Model "{rm.key}" sets api="{rm.api}"; '
                'an "openai-compatible" vendor only speaks Chat Completions in '
                "the Pydantic AI adapter."
            )
        provider = OpenAIProvider(**self._provider_kwargs(rm))
        return OpenAIChatModel(rm.model_id, provider=provider, settings=settings)

    def _google(self, rm: ResolvedModel, settings: ModelSettings | None) -> GoogleModel:
        # NOTE (§9, requires verification): some google-genai versions ignore a
        # custom HTTP client/transport. If GatewayTransport turns out not to
        # take effect here, the fallback is to build a genai Client with the
        # transport wired in and pass it via GoogleProvider(client=...).
        provider = GoogleProvider(**self._provider_kwargs(rm))
        return GoogleModel(rm.model_id, provider=provider, settings=settings)
