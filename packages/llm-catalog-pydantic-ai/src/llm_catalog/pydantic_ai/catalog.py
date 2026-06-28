# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Build native Pydantic AI models from a catalog, routed through the gateway.

:class:`PydanticAICatalog` wraps the core :class:`~llm_catalog.core.Catalog` and,
for a given role/key, constructs the right Pydantic AI ``Model`` + ``Provider``
with an ``httpx.AsyncClient`` whose transport is the core
:class:`~llm_catalog.core.GatewayTransport`. Nothing gateway-specific is
hardcoded — every quirk comes from ``catalog.yaml``.
"""

from pathlib import Path
from typing import Any, cast

import httpx

from llm_catalog.core import Catalog, GatewayTransport, ResolvedModel
from llm_catalog.core.errors import LLMCatalogError
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
    """A thin Pydantic AI layer over the core :class:`Catalog`."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    @classmethod
    def from_file(cls, path: str | Path) -> "PydanticAICatalog":
        """Load ``catalog.yaml`` and wrap it."""
        return cls(Catalog.from_file(path))

    @property
    def catalog(self) -> Catalog:
        """The underlying gateway-agnostic catalog (for capability queries)."""
        return self._catalog

    def model_for_role(self, role: str) -> Model:
        """Build a native Pydantic AI ``Model`` for a role."""
        return self._build(self._catalog.resolve_role(role))

    def model(self, key: str) -> Model:
        """Build a native Pydantic AI ``Model`` for a ``provider:model_id`` key."""
        return self._build(self._catalog.resolve_key(key))

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

    def _build(self, rm: ResolvedModel) -> Model:
        settings = _to_model_settings(rm.settings)
        if rm.backend == "anthropic":
            return self._anthropic(rm, settings)
        if rm.backend == "openai":
            return self._openai(rm, settings)
        if rm.backend == "google":
            return self._google(rm, settings)
        # Unreachable: backend is validated to one of three by core.
        raise LLMCatalogError(f'Unsupported backend "{rm.backend}".')

    def _client(self, rm: ResolvedModel) -> httpx.AsyncClient:
        transport = GatewayTransport(
            httpx.AsyncHTTPTransport(),
            base_url=rm.base_url,
            path_template=rm.path_template,
            backend=rm.backend,
            action_map=rm.action_map,
            slug=rm.slug,
        )
        return httpx.AsyncClient(transport=transport)

    def _anthropic(
        self, rm: ResolvedModel, settings: ModelSettings | None
    ) -> AnthropicModel:
        provider = AnthropicProvider(
            base_url=rm.base_url, api_key=rm.api_key(), http_client=self._client(rm)
        )
        return AnthropicModel(rm.model_id, provider=provider, settings=settings)

    def _openai(self, rm: ResolvedModel, settings: ModelSettings | None) -> Model:
        provider = OpenAIProvider(
            base_url=rm.base_url, api_key=rm.api_key(), http_client=self._client(rm)
        )
        if rm.api == "responses":
            return OpenAIResponsesModel(
                rm.model_id, provider=provider, settings=settings
            )
        return OpenAIChatModel(rm.model_id, provider=provider, settings=settings)

    def _google(self, rm: ResolvedModel, settings: ModelSettings | None) -> GoogleModel:
        # NOTE (§9, requires verification): some google-genai versions ignore a
        # custom httpx client/transport. If GatewayTransport turns out not to
        # take effect here, the fallback is to build a genai Client with the
        # transport wired in and pass it via GoogleProvider(client=...).
        provider = GoogleProvider(
            base_url=rm.base_url, api_key=rm.api_key(), http_client=self._client(rm)
        )
        return GoogleModel(rm.model_id, provider=provider, settings=settings)
