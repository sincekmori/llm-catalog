# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""A LiteLLM ``CustomLLM`` that drives a catalog config through the gateway.

One implementation serves both deployment shapes:

* **in-process** — call :func:`llm_catalog.litellm.register` once to wire the
  module-level :data:`handler` into LiteLLM, then
  ``litellm.completion("examplegw/fast", ...)`` works; and
* **proxy** — the proxy config references ``llm_catalog.litellm.handler`` from
  its ``custom_provider_map`` (no ``register`` call needed).

Resolution is self-contained: the handler reads the catalog config JSON itself
(``LLM_CATALOG_CONFIG`` or ``llm-catalog.json``) and resolves the role/model
from it, so the proxy never needs gateway details in ``litellm_params``
(sidestepping LiteLLM issue #18216).

Upstream strategy (route 1 of the spec's §6.2)
----------------------------------------------
The handler reuses LiteLLM's own request building and response conversion: it
calls ``litellm.(a)completion`` against the matching **built-in** provider
(``anthropic`` / ``openai`` / ``gemini``) with a client whose transport is the
core :class:`~llm_catalog.core.GatewayTransport`. LiteLLM builds the native body
and parses the native response into OpenAI form; the transport rewrites the path
to the gateway's layout. (If a future LiteLLM/version stops honouring a custom
client, the documented fallback — §6.2 route 2 — is to convert native responses
to ``ModelResponse`` / ``GenericStreamingChunk`` by hand.)
"""

import json
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm import CustomLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from llm_catalog.core import (
    Catalog,
    GatewayTransport,
    GatewayTransportSync,
    ResolvedModel,
)
from llm_catalog.core.errors import ConfigError, ResolutionError

__all__ = ["CONFIG_ENV_VAR", "ChatCatalogLLM"]

# Environment variable naming the catalog config JSON (in-process & proxy).
CONFIG_ENV_VAR = "LLM_CATALOG_CONFIG"

# Default config location when CONFIG_ENV_VAR is unset.
_DEFAULT_CONFIG_PATH = "llm-catalog.json"

# catalog backend -> LiteLLM built-in provider prefix used for the inner call.
_BACKEND_TO_LITELLM: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
}


class ChatCatalogLLM(CustomLLM):
    """A catalog-driven LiteLLM custom provider (in-process and proxy)."""

    def __init__(
        self, config_path: str | Path | None = None, catalog: Catalog | None = None
    ) -> None:
        self._config_path = config_path
        self._catalog = catalog

    # --- config / resolution ------------------------------------------------

    def get_catalog(self) -> Catalog:
        """Return the catalog, loading its config JSON from disk on first use.

        This adapter is the deployment edge (the proxy points at a file via
        ``LLM_CATALOG_CONFIG``), so the file read lives here — core itself
        never touches the filesystem.
        """
        if self._catalog is None:
            path = Path(
                self._config_path
                or os.environ.get(CONFIG_ENV_VAR)
                or _DEFAULT_CONFIG_PATH
            )
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{path}: not valid JSON: {exc}") from exc
            self._catalog = Catalog(data)
        return self._catalog

    def set_catalog(self, catalog: Catalog) -> None:
        """Inject a catalog directly (used by the proxy launcher and tests)."""
        self._catalog = catalog

    def reload(self) -> None:
        """Drop the cached catalog so the next call re-reads it from disk."""
        self._catalog = None

    def _resolve(
        self, model: str, litellm_params: dict[str, Any] | None
    ) -> ResolvedModel:
        catalog = self.get_catalog()
        # A model string "{provider}/{name}" reaches us as name="{name}"; the
        # provider name comes through litellm_params.
        if model in catalog.config.roles:
            return catalog.resolve_role(model)
        provider_id = (litellm_params or {}).get("custom_llm_provider")
        if provider_id:
            return catalog.resolve_key(f"{provider_id}:{model}")
        if ":" in model:
            return catalog.resolve_key(model)
        raise ResolutionError(
            f'Cannot resolve "{model}": it is not a role and no provider was supplied.'
        )

    def _prep(
        self,
        model: str,
        optional_params: dict[str, Any] | None,
        litellm_params: dict[str, Any] | None,
    ) -> tuple[ResolvedModel, str, dict[str, Any]]:
        rm = self._resolve(model, litellm_params)
        prefix = _BACKEND_TO_LITELLM.get(rm.backend)
        if prefix is None:
            # The config schema accepts every ai-sdk-catalog backend so a shared
            # file validates as-is; this adapter can only drive these three.
            raise ResolutionError(
                f'Backend "{rm.backend}" (model "{rm.provider_id}:{rm.model_id}") '
                "is not supported by the LiteLLM adapter. Supported backends: "
                f"{sorted(_BACKEND_TO_LITELLM)}."
            )
        inner_model = f"{prefix}/{rm.slug}"
        # provider/model settings are defaults; the caller's params win.
        params: dict[str, Any] = {**rm.settings, **(optional_params or {})}
        params.pop("stream", None)  # set explicitly per method
        return rm, inner_model, params

    # --- client construction (transport carries the path rewrite) -----------

    def _async_client(self, rm: ResolvedModel) -> AsyncHTTPHandler | AsyncOpenAI:
        client = httpx.AsyncClient(
            transport=GatewayTransport(
                httpx.AsyncHTTPTransport(),
                base_url=rm.base_url,
                path_template=rm.path_template,
                backend=rm.backend,
                action_map=rm.action_map,
                slug=rm.slug,
            )
        )
        if rm.backend == "openai":
            return AsyncOpenAI(
                api_key=rm.api_key(), base_url=rm.base_url, http_client=client
            )
        handler = AsyncHTTPHandler()
        handler.client = client
        return handler

    def _sync_client(self, rm: ResolvedModel) -> HTTPHandler | OpenAI:
        client = httpx.Client(
            transport=GatewayTransportSync(
                httpx.HTTPTransport(),
                base_url=rm.base_url,
                path_template=rm.path_template,
                backend=rm.backend,
                action_map=rm.action_map,
                slug=rm.slug,
            )
        )
        if rm.backend == "openai":
            return OpenAI(
                api_key=rm.api_key(), base_url=rm.base_url, http_client=client
            )
        return HTTPHandler(client=client)

    # --- CustomLLM interface ------------------------------------------------

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: dict | None = None,
        logger_fn: Any = None,
        headers: dict | None = None,
        timeout: Any = None,
        client: Any = None,
    ) -> ModelResponse:
        rm, inner_model, params = self._prep(model, optional_params, litellm_params)
        return litellm.completion(
            model=inner_model,
            messages=messages,
            api_base=rm.base_url,
            api_key=rm.api_key(),
            client=self._sync_client(rm),
            **params,
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: dict | None = None,
        logger_fn: Any = None,
        headers: dict | None = None,
        timeout: Any = None,
        client: Any = None,
    ) -> ModelResponse:
        rm, inner_model, params = self._prep(model, optional_params, litellm_params)
        return await litellm.acompletion(
            model=inner_model,
            messages=messages,
            api_base=rm.base_url,
            api_key=rm.api_key(),
            client=self._async_client(rm),
            **params,
        )

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: dict | None = None,
        logger_fn: Any = None,
        headers: dict | None = None,
        timeout: Any = None,
        client: Any = None,
    ) -> Iterator[GenericStreamingChunk]:
        rm, inner_model, params = self._prep(model, optional_params, litellm_params)
        response = litellm.completion(
            model=inner_model,
            messages=messages,
            api_base=rm.base_url,
            api_key=rm.api_key(),
            client=self._sync_client(rm),
            stream=True,
            **params,
        )
        for chunk in response:
            yield _to_generic_chunk(chunk)

    async def astreaming(  # ty: ignore[invalid-method-override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: dict | None = None,
        logger_fn: Any = None,
        headers: dict | None = None,
        timeout: Any = None,
        client: Any = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        rm, inner_model, params = self._prep(model, optional_params, litellm_params)
        response = await litellm.acompletion(
            model=inner_model,
            messages=messages,
            api_base=rm.base_url,
            api_key=rm.api_key(),
            client=self._async_client(rm),
            stream=True,
            **params,
        )
        async for chunk in response:
            yield _to_generic_chunk(chunk)


def _to_generic_chunk(chunk: Any) -> GenericStreamingChunk:
    """Map a LiteLLM streaming chunk (OpenAI form) to a GenericStreamingChunk."""
    choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
    delta = getattr(choice, "delta", None)
    text = (getattr(delta, "content", None) or "") if delta is not None else ""
    finish = getattr(choice, "finish_reason", None) if choice is not None else None

    usage_block: ChatCompletionUsageBlock | None = None
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        usage_block = cast(
            "ChatCompletionUsageBlock",
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            },
        )

    return {
        "text": text,
        "tool_use": None,
        "is_finished": bool(finish),
        "finish_reason": finish or "",
        "usage": usage_block,
        "index": getattr(choice, "index", 0) if choice is not None else 0,
    }
