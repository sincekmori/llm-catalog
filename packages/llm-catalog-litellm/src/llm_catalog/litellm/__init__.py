# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM adapter for llm-catalog.

Setup is explicit: call :func:`register` once to wire the module-level
:data:`handler` into LiteLLM for every provider id in your catalog config.

    from llm_catalog.litellm import register

    register()

    import litellm
    litellm.completion(model="examplegw/fast", messages=[...])

The same :data:`handler` instance is referenced by the proxy config via
``custom_handler: llm_catalog.litellm.handler``; the proxy does not need
:func:`register`.

import namespace: ``llm_catalog.litellm``
"""

import warnings
from typing import Any

import litellm

from .custom_llm import CONFIG_ENV_VAR, ChatCatalogLLM

__all__ = [
    "CONFIG_ENV_VAR",
    "ChatCatalogLLM",
    "ProviderIdCollisionWarning",
    "handler",
    "register",
]


class ProviderIdCollisionWarning(UserWarning):
    """A provider ``id`` collides with a built-in LiteLLM provider name.

    LiteLLM silently bypasses a ``custom_provider_map`` entry whose name clashes
    with a built-in provider (upstream issue #23352). We surface this as a
    warning at :func:`register` time so the collision is caught before it
    manifests as a confusing routing bug.

    Note the collision only matters for this adapter's ``custom_provider_map``
    routing: a *direct* provider naturally named after its vendor (e.g. a plain
    ``{"id": "openai"}``) is perfectly valid config — but calls routed via
    ``litellm.completion("openai/...")`` will hit LiteLLM's built-in provider,
    not this handler, so the catalog's overrides would be ignored.
    """


# Built-in LiteLLM provider names. A custom provider id that collides with one
# of these is silently bypassed by LiteLLM's custom_provider_map (upstream issue
# #23352), so register() warns. This list is intentionally conservative — it
# covers the common built-ins; it does not need to be exhaustive to be useful.
_LITELLM_BUILTIN_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "azure",
        "azure_ai",
        "anthropic",
        "bedrock",
        "vertex_ai",
        "vertex_ai_beta",
        "gemini",
        "google",
        "palm",
        "cohere",
        "cohere_chat",
        "mistral",
        "groq",
        "xai",
        "deepseek",
        "perplexity",
        "ollama",
        "ollama_chat",
        "openrouter",
        "fireworks_ai",
        "together_ai",
        "deepinfra",
        "cerebras",
        "replicate",
        "huggingface",
        "nlp_cloud",
        "ai21",
        "voyage",
        "databricks",
        "watsonx",
        "sagemaker",
        "cloudflare",
        "nvidia_nim",
        "sambanova",
    }
)

# The single shared instance: registered by register() and referenced by the proxy.
handler = ChatCatalogLLM()

# Provider ids we've already wired, so register() is idempotent across calls.
_registered: set[str] = set()


def register() -> None:
    """Register :data:`handler` for each provider id in the catalog.

    Loads the catalog config JSON (via ``LLM_CATALOG_CONFIG`` or the default
    ``llm-catalog.json``) and adds a ``custom_provider_map`` entry per provider
    id. Idempotent: calling it repeatedly is safe and adds nothing the second
    time. Raises if no catalog can be loaded, so a misconfigured setup fails
    loudly rather than silently. A provider id that collides with a built-in
    LiteLLM provider name triggers a :class:`ProviderIdCollisionWarning`.
    """
    catalog = handler.get_catalog()
    existing = {entry.get("provider") for entry in litellm.custom_provider_map}
    for provider in catalog.config.providers:
        pid = provider.id
        if pid in _LITELLM_BUILTIN_PROVIDERS:
            warnings.warn(
                f'Provider id "{pid}" collides with a built-in LiteLLM provider '
                "name; LiteLLM may silently bypass the custom handler (issue "
                f'#23352). Calls to "{pid}/..." will use LiteLLM\'s built-in '
                "provider, ignoring this catalog's config for it.",
                ProviderIdCollisionWarning,
                stacklevel=2,
            )
        if pid in _registered or pid in existing:
            _registered.add(pid)
            continue
        entry: Any = {"provider": pid, "custom_handler": handler}
        litellm.custom_provider_map.append(entry)
        _registered.add(pid)
