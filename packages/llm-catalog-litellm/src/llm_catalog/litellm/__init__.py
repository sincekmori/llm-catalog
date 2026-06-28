# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM adapter for llm-catalog.

Setup is explicit: call :func:`register` once to wire the module-level
:data:`handler` into LiteLLM for every provider id in your ``catalog.yaml``.

    from llm_catalog.litellm import register

    register()

    import litellm
    litellm.completion(model="examplegw/fast", messages=[...])

The same :data:`handler` instance is referenced by the proxy ``config.yaml`` via
``custom_handler: llm_catalog.litellm.handler``; the proxy does not need
:func:`register`.

import namespace: ``llm_catalog.litellm``
"""

from typing import Any

import litellm

from .custom_llm import CONFIG_ENV_VAR, ChatCatalogLLM

__all__ = ["CONFIG_ENV_VAR", "ChatCatalogLLM", "handler", "register"]

# The single shared instance: registered by register() and referenced by the proxy.
handler = ChatCatalogLLM()

# Provider ids we've already wired, so register() is idempotent across calls.
_registered: set[str] = set()


def register() -> None:
    """Register :data:`handler` for each provider id in the catalog.

    Loads ``catalog.yaml`` (via ``LLM_CATALOG_CONFIG`` or the default path) and
    adds a ``custom_provider_map`` entry per provider id. Idempotent: calling it
    repeatedly is safe and adds nothing the second time. Raises if no catalog can
    be loaded, so a misconfigured setup fails loudly rather than silently.
    """
    catalog = handler.get_catalog()
    existing = {entry.get("provider") for entry in litellm.custom_provider_map}
    for provider in catalog.config.providers:
        pid = provider.id
        if pid in _registered or pid in existing:
            _registered.add(pid)
            continue
        entry: Any = {"provider": pid, "custom_handler": handler}
        litellm.custom_provider_map.append(entry)
        _registered.add(pid)
