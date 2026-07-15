# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The resolved view of a single model, produced by :class:`Catalog`.

A :class:`ResolvedModel` is a flat, adapter-agnostic snapshot of everything
needed to call one model: the kind (``direct`` or ``gateway``), the vendor it
speaks, the addressing (backend key, slug, api), the endpoint (``base_url`` and
— for a gateway — the path template), the key source, the merged transport
extras (``headers``/``query``), the merged call settings, and the capability
hints.

It deliberately holds no Pydantic AI / LiteLLM types and no production secrets —
keys and env-var-backed header values are read from the environment lazily, at
:meth:`ResolvedModel.api_key` / :meth:`ResolvedModel.resolved_headers` time.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import (
    API_KEY_PLACEHOLDER,
    EnvVarRef,
    HeaderValue,
    ModelCapabilities,
)
from .errors import ResolutionError

__all__ = ["ResolvedModel"]


def _read_env(name: str, needed_for: str) -> str:
    try:
        return os.environ[name]
    except KeyError as exc:
        raise KeyError(
            f'environment variable "{name}" is not set (needed for {needed_for})'
        ) from exc


@dataclass(frozen=True)
class ResolvedModel:
    """Flat description of one model. Adapter-agnostic.

    ``vendor`` is the resolved vendor id: for a gateway model, its backend's
    ``vendor``; for a direct model, the vendor block's ``id`` (defaulting to
    the provider id). A direct provider whose id is not a known vendor still
    resolves — the adapters reject it at use time.

    ``api_key_config`` is the raw config value (a literal, an
    :class:`~llm_catalog.core.EnvVarRef`, or ``None``);
    ``default_api_key_env`` is the environment variable read when it is
    ``None`` (``AI_GATEWAY_API_KEY`` for gateway providers, ``None`` for
    direct providers — the vendor SDK's own default then applies).
    """

    kind: Literal["direct", "gateway"]
    provider_id: str
    model_id: str
    vendor: str
    api: str | None = None
    base_url: str | None = None
    api_key_config: str | EnvVarRef | None = None
    default_api_key_env: str | None = None
    name: str | None = None  # openai-compatible metadata namespace
    # gateway-only addressing
    backend: str | None = None  # gateway.backends key
    slug: str | None = None  # path segment (defaults to model_id)
    path_template: str | None = None
    action_map: dict[str, str] = field(default_factory=dict)
    # transport extras (for a gateway: gateway-level merged with backend-level)
    headers: dict[str, HeaderValue] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def api_key(self) -> str | None:
        """Return the API key, lazily and per call.

        A literal ``apiKey`` is returned as-is; an ``{"envVarName": ...}``
        reference reads that environment variable (raising :class:`KeyError`
        with an actionable message when unset). With no key configured, the
        ``default_api_key_env`` fallback is read (gateway providers); direct
        providers return ``None`` so the vendor SDK's own default applies.
        """
        needed_for = f'provider "{self.provider_id}"'
        if isinstance(self.api_key_config, str):
            return self.api_key_config
        if isinstance(self.api_key_config, EnvVarRef):
            return _read_env(self.api_key_config.env_var_name, needed_for)
        if self.default_api_key_env is not None:
            return _read_env(self.default_api_key_env, needed_for)
        return None

    def resolved_headers(self) -> dict[str, str]:
        """Resolve the configured headers to concrete string values.

        Env-var-backed values are read from the environment (raising
        :class:`KeyError` when unset) and the ``{apiKey}`` placeholder in
        inline values is replaced with :meth:`api_key`. Call this lazily,
        when the model is actually used.
        """
        resolved: dict[str, str] = {}
        for name, value in self.headers.items():
            if isinstance(value, EnvVarRef):
                resolved[name] = _read_env(
                    value.env_var_name,
                    f'header "{name}" of provider "{self.provider_id}"',
                )
                continue
            final = value
            if API_KEY_PLACEHOLDER in value:
                key = self.api_key()
                if key is None:
                    raise ResolutionError(
                        f'Header "{name}" of provider "{self.provider_id}" uses '
                        f'"{API_KEY_PLACEHOLDER}", but no "apiKey" is configured.'
                    )
                final = value.replace(API_KEY_PLACEHOLDER, key)
            resolved[name] = final
        return resolved
