# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Declarative ``catalog.yaml`` schema, loader, and validation.

This is the single source of truth for providers, the models they serve, and the
roles an application references. It mirrors the structure of the TypeScript
``ai-sdk-catalog`` config so the *same* file can drive both ecosystems: the
camelCase keys (``baseURL``, ``apiKeyEnvVarName``, ``pathTemplate``,
``actionMap``) are accepted via Pydantic aliases, while ``populate_by_name``
also lets Python code build models with snake_case names.

Unlike the TypeScript schema, this Python schema models **gateway providers
only** — every provider routes through a gateway and every model names a
``backend``. Direct/resolver providers are out of scope here.

Secrets never live in config: a model carries only ``api_key_env`` (the *name*
of an environment variable), never the key value itself.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ConfigError, ProviderIdCollisionWarning

__all__ = [
    "Backend",
    "CatalogConfig",
    "Gateway",
    "ModelCapabilities",
    "ModelEntry",
    "Provider",
    "RoleRef",
    "load_config",
    "parse_config",
]

# Built-in LiteLLM provider names. A custom provider id that collides with one of
# these is silently bypassed by LiteLLM's custom_provider_map (upstream issue
# #23352), so we warn at load time. This list is intentionally conservative — it
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

# camelCase aliases must round-trip both ways (read camelCase from YAML, or build
# from snake_case in code), so every model uses this config.
_MODEL_CONFIG = ConfigDict(populate_by_name=True, extra="forbid")


class Backend(BaseModel):
    """How one upstream backend is laid out on the gateway.

    ``path_template`` is the path (relative to the gateway ``base_url``) the
    gateway expects, with ``{slug}`` (the model) and optionally ``{action}`` (the
    operation, google only) placeholders. ``action_map`` renames an operation to
    the gateway's name (e.g. ``streamGenerateContent`` ->
    ``customStreamGenerateContent``); operations not listed pass through unchanged.
    """

    model_config = _MODEL_CONFIG

    path_template: str = Field(alias="pathTemplate")
    action_map: dict[str, str] = Field(default_factory=dict, alias="actionMap")


class ModelCapabilities(BaseModel):
    """Python-side capability hints, namespaced under ``capabilities:``.

    These are optional and ignored by the TypeScript side. They tell the adapters
    how to drive a model without hardcoding anything gateway-specific.
    """

    model_config = _MODEL_CONFIG

    structured_output: Literal["native", "tool", "prompted"] = "native"
    multi_step_tools: bool = True
    grounding: list[str] = Field(default_factory=list)


class ModelEntry(BaseModel):
    """One model a provider serves, tagged with the backend that handles it."""

    model_config = _MODEL_CONFIG

    id: str
    backend: Literal["anthropic", "openai", "google"]
    slug: str | None = None  # path segment; defaults to ``id`` when omitted
    api: Literal["chat", "responses"] | None = None  # openai backend only
    settings: dict[str, Any] = Field(default_factory=dict)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class Gateway(BaseModel):
    """Where the gateway lives, which env var holds its key, and its topology."""

    model_config = _MODEL_CONFIG

    base_url: str = Field(alias="baseURL")
    api_key_env: str = Field(alias="apiKeyEnvVarName")
    backends: dict[str, Backend]


class Provider(BaseModel):
    """A gateway provider and the models it serves.

    ``settings`` are provider-level call defaults; each model's own ``settings``
    are merged on top (model wins).
    """

    model_config = _MODEL_CONFIG

    id: str
    gateway: Gateway
    settings: dict[str, Any] = Field(default_factory=dict)
    models: list[ModelEntry]


class RoleRef(BaseModel):
    """A role points at exactly one ``(provider, model)`` pair."""

    model_config = _MODEL_CONFIG

    provider: str
    model: str


class CatalogConfig(BaseModel):
    """The whole catalog: providers plus the roles an app references."""

    model_config = _MODEL_CONFIG

    providers: list[Provider]
    roles: dict[str, RoleRef] = Field(default_factory=dict)


def _check_invariants(config: CatalogConfig) -> None:
    """Cross-field checks that need the whole object.

    Raises :class:`ConfigError` for hard inconsistencies and emits
    :class:`ProviderIdCollisionWarning` for the soft LiteLLM-collision case.
    """
    provider_ids: set[str] = set()
    # provider id -> {model id -> ModelEntry}
    index: dict[str, dict[str, ModelEntry]] = {}

    for provider in config.providers:
        if provider.id in provider_ids:
            raise ConfigError(f'Duplicate provider id "{provider.id}".')
        provider_ids.add(provider.id)

        if provider.id in _LITELLM_BUILTIN_PROVIDERS:
            warnings.warn(
                f'Provider id "{provider.id}" collides with a built-in LiteLLM '
                "provider name; LiteLLM may silently bypass the custom handler "
                "(issue #23352). Choose a distinct provider id.",
                ProviderIdCollisionWarning,
                stacklevel=3,
            )

        models: dict[str, ModelEntry] = {}
        for model in provider.models:
            if model.id in models:
                raise ConfigError(
                    f'Duplicate model id "{model.id}" in provider "{provider.id}".'
                )
            models[model.id] = model

            if model.backend not in provider.gateway.backends:
                raise ConfigError(
                    f'Model "{model.id}" uses backend "{model.backend}", but '
                    f'"{provider.id}.gateway.backends.{model.backend}" is not '
                    "configured."
                )
            if model.api is not None and model.backend != "openai":
                raise ConfigError(
                    f'Model "{model.id}" sets api="{model.api}", which is only '
                    f'allowed when backend == "openai" (got "{model.backend}").'
                )
        index[provider.id] = models

    for role, ref in config.roles.items():
        models = index.get(ref.provider, {})
        if ref.provider not in index:
            raise ConfigError(
                f'Role "{role}" references unknown provider "{ref.provider}".'
            )
        if ref.model not in models:
            raise ConfigError(
                f'Role "{role}" references unknown model "{ref.provider}:{ref.model}".'
            )


def parse_config(data: dict[str, Any]) -> CatalogConfig:
    """Validate an already-parsed mapping and return a typed :class:`CatalogConfig`.

    This is the portable core (no filesystem access): hand it a dict from any
    source — parsed YAML/JSON, a fixture, an API response. Raises
    :class:`ConfigError` on any schema or invariant violation.
    """
    try:
        config = CatalogConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    _check_invariants(config)
    return config


def load_config(path: str | Path) -> CatalogConfig:
    """Read ``path`` (.yaml/.yml/.json), validate it, and return the config.

    The extension selects the parser; YAML is a superset of JSON so ``.json``
    also parses as YAML, but we honour the extension for clarity. Raises
    :class:`ConfigError` (with the file path) on any problem.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{p}: failed to parse: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"{p}: expected a mapping at the top level, got {type(data).__name__}."
        )

    try:
        return parse_config(data)
    except ConfigError as exc:
        raise ConfigError(f"{p}:\n{exc}") from exc
