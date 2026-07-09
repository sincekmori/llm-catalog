# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Declarative catalog config schema and validation.

This is the single source of truth for providers, the models they serve, and the
roles an application references. It mirrors the TypeScript ``ai-sdk-catalog``
config so the *same* JSON file can drive both ecosystems: the camelCase keys
(``baseURL``, ``apiKeyEnvVarName``, ``pathTemplate``, ``actionMap``,
``structuredOutput``, ...) are accepted via Pydantic aliases, while
``populate_by_name`` also lets Python code build models with snake_case names.

The documented config format is **JSON** (this package ships its JSON Schema as
``schema.json``; see :func:`config_json_schema`). Nothing here touches the
filesystem — read the file however you like and hand the parsed mapping to
:class:`~llm_catalog.core.Catalog`, which validates it. To keep using YAML,
parse it yourself (e.g. ``yaml.safe_load``) and pass the result the same way.

Unlike the TypeScript schema, this Python schema models **gateway providers
only** — every provider routes through a gateway and every model names a
``backend``. Direct/resolver providers are out of scope here.

Secrets never live in config: a gateway carries ``apiKeyEnvVarName`` (the *name*
of an environment variable) — or, for a local endpoint, an ``apiKey`` literal —
never a production key value.
"""

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ProviderIdCollisionWarning

__all__ = [
    "Backend",
    "CatalogConfig",
    "Gateway",
    "ModelCapabilities",
    "ModelEntry",
    "Provider",
    "RoleRef",
    "VendorName",
    "config_json_schema",
]

# The upstream backends a gateway can front, matching ai-sdk-catalog's `Vendor`
# enum one-to-one so a shared config validates identically on both sides. Which
# of these a given adapter can actually drive is the adapter's business — an
# unsupported backend fails at use time, not at validation time.
VendorName = Literal[
    "anthropic",
    "openai",
    "openai-compatible",
    "mistral",
    "cohere",
    "groq",
    "xai",
    "deepseek",
    "perplexity",
    "google",
]

# Built-in LiteLLM provider names. A custom provider id that collides with one of
# these is silently bypassed by LiteLLM's custom_provider_map (upstream issue
# #23352), so we warn at validation time. This list is intentionally
# conservative — it covers the common built-ins; it does not need to be
# exhaustive to be useful.
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

# camelCase aliases must round-trip both ways (read camelCase from JSON, or build
# from snake_case in code), so every model uses this config.
_MODEL_CONFIG = ConfigDict(populate_by_name=True, extra="forbid")


class Backend(BaseModel):
    """How one upstream backend is laid out on the gateway.

    ``path_template`` is the path (relative to the gateway ``base_url``) the
    gateway expects, with ``{slug}`` (the model) and — for the ``google``
    backend, whose model lives in the URL — ``{action}`` (the operation)
    placeholders. ``action_map`` (google only) renames an operation to the
    gateway's name (e.g. ``streamGenerateContent`` ->
    ``customStreamGenerateContent``); operations not listed pass through
    unchanged. ``name`` (openai-compatible only) sets the metadata namespace.
    """

    model_config = _MODEL_CONFIG

    path_template: str = Field(alias="pathTemplate")
    action_map: dict[str, str] = Field(default_factory=dict, alias="actionMap")
    name: str | None = None


class ModelCapabilities(BaseModel):
    """Python-side capability hints, namespaced under ``capabilities:``.

    These are optional and ignored by the TypeScript side (whose runtime strips
    unknown keys). They tell the adapters how to drive a model without
    hardcoding anything gateway-specific.
    """

    model_config = _MODEL_CONFIG

    structured_output: Literal["native", "tool", "prompted"] = Field(
        default="native", alias="structuredOutput"
    )
    multi_step_tools: bool = Field(default=True, alias="multiStepTools")
    grounding: list[str] = Field(default_factory=list)


class ModelEntry(BaseModel):
    """One model a provider serves, tagged with the backend that handles it.

    ``api`` picks the call surface for backends that have more than one
    (``responses`` | ``chat`` | ``completion``); omit it for the backend's
    default. It mirrors ai-sdk-catalog's ``ModelApi`` — adapters use it where it
    applies and ignore it elsewhere.
    """

    model_config = _MODEL_CONFIG

    id: str
    backend: VendorName
    slug: str | None = None  # path segment; defaults to ``id`` when omitted
    api: Literal["responses", "chat", "completion"] | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class Gateway(BaseModel):
    """Where the gateway lives, its key, and its topology.

    ``api_key`` is a literal key value (for a local endpoint); when omitted the
    key is read at call time from the environment variable named by
    ``api_key_env`` (default ``AI_GATEWAY_API_KEY``, matching ai-sdk-catalog).
    """

    model_config = _MODEL_CONFIG

    base_url: str = Field(alias="baseURL")
    api_key: str | None = Field(default=None, alias="apiKey")
    api_key_env: str = Field(default="AI_GATEWAY_API_KEY", alias="apiKeyEnvVarName")
    backends: dict[VendorName, Backend]


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
    """The whole catalog: providers plus the roles an app references.

    Structural validation lives in the field schemas; whole-config invariants
    (uniqueness, backend coherence, referential integrity) live in the
    ``model_validator`` below, where the full object is available. Every issue
    is collected before raising, so one round trip surfaces them all.
    """

    model_config = _MODEL_CONFIG

    # Editor affordance: configs may point at schema.json. Accepted and ignored.
    schema_: str | None = Field(default=None, alias="$schema")
    providers: list[Provider]
    roles: dict[str, RoleRef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_invariants(self) -> "CatalogConfig":
        issues: list[str] = []
        provider_ids: set[str] = set()
        # provider id -> set of model ids
        index: dict[str, set[str]] = {}

        for provider in self.providers:
            if provider.id in provider_ids:
                issues.append(f'Duplicate provider id "{provider.id}".')
            provider_ids.add(provider.id)

            if provider.id in _LITELLM_BUILTIN_PROVIDERS:
                warnings.warn(
                    f'Provider id "{provider.id}" collides with a built-in '
                    "LiteLLM provider name; LiteLLM may silently bypass the "
                    "custom handler (issue #23352). Choose a distinct provider "
                    "id.",
                    ProviderIdCollisionWarning,
                    stacklevel=2,
                )

            for backend_name, backend in provider.gateway.backends.items():
                path = f"{provider.id}.gateway.backends.{backend_name}"
                if "{slug}" not in backend.path_template:
                    issues.append(
                        f'"{path}.pathTemplate" must contain the "{{slug}}" '
                        "placeholder."
                    )
                if backend_name == "google":
                    if "{action}" not in backend.path_template:
                        issues.append(
                            f'"{path}.pathTemplate" must contain the '
                            '"{action}" placeholder.'
                        )
                elif backend.action_map:
                    issues.append(
                        f'"{path}" sets "actionMap", which only applies to the '
                        '"google" backend.'
                    )
                if backend.name is not None and backend_name != "openai-compatible":
                    issues.append(
                        f'"{path}" sets "name", which only applies to the '
                        '"openai-compatible" backend.'
                    )

            model_ids: set[str] = set()
            for model in provider.models:
                if model.id in model_ids:
                    issues.append(
                        f'Duplicate model id "{model.id}" in provider "{provider.id}".'
                    )
                model_ids.add(model.id)

                if model.backend not in provider.gateway.backends:
                    issues.append(
                        f'Model "{model.id}" uses backend "{model.backend}", '
                        f'but "{provider.id}.gateway.backends.{model.backend}" '
                        "is not configured."
                    )
            index[provider.id] = model_ids

        for role, ref in self.roles.items():
            if ref.provider not in index:
                issues.append(
                    f'Role "{role}" references unknown provider "{ref.provider}".'
                )
            elif ref.model not in index[ref.provider]:
                issues.append(
                    f'Role "{role}" references unknown model '
                    f'"{ref.provider}:{ref.model}".'
                )

        if issues:
            raise ValueError("\n".join(issues))
        return self


def config_json_schema() -> dict[str, Any]:
    """Return the config's JSON Schema in its camelCase (file) form.

    This is what ships as ``schema.json`` in this package — point a config's
    ``"$schema"`` at that file for editor validation and autocompletion.
    Regenerate the shipped copy with ``scripts/generate_schema.py`` after
    changing the models here; a test fails if it drifts.
    """
    schema = CatalogConfig.model_json_schema(by_alias=True)
    # Declare the dialect first, like any hand-written schema would.
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic error as a readable issue list (one line per issue)."""
    lines = [f"Invalid catalog config ({exc.error_count()} issue(s)):"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        msg = err["msg"]
        lines.append(f"  {loc}: {msg}" if loc else f"  {msg}")
    return "\n".join(lines)
