# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Declarative catalog config schema and validation.

This is the single source of truth for providers, the models they serve, and the
roles an application references. It mirrors the TypeScript ``ai-sdk-catalog``
(0.7) config so the *same* JSON file can drive both ecosystems: the camelCase
keys (``baseURL``, ``envVarName``, ``pathTemplate``, ``actionMap``, ...) are
accepted via Pydantic aliases, while ``populate_by_name`` also lets Python code
build models with snake_case names.

A provider resolves in one of two ways here (mirroring ai-sdk-catalog's three,
minus the code-level resolver):

* **direct** — no ``gateway`` block. Its vendor is ``vendor`` (string shorthand
  or a :class:`VendorBlock`), defaulting to the provider ``id``, and the
  adapters call the vendor's own endpoint (or the block's ``baseURL``).
* **gateway** — a ``gateway`` block describes your own LLM gateway's topology
  (``baseURL`` + free-form ``backends``); each model names its ``backend`` key.

The documented config format is **JSON** (this package ships its JSON Schema as
``schema.json``; see :func:`config_json_schema`). Nothing here touches the
filesystem — read the file however you like and hand the parsed mapping to
:class:`~llm_catalog.core.Catalog`, which validates it. To keep using YAML,
parse it yourself (e.g. ``yaml.safe_load``) and pass the result the same way.

Secrets never live in config: an ``apiKey`` is either a literal (local
endpoints only) or ``{"envVarName": "..."}`` — the *name* of an environment
variable, read lazily when the model is first used. Header values accept the
same union, and an inline header value may embed the resolved key via the
``{apiKey}`` placeholder.

Every object is strict (``extra="forbid"``): an unknown key fails validation
instead of being silently dropped, exactly like ai-sdk-catalog. The one
extension is the Python-only ``capabilities`` block on a model — ai-sdk-catalog
0.7+ rejects unknown keys, so keep ``capabilities`` out of a file shared with
the TypeScript side (defaults then apply here).
"""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

__all__ = [
    "API_KEY_PLACEHOLDER",
    "ApiKey",
    "CatalogConfig",
    "EnvVarRef",
    "Gateway",
    "GatewayBackend",
    "HeaderValue",
    "ModelCapabilities",
    "ModelEntry",
    "ModelSettings",
    "Provider",
    "QueryParams",
    "RequestHeaders",
    "RoleRef",
    "RoleTarget",
    "VendorBlock",
    "VendorName",
    "config_json_schema",
    "parse_role_ref",
    "vendor_block_of",
]

# The bundled vendors, matching ai-sdk-catalog's `Vendor` enum one-to-one so a
# shared config validates identically on both sides. Which of these a given
# adapter can actually drive is the adapter's business — an unsupported vendor
# fails at use time, not at validation time.
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

# Placeholder in a string header value, replaced with the resolved API key.
API_KEY_PLACEHOLDER = "{apiKey}"

# camelCase aliases must round-trip both ways (read camelCase from JSON, or build
# from snake_case in code), so every model uses this config.
_MODEL_CONFIG = ConfigDict(populate_by_name=True, extra="forbid")

_NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class EnvVarRef(BaseModel):
    """A reference to an environment variable, read lazily at first use."""

    model_config = _MODEL_CONFIG

    env_var_name: _NonEmptyStr = Field(alias="envVarName")


# An API key: a literal string, or {"envVarName": "..."} to read it from that
# environment variable at first use. Prefer the env-var form for real keys.
ApiKey = _NonEmptyStr | EnvVarRef

# One header value: a string sent as-is (the "{apiKey}" placeholder inside it is
# replaced with the provider's resolved API key), or an EnvVarRef read at first
# use and sent verbatim.
HeaderValue = _NonEmptyStr | EnvVarRef

# Extra request headers, by header name. Merged into every request on top of the
# vendor SDK's own headers — a header named here overrides the SDK's.
RequestHeaders = dict[_NonEmptyStr, HeaderValue]

# Query parameters appended to every request URL (for a gateway, after the path
# rewriting). Values are plain text — don't put secrets in URLs.
QueryParams = dict[_NonEmptyStr, str]


def headers_need_api_key(headers: RequestHeaders) -> bool:
    """Return True when any inline header value references ``{apiKey}``."""
    return any(
        isinstance(value, str) and API_KEY_PLACEHOLDER in value
        for value in headers.values()
    )


class ModelSettings(BaseModel):
    """Default AI SDK call settings, mirroring ai-sdk-catalog's ``ModelSettings``.

    They are defaults the adapters bake into the model they build, so anything
    set here can also be overridden per call. Every field is optional; unknown
    keys are rejected (strict, like the TypeScript side).
    """

    model_config = _MODEL_CONFIG

    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens", gt=0)
    temperature: float | None = None
    top_p: float | None = Field(default=None, alias="topP")
    top_k: int | None = Field(default=None, alias="topK")
    presence_penalty: float | None = Field(default=None, alias="presencePenalty")
    frequency_penalty: float | None = Field(default=None, alias="frequencyPenalty")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")
    seed: int | None = None
    # Provider-specific options, passed through untouched
    # (e.g. { "openai": { "reasoningEffort": "low" } }). Values must be JSON.
    provider_options: dict[str, dict[str, JsonValue]] | None = Field(
        default=None, alias="providerOptions"
    )

    def as_dict(self) -> dict[str, Any]:
        """Return the settings in their camelCase (file) form, unset fields dropped."""
        return self.model_dump(by_alias=True, exclude_none=True)


class ModelCapabilities(BaseModel):
    """Python-side capability hints, namespaced under ``capabilities:``.

    This is the one llm-catalog extension over the ai-sdk-catalog schema. It
    tells the adapters how to drive a model (structured-output mode, grounding
    tools) without hardcoding anything gateway-specific.

    ai-sdk-catalog 0.7+ validates strictly and **rejects** unknown keys, so a
    config that sets ``capabilities`` no longer validates on the TypeScript
    side. Keep it out of a shared file (the defaults below then apply) and use
    it only in Python-only configs.
    """

    model_config = _MODEL_CONFIG

    structured_output: Literal["native", "tool", "prompted"] = Field(
        default="native", alias="structuredOutput"
    )
    multi_step_tools: bool = Field(default=True, alias="multiStepTools")
    grounding: list[str] = Field(default_factory=list)


class ModelEntry(BaseModel):
    """One model a provider serves.

    ``api`` picks the call surface for vendors that have more than one
    (``responses`` | ``chat`` | ``completion``); omit it for the vendor's
    default (Responses for OpenAI, Chat Completions for an OpenAI-compatible
    server). ``backend``/``slug`` apply to gateway providers only (the
    ``gateway.backends`` key that serves the model, and the path segment when
    it differs from ``id``). The schema keeps every field optional;
    :class:`CatalogConfig`'s validator enforces that the right ones are present
    for the provider's kind.
    """

    model_config = _MODEL_CONFIG

    id: _NonEmptyStr
    api: Literal["responses", "chat", "completion"] | None = None
    backend: _NonEmptyStr | None = None  # gateway providers only (backends key)
    slug: _NonEmptyStr | None = None  # gateway providers only (path override)
    settings: ModelSettings | None = None
    # Python-only extension; see ModelCapabilities.
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class VendorBlock(BaseModel):
    """A direct provider's vendor: which bundled vendor backs it, plus overrides.

    Everything is optional — ``id`` defaults to the provider's own id, and with
    no overrides the vendor SDK's defaults apply (its endpoint, its key env
    var). The string shorthand ``"vendor": "x"`` means ``{"id": "x"}``.
    """

    model_config = _MODEL_CONFIG

    id: VendorName | None = None  # defaults to the provider id
    base_url: _NonEmptyStr | None = Field(default=None, alias="baseURL")
    api_key: ApiKey | None = Field(default=None, alias="apiKey")
    name: _NonEmptyStr | None = None  # openai-compatible metadata namespace
    # Extra headers sent with every request (merged over the vendor SDK's own,
    # same-name wins). An inline value may embed the key via "{apiKey}".
    headers: RequestHeaders | None = None
    # Query params appended to every request URL, e.g. { "api-version": "..." }.
    query: QueryParams | None = None


class GatewayBackend(BaseModel):
    """One upstream backend on the gateway.

    Backends live in a map under a key of your choice, so the same vendor can
    appear more than once (e.g. two regions); each model picks its backend by
    that key via ``backend``.

    ``path_template`` is the path (relative to the gateway ``base_url``) the
    gateway expects, with ``{slug}`` (the model) and — for a ``google``
    backend, whose model lives in the URL — ``{action}`` (the operation)
    placeholders. ``action_map`` (google only) renames an operation to the
    gateway's name (e.g. ``streamGenerateContent`` ->
    ``customStreamGenerateContent``); operations not listed pass through
    unchanged. ``name`` (openai-compatible only) sets the metadata namespace.
    ``headers``/``query`` apply to this backend only, merged over the
    gateway-level ones (backend wins per name).
    """

    model_config = _MODEL_CONFIG

    vendor: VendorName
    path_template: _NonEmptyStr = Field(alias="pathTemplate")
    action_map: dict[str, str] | None = Field(default=None, alias="actionMap")
    name: _NonEmptyStr | None = None
    headers: RequestHeaders | None = None
    query: QueryParams | None = None


class Gateway(BaseModel):
    """Where the gateway lives, its key, its transport extras, and its topology.

    ``api_key`` is a literal string or ``{"envVarName": "..."}``; when omitted,
    the ``AI_GATEWAY_API_KEY`` environment variable is read instead (matching
    ai-sdk-catalog). ``headers``/``query`` apply to every request to the
    gateway (all backends); an inline header value may embed the gateway key
    via ``{apiKey}``.
    """

    model_config = _MODEL_CONFIG

    base_url: _NonEmptyStr = Field(alias="baseURL")
    api_key: ApiKey | None = Field(default=None, alias="apiKey")
    headers: RequestHeaders | None = None
    query: QueryParams | None = None
    backends: dict[_NonEmptyStr, GatewayBackend]


# Environment variable read when a gateway configures no apiKey.
GATEWAY_DEFAULT_API_KEY_ENV = "AI_GATEWAY_API_KEY"


class Provider(BaseModel):
    """A provider and the models it serves.

    Exactly one kind:

    * **direct** — no ``gateway`` block. Its vendor is ``vendor`` (string
      shorthand or a :class:`VendorBlock`), defaulting to ``id``.
    * **gateway** — a ``gateway`` block routes it through your own gateway
      (its models then require a ``backend``). ``vendor`` must not be set.

    ``settings`` are provider-level call defaults; each model's own
    ``settings`` are merged on top (model wins).
    """

    model_config = _MODEL_CONFIG

    id: _NonEmptyStr  # becomes the key prefix => "openai:gpt-5.6"
    vendor: VendorName | VendorBlock | None = None  # direct providers only
    gateway: Gateway | None = None  # gateway providers only
    settings: ModelSettings | None = None
    models: list[ModelEntry] = Field(min_length=1)


class RoleTarget(BaseModel):
    """A role's target, spelled out as an object."""

    model_config = _MODEL_CONFIG

    provider: _NonEmptyStr
    model: _NonEmptyStr


# A role points at exactly one provider+model pair: either the shorthand string
# "provider:model" (split at the first ":", so model ids may contain colons),
# or a RoleTarget object. Both forms are equivalent.
RoleRef = Annotated[str, StringConstraints(pattern=r"^[^:]+:.")] | RoleTarget


def parse_role_ref(ref: RoleRef) -> RoleTarget:
    """Normalize a role reference to its ``RoleTarget`` form.

    The string shorthand splits at the **first** ``:``, so model ids may
    contain colons (``"ollama:qwen3.6:35b"`` -> provider ``ollama``, model
    ``qwen3.6:35b``).
    """
    if isinstance(ref, str):
        provider, _, model = ref.partition(":")
        return RoleTarget(provider=provider, model=model)
    return ref


def vendor_block_of(provider: Provider) -> VendorBlock | None:
    """Return a direct provider's vendor block, normalizing the string shorthand.

    ``None`` when the provider sets no ``vendor`` at all (its vendor then
    defaults to the provider id, with no overrides).
    """
    if isinstance(provider.vendor, str):
        return VendorBlock(id=provider.vendor)
    return provider.vendor


class CatalogConfig(BaseModel):
    """The whole catalog: providers plus the roles an app references.

    Structural validation lives in the field schemas; whole-config invariants
    (uniqueness, provider-kind coherence, gateway/backend coherence,
    referential integrity) live in the ``model_validator`` below, where the
    full object is available. Every issue is collected before raising, so one
    round trip surfaces them all.
    """

    model_config = _MODEL_CONFIG

    # Editor affordance: configs may point at schema.json. Accepted and ignored.
    schema_: str | None = Field(default=None, alias="$schema")
    providers: list[Provider] = Field(min_length=1)
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

            if ":" in provider.id:
                # ":" would make the "provider:model" role shorthand ambiguous.
                issues.append(f'Provider id "{provider.id}" must not contain ":".')

            issues.extend(_provider_kind_issues(provider))
            if provider.gateway is not None:
                issues.extend(_gateway_issues(provider.id, provider.gateway))

            model_ids: set[str] = set()
            for model in provider.models:
                if model.id in model_ids:
                    issues.append(
                        f'Duplicate model id "{model.id}" in provider "{provider.id}".'
                    )
                model_ids.add(model.id)
                issues.extend(_model_kind_issues(provider, model))
            index[provider.id] = model_ids

        for role, ref in self.roles.items():
            target = parse_role_ref(ref)
            if target.provider not in index:
                issues.append(
                    f'Role "{role}" references unknown provider "{target.provider}".'
                )
            elif target.model not in index[target.provider]:
                issues.append(
                    f'Role "{role}" references unknown model '
                    f'"{target.provider}:{target.model}".'
                )

        if issues:
            raise ValueError("\n".join(issues))
        return self


def _provider_kind_issues(provider: Provider) -> list[str]:
    """Per-kind coherence of a provider's own fields (direct vs gateway)."""
    if provider.gateway is not None:
        if provider.vendor is not None:
            return [
                f'Provider "{provider.id}" sets both "vendor" and "gateway"; '
                "a provider is either direct or gateway-routed."
            ]
        return []

    issues: list[str] = []
    block = vendor_block_of(provider)
    vendor_id = (block.id if block is not None else None) or provider.id
    if vendor_id == "openai-compatible" and (block is None or block.base_url is None):
        # The OpenAI-compatible vendor has no canonical endpoint.
        issues.append(
            f'Provider "{provider.id}" uses the "openai-compatible" vendor and '
            'must set a "baseURL" in its "vendor" block.'
        )
    if (
        block is not None
        and block.headers is not None
        and headers_need_api_key(block.headers)
        and block.api_key is None
    ):
        # A vendor's own default key (e.g. OPENAI_API_KEY) is read inside the
        # SDK and never surfaces here, so there is nothing to substitute.
        issues.append(
            f'Provider "{provider.id}" uses "{API_KEY_PLACEHOLDER}" in '
            '"vendor.headers" but its "vendor" block sets no "apiKey".'
        )
    return issues


def _gateway_issues(provider_id: str, gateway: Gateway) -> list[str]:
    """Coherence of each gateway backend's fields."""
    issues: list[str] = []
    for backend_key, backend in gateway.backends.items():
        path = f"{provider_id}.gateway.backends.{backend_key}"
        if "{slug}" not in backend.path_template:
            issues.append(
                f'"{path}.pathTemplate" must contain the "{{slug}}" placeholder.'
            )
        if backend.vendor == "google":
            if "{action}" not in backend.path_template:
                issues.append(
                    f'"{path}.pathTemplate" must contain the "{{action}}" placeholder.'
                )
        elif backend.action_map is not None:
            issues.append(
                f'"{path}" sets "actionMap", which only applies to a "google" backend.'
            )
        if backend.name is not None and backend.vendor != "openai-compatible":
            issues.append(
                f'"{path}" sets "name", which only applies to an '
                '"openai-compatible" backend.'
            )
    return issues


def _model_kind_issues(provider: Provider, model: ModelEntry) -> list[str]:
    """Per-kind coherence of one model's fields."""
    if provider.gateway is not None:
        if model.backend is None:
            return [
                f'Model "{model.id}" in gateway provider "{provider.id}" must '
                'set a "backend".'
            ]
        if model.backend not in provider.gateway.backends:
            return [
                f'Model "{model.id}" uses backend "{model.backend}", but '
                f'"{provider.id}.gateway.backends.{model.backend}" is not '
                "configured."
            ]
        return []

    issues: list[str] = []
    if model.backend is not None:
        issues.append(
            f'Model "{model.id}" sets "backend", but provider "{provider.id}" '
            'has no "gateway" block.'
        )
    if model.slug is not None:
        issues.append(
            f'Model "{model.id}" sets "slug", but provider "{provider.id}" '
            'has no "gateway" block.'
        )
    return issues


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
