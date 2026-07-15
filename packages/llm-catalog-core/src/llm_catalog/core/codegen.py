# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Generate a native LiteLLM proxy config from a catalog config (route 2a).

This is the *lean alternative* to the LiteLLM plugin: it emits a plain LiteLLM
config that uses LiteLLM's **built-in** providers (``anthropic/...``,
``openai/...``, ``gemini/...``) plus ``api_base``, with a ``model_group_alias``
for the roles.

It imports neither ``litellm`` nor any adapter — it is pure dict/JSON text — so
it ships in *core* and runs with only ``llm-catalog-core`` installed. The CLI
writes JSON; JSON is a subset of YAML, so LiteLLM's YAML config loader reads the
generated file directly (``litellm --config litellm.config.json``).

Limitations (documented in the README): because it relies on LiteLLM's built-in
path construction, it does **not** honour a custom ``pathTemplate``, the
declarative ``headers``/``query`` extras, or native grounding, and it only
covers the vendors LiteLLM has built-ins for (``anthropic`` / ``openai`` /
``openai-compatible`` / ``google``). Use the ``llm-catalog-litellm`` plugin
when those matter.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .config import (
    GATEWAY_DEFAULT_API_KEY_ENV,
    ApiKey,
    CatalogConfig,
    EnvVarRef,
    ModelEntry,
    ModelSettings,
    Provider,
    parse_role_ref,
    vendor_block_of,
)
from .errors import ConfigError

__all__ = ["main", "to_litellm_config", "to_litellm_settings"]

# catalog vendor -> LiteLLM built-in provider prefix.
_VENDOR_TO_LITELLM: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "openai-compatible": "openai",
    "google": "gemini",
}

# ai-sdk-catalog camelCase setting -> the LiteLLM/OpenAI parameter name.
_SETTINGS_TO_LITELLM: dict[str, str] = {
    "maxOutputTokens": "max_tokens",
    "temperature": "temperature",
    "topP": "top_p",
    "topK": "top_k",
    "presencePenalty": "presence_penalty",
    "frequencyPenalty": "frequency_penalty",
    "stopSequences": "stop",
    "seed": "seed",
}


def to_litellm_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate merged catalog settings (camelCase) to LiteLLM parameters.

    ``providerOptions`` has no LiteLLM equivalent and is dropped; everything
    else maps 1:1 via ``_SETTINGS_TO_LITELLM``.
    """
    return {
        _SETTINGS_TO_LITELLM[key]: value
        for key, value in settings.items()
        if key in _SETTINGS_TO_LITELLM
    }


def _merged_settings(provider: Provider, model: ModelEntry) -> dict[str, Any]:
    def as_dict(settings: ModelSettings | None) -> dict[str, Any]:
        return {} if settings is None else settings.as_dict()

    # Merge provider then model defaults (model wins). The per-namespace
    # providerOptions merge is irrelevant here: providerOptions is dropped.
    return to_litellm_settings(
        {**as_dict(provider.settings), **as_dict(model.settings)}
    )


def _api_key_value(api_key: ApiKey | None, default_env: str | None) -> str | None:
    """Render an ``apiKey`` config value for a LiteLLM config file.

    LiteLLM reads ``os.environ/NAME`` as an env reference — a production key
    value is never written into the generated file. A literal ``apiKey``
    (local endpoints) is carried over as-is. ``None`` with no default env
    (direct providers) yields ``None``: LiteLLM's own vendor default applies.
    """
    if isinstance(api_key, str):
        return api_key
    if isinstance(api_key, EnvVarRef):
        return f"os.environ/{api_key.env_var_name}"
    if default_env is not None:
        return f"os.environ/{default_env}"
    return None


def _reject_transport_extras(provider: Provider) -> None:
    """Reject declarative headers/query — fail loudly rather than drop them."""
    extras: list[Any] = []
    if provider.gateway is not None:
        extras = [provider.gateway.headers, provider.gateway.query]
        for backend in provider.gateway.backends.values():
            extras += [backend.headers, backend.query]
    else:
        block = vendor_block_of(provider)
        if block is not None:
            extras = [block.headers, block.query]
    if any(extras):
        raise ConfigError(
            f'Provider "{provider.id}" configures "headers"/"query", which the '
            "LiteLLM codegen cannot express. Use the llm-catalog-litellm plugin "
            "instead."
        )


def _litellm_params(provider: Provider, model: ModelEntry) -> dict[str, Any]:
    if provider.gateway is not None:
        if model.backend is None:  # config validation guarantees it is set
            raise ConfigError(f'Model "{provider.id}:{model.id}" has no "backend".')
        backend = provider.gateway.backends[model.backend]
        prefix = _VENDOR_TO_LITELLM.get(backend.vendor)
        if prefix is None:
            raise ConfigError(
                f'Model "{provider.id}:{model.id}" uses vendor "{backend.vendor}", '
                "which has no LiteLLM built-in provider. Supported by the "
                f"codegen: {sorted(_VENDOR_TO_LITELLM)}."
            )
        params: dict[str, Any] = {
            "model": f"{prefix}/{model.slug or model.id}",
            "api_base": provider.gateway.base_url,
            "api_key": _api_key_value(
                provider.gateway.api_key, GATEWAY_DEFAULT_API_KEY_ENV
            ),
        }
    else:
        block = vendor_block_of(provider)
        vendor = (block.id if block is not None else None) or provider.id
        prefix = _VENDOR_TO_LITELLM.get(vendor)
        if prefix is None:
            raise ConfigError(
                f'Model "{provider.id}:{model.id}" uses vendor "{vendor}", '
                "which has no LiteLLM built-in provider. Supported by the "
                f"codegen: {sorted(_VENDOR_TO_LITELLM)}."
            )
        params = {"model": f"{prefix}/{model.id}"}
        if block is not None and block.base_url is not None:
            params["api_base"] = block.base_url
        api_key = _api_key_value(
            block.api_key if block is not None else None, default_env=None
        )
        if api_key is not None:
            params["api_key"] = api_key

    params.update(_merged_settings(provider, model))
    return params


def to_litellm_config(config: CatalogConfig) -> dict[str, Any]:
    """Build a LiteLLM proxy config structure (model_list + role aliases)."""
    for provider in config.providers:
        _reject_transport_extras(provider)

    model_list: list[dict[str, Any]] = [
        {
            "model_name": f"{provider.id}/{model.id}",
            "litellm_params": _litellm_params(provider, model),
        }
        for provider in config.providers
        for model in provider.models
    ]

    result: dict[str, Any] = {"model_list": model_list}

    if config.roles:
        aliases: dict[str, str] = {}
        for role, ref in config.roles.items():
            target = parse_role_ref(ref)
            aliases[role] = f"{target.provider}/{target.model}"
        result["router_settings"] = {"model_group_alias": aliases}
    return result


def main(argv: list[str] | None = None) -> int:
    """Generate a LiteLLM proxy config from a catalog config (CLI entry point).

    Usage: ``llm-catalog-codegen llm-catalog.json -o litellm.config.json``.
    """
    parser = argparse.ArgumentParser(
        prog="llm-catalog-codegen",
        description=(
            "Generate a native LiteLLM proxy config (JSON, readable by "
            "LiteLLM's YAML loader) from a catalog config JSON."
        ),
    )
    parser.add_argument("catalog", type=Path, help="path to the catalog config JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        data = json.loads(args.catalog.read_text(encoding="utf-8"))
        config = Catalog(data).config
        rendered = json.dumps(to_litellm_config(config), indent=2) + "\n"
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        sys.stderr.write(f"{parser.prog}: {args.catalog}: {exc}\n")
        return 1

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
