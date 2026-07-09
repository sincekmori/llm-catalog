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
path construction, it does **not** honour a custom ``pathTemplate`` or native
grounding, and it only covers the backends LiteLLM has built-ins for
(``anthropic`` / ``openai`` / ``google``). Use the ``llm-catalog-litellm``
plugin when those matter.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .config import CatalogConfig, ModelEntry, Provider
from .errors import ConfigError

__all__ = ["main", "to_litellm_config"]

# catalog backend -> LiteLLM built-in provider prefix.
_BACKEND_TO_LITELLM: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
}


def _litellm_params(provider: Provider, model: ModelEntry) -> dict[str, Any]:
    prefix = _BACKEND_TO_LITELLM.get(model.backend)
    if prefix is None:
        raise ConfigError(
            f'Model "{provider.id}:{model.id}" uses backend "{model.backend}", '
            "which has no LiteLLM built-in provider. Supported by the codegen: "
            f"{sorted(_BACKEND_TO_LITELLM)}."
        )
    slug = model.slug or model.id

    params: dict[str, Any] = {
        "model": f"{prefix}/{slug}",
        "api_base": provider.gateway.base_url,
        # LiteLLM reads "os.environ/NAME" as an env reference — a production key
        # value is never written into the generated file. A literal `apiKey`
        # (local endpoints) is carried over as-is.
        "api_key": provider.gateway.api_key
        or f"os.environ/{provider.gateway.api_key_env}",
    }
    # Merge provider then model defaults (model wins).
    merged = {**provider.settings, **model.settings}
    params.update(merged)
    return params


def to_litellm_config(config: CatalogConfig) -> dict[str, Any]:
    """Build a LiteLLM proxy config structure (model_list + role aliases)."""
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
        result["router_settings"] = {
            "model_group_alias": {
                role: f"{ref.provider}/{ref.model}"
                for role, ref in config.roles.items()
            }
        }
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
