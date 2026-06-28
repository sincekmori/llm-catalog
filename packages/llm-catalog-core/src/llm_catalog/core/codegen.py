# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Generate a native LiteLLM ``config.yaml`` from a ``catalog.yaml`` (route 2a).

This is the *lean alternative* to the LiteLLM plugin: it emits a plain LiteLLM
config that uses LiteLLM's **built-in** providers (``anthropic/...``,
``openai/...``, ``gemini/...``) plus ``api_base``, with a ``model_group_alias``
for the roles.

It imports neither ``litellm`` nor any adapter — it is pure dict/YAML text — so
it ships in *core* and runs with only ``llm-catalog-core`` installed.

Limitations (documented in the README): because it relies on LiteLLM's built-in
path construction, it does **not** honour a custom ``pathTemplate`` or native
grounding. Use the ``llm-catalog-litellm`` plugin when those matter.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import CatalogConfig, ModelEntry, Provider, load_config

__all__ = ["main", "to_litellm_config"]

# catalog backend -> LiteLLM built-in provider prefix.
_BACKEND_TO_LITELLM: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
}


def _litellm_params(provider: Provider, model: ModelEntry) -> dict[str, Any]:
    prefix = _BACKEND_TO_LITELLM[model.backend]
    slug = model.slug or model.id

    params: dict[str, Any] = {
        "model": f"{prefix}/{slug}",
        "api_base": provider.gateway.base_url,
        # LiteLLM reads "os.environ/NAME" as an env reference — the key value is
        # never written into the generated file.
        "api_key": f"os.environ/{provider.gateway.api_key_env}",
    }
    # Merge provider then model defaults (model wins).
    merged = {**provider.settings, **model.settings}
    params.update(merged)
    return params


def to_litellm_config(config: CatalogConfig) -> dict[str, Any]:
    """Build a LiteLLM ``config.yaml`` structure (model_list + role aliases)."""
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
    """Generate a LiteLLM config.yaml from a catalog (CLI entry point).

    Usage: ``python -m llm_catalog.core.codegen catalog.yaml -o out.yaml``.
    """
    parser = argparse.ArgumentParser(
        prog="llm-catalog-codegen",
        description="Generate a native LiteLLM config.yaml from a catalog.yaml.",
    )
    parser.add_argument("catalog", type=Path, help="path to catalog.yaml / .json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: stdout)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.catalog)
    rendered = yaml.safe_dump(
        to_litellm_config(config), sort_keys=False, allow_unicode=True
    )

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
