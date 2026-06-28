# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The resolved view of a single model, produced by :class:`Catalog`.

A :class:`ResolvedModel` is a flat, adapter-agnostic snapshot of everything
needed to call one model through its gateway: the addressing (backend, slug,
api), the gateway endpoint (``base_url`` + path template), the *name* of the env
var holding the key, the merged call settings, and the capability hints.

It deliberately holds no Pydantic AI / LiteLLM types and no secret values — the
key is read from the environment at call time by whichever adapter uses it.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from .config import ModelCapabilities

__all__ = ["ResolvedModel"]


@dataclass(frozen=True)
class ResolvedModel:
    """Flat, gateway-aware description of one model. Adapter-agnostic."""

    provider_id: str
    model_id: str
    backend: str
    slug: str
    api: str | None
    base_url: str
    api_key_env: str
    path_template: str
    action_map: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def api_key(self) -> str:
        """Read the key value from the environment, lazily and per call.

        Raises :class:`KeyError` (via ``os.environ``) when the variable is unset,
        which surfaces a clear, actionable message naming the missing var.
        """
        try:
            return os.environ[self.api_key_env]
        except KeyError as exc:
            raise KeyError(
                f'environment variable "{self.api_key_env}" is not set '
                f'(needed for provider "{self.provider_id}")'
            ) from exc
