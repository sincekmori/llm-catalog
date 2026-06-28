# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""llm-catalog-core: gateway-agnostic config, resolution, transport, and codegen.

This distribution knows nothing about any runtime adapter (Pydantic AI, LiteLLM).
It loads and validates ``catalog.yaml``, resolves roles/keys to
:class:`ResolvedModel`, provides the path-rewriting :class:`GatewayTransport`,
and generates a native LiteLLM config (:func:`to_litellm_config`).

``llm_catalog`` is a PEP 420 namespace shared with the adapter distributions;
this package ships only ``llm_catalog.core``.
"""

from .catalog import Catalog
from .codegen import to_litellm_config
from .config import (
    Backend,
    CatalogConfig,
    Gateway,
    ModelCapabilities,
    ModelEntry,
    Provider,
    RoleRef,
    load_config,
    parse_config,
)
from .errors import (
    ConfigError,
    LLMCatalogError,
    ProviderIdCollisionWarning,
    ResolutionError,
)
from .resolve import ResolvedModel
from .transport import GatewayTransport, GatewayTransportSync, HeaderRewrite

__all__ = [
    "Backend",
    "Catalog",
    "CatalogConfig",
    "ConfigError",
    "Gateway",
    "GatewayTransport",
    "GatewayTransportSync",
    "HeaderRewrite",
    "LLMCatalogError",
    "ModelCapabilities",
    "ModelEntry",
    "Provider",
    "ProviderIdCollisionWarning",
    "ResolutionError",
    "ResolvedModel",
    "RoleRef",
    "load_config",
    "parse_config",
    "to_litellm_config",
]
