# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""llm-catalog-core: runtime-agnostic config, resolution, transport, and codegen.

This distribution knows nothing about any runtime adapter (Pydantic AI, LiteLLM)
and never touches the filesystem. Parse your catalog config JSON yourself and
hand the mapping to :class:`Catalog`, which validates it and resolves roles/keys
to :class:`ResolvedModel`. It also provides the path-rewriting
:class:`GatewayTransport` and generates a native LiteLLM config
(:func:`to_litellm_config`).

The config's JSON Schema ships as ``schema.json`` inside this package (see
:func:`config_json_schema`) — point a config's ``"$schema"`` at it for editor
validation and autocompletion.

``llm_catalog`` is a PEP 420 namespace shared with the adapter distributions;
this package ships only ``llm_catalog.core``.
"""

from .catalog import Catalog
from .codegen import to_litellm_config
from .config import (
    API_KEY_PLACEHOLDER,
    CatalogConfig,
    EnvVarRef,
    Gateway,
    GatewayBackend,
    ModelCapabilities,
    ModelEntry,
    ModelSettings,
    Provider,
    RoleRef,
    RoleTarget,
    VendorBlock,
    VendorName,
    config_json_schema,
    parse_role_ref,
    vendor_block_of,
)
from .errors import ConfigError, LLMCatalogError, ResolutionError
from .resolve import ResolvedModel
from .transport import (
    BodyRewrite,
    GatewayTransport,
    GatewayTransportSync,
    HeaderRewrite,
)

__all__ = [
    "API_KEY_PLACEHOLDER",
    "BodyRewrite",
    "Catalog",
    "CatalogConfig",
    "ConfigError",
    "EnvVarRef",
    "Gateway",
    "GatewayBackend",
    "GatewayTransport",
    "GatewayTransportSync",
    "HeaderRewrite",
    "LLMCatalogError",
    "ModelCapabilities",
    "ModelEntry",
    "ModelSettings",
    "Provider",
    "ResolutionError",
    "ResolvedModel",
    "RoleRef",
    "RoleTarget",
    "VendorBlock",
    "VendorName",
    "config_json_schema",
    "parse_role_ref",
    "to_litellm_config",
    "vendor_block_of",
]
