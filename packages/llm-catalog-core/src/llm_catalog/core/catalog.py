# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The runtime-agnostic resolution entry point.

:class:`Catalog` is the centre of ``llm-catalog-core``. It validates its own
input — hand it the mapping you parsed from JSON (or a ready
:class:`CatalogConfig`) — and does **resolution only**: role/key ->
:class:`ResolvedModel`. It returns no Pydantic AI / LiteLLM objects (those live
in the adapter distributions), so importing it pulls in neither runtime, and it
never touches the filesystem:

    import json
    from pathlib import Path
    from llm_catalog.core import Catalog

    config = json.loads(Path("llm-catalog.json").read_text(encoding="utf-8"))
    catalog = Catalog(config)
"""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .config import (
    GATEWAY_DEFAULT_API_KEY_ENV,
    CatalogConfig,
    ModelEntry,
    ModelSettings,
    Provider,
    format_validation_error,
    parse_role_ref,
    vendor_block_of,
)
from .errors import ConfigError, ResolutionError
from .resolve import ResolvedModel

__all__ = ["Catalog"]


def _settings_dict(settings: ModelSettings | None) -> dict[str, Any]:
    return {} if settings is None else settings.as_dict()


def _merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge call settings, override winning for scalar fields.

    ``providerOptions`` is merged per provider namespace so a model can add or
    override individual options without dropping the provider-level ones
    (mirroring ai-sdk-catalog's merge).
    """
    merged = {**base, **override}
    base_po = base.get("providerOptions")
    override_po = override.get("providerOptions")
    if isinstance(base_po, dict) and isinstance(override_po, dict):
        po: dict[str, Any] = dict(base_po)
        for ns, opts in override_po.items():
            existing = po.get(ns)
            if isinstance(existing, dict) and isinstance(opts, dict):
                po[ns] = {**existing, **opts}
            else:
                po[ns] = opts
        merged["providerOptions"] = po
    return merged


class Catalog:
    """Resolves roles and ``provider:model`` keys to :class:`ResolvedModel`.

    The constructor validates its input itself: pass the plain mapping you
    parsed from JSON (or built in code) and invalid input raises
    :class:`ConfigError` with a readable issue list. A ready
    :class:`CatalogConfig` passes through as-is.
    """

    def __init__(self, config: CatalogConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, CatalogConfig):
            try:
                config = CatalogConfig.model_validate(dict(config))
            except ValidationError as exc:
                raise ConfigError(format_validation_error(exc)) from exc
        self._config = config
        self._providers: dict[str, Provider] = {p.id: p for p in config.providers}

    @property
    def config(self) -> CatalogConfig:
        """The validated underlying config."""
        return self._config

    @property
    def roles(self) -> list[str]:
        """The names of every defined role."""
        return list(self._config.roles)

    def resolve_role(self, role: str) -> ResolvedModel:
        """Resolve a role name to a :class:`ResolvedModel`."""
        ref = self._config.roles.get(role)
        if ref is None:
            raise ResolutionError(
                f'Unknown role "{role}". Defined roles: {sorted(self._config.roles)}.'
            )
        target = parse_role_ref(ref)
        return self._resolve(target.provider, target.model)

    def resolve_key(self, key: str) -> ResolvedModel:
        """Resolve a ``"provider:model_id"`` key to a :class:`ResolvedModel`."""
        provider_id, sep, model_id = key.partition(":")
        if not sep:
            raise ResolutionError(
                f'Invalid model key "{key}"; expected "provider:model_id".'
            )
        return self._resolve(provider_id, model_id)

    def meta_for_role(self, role: str) -> ResolvedModel:
        """Alias of :meth:`resolve_role`, named for capability/metadata lookups."""
        return self.resolve_role(role)

    def _resolve(self, provider_id: str, model_id: str) -> ResolvedModel:
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ResolutionError(f'Unknown provider "{provider_id}".')

        model = _find_model(provider, model_id)
        if model is None:
            raise ResolutionError(
                f'Unknown model "{model_id}" in provider "{provider_id}".'
            )

        # provider defaults first, then model settings win on conflict.
        settings = _merge_settings(
            _settings_dict(provider.settings), _settings_dict(model.settings)
        )

        if provider.gateway is not None:
            gateway = provider.gateway
            if model.backend is None:  # config validation guarantees it is set
                raise ResolutionError(
                    f'Model "{model_id}" in gateway provider "{provider_id}" '
                    'has no "backend".'
                )
            backend = gateway.backends[model.backend]
            return ResolvedModel(
                kind="gateway",
                provider_id=provider.id,
                model_id=model.id,
                vendor=backend.vendor,
                api=model.api,
                base_url=gateway.base_url,
                api_key_config=gateway.api_key,
                default_api_key_env=GATEWAY_DEFAULT_API_KEY_ENV,
                name=backend.name,
                backend=model.backend,
                slug=model.slug or model.id,
                path_template=backend.path_template,
                action_map=dict(backend.action_map or {}),
                # gateway-level extras apply to every backend; the backend's own
                # entries are merged on top (backend wins per name).
                headers={**(gateway.headers or {}), **(backend.headers or {})},
                query={**(gateway.query or {}), **(backend.query or {})},
                settings=settings,
                capabilities=model.capabilities,
            )

        block = vendor_block_of(provider)
        return ResolvedModel(
            kind="direct",
            provider_id=provider.id,
            model_id=model.id,
            vendor=(block.id if block is not None else None) or provider.id,
            api=model.api,
            base_url=block.base_url if block is not None else None,
            api_key_config=block.api_key if block is not None else None,
            default_api_key_env=None,  # the vendor SDK's own default applies
            name=block.name if block is not None else None,
            headers=dict(block.headers or {}) if block is not None else {},
            query=dict(block.query or {}) if block is not None else {},
            settings=settings,
            capabilities=model.capabilities,
        )


def _find_model(provider: Provider, model_id: str) -> ModelEntry | None:
    return next((m for m in provider.models if m.id == model_id), None)
