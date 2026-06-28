# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The gateway-agnostic resolution entry point.

:class:`Catalog` is the centre of ``llm-catalog-core``. It does **resolution
only**: role/key -> :class:`ResolvedModel`. It returns no Pydantic AI / LiteLLM
objects (those live in the adapter distributions), so importing it pulls in
neither runtime.
"""

from pathlib import Path

from .config import CatalogConfig, ModelEntry, Provider, load_config
from .errors import ResolutionError
from .resolve import ResolvedModel

__all__ = ["Catalog"]


class Catalog:
    """Resolves roles and ``provider:model`` keys to :class:`ResolvedModel`."""

    def __init__(self, config: CatalogConfig) -> None:
        self._config = config
        self._providers: dict[str, Provider] = {p.id: p for p in config.providers}

    @classmethod
    def from_file(cls, path: str | Path) -> "Catalog":
        """Load and validate ``catalog.yaml`` (or .json) and build a catalog."""
        return cls(load_config(path))

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
        return self._resolve(ref.provider, ref.model)

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

        backend = provider.gateway.backends[model.backend]
        # provider defaults first, then model settings win on conflict.
        settings = {**provider.settings, **model.settings}

        return ResolvedModel(
            provider_id=provider.id,
            model_id=model.id,
            backend=model.backend,
            slug=model.slug or model.id,
            api=model.api,
            base_url=provider.gateway.base_url,
            api_key_env=provider.gateway.api_key_env,
            path_template=backend.path_template,
            action_map=dict(backend.action_map),
            settings=settings,
            capabilities=model.capabilities,
        )


def _find_model(provider: Provider, model_id: str) -> ModelEntry | None:
    return next((m for m in provider.models if m.id == model_id), None)
