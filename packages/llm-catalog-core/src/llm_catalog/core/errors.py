# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Error and warning types raised by llm-catalog-core.

Everything user-facing inherits from :class:`LLMCatalogError` so callers can
catch the whole family with one ``except``.
"""


class LLMCatalogError(Exception):
    """Base class for every error raised by llm-catalog."""


class ConfigError(LLMCatalogError):
    """The catalog config is structurally invalid or internally inconsistent.

    Raised by :class:`llm_catalog.core.Catalog` for schema violations (wrapping
    Pydantic's ``ValidationError`` as a readable issue list) and for the
    cross-field checks (unknown role target, backend not configured, misplaced
    ``actionMap``, ...).
    """


class ResolutionError(LLMCatalogError):
    """A role or ``provider:model`` key could not be resolved against the config."""
