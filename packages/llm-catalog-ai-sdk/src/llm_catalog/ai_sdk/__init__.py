# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""AI SDK for Python adapter for llm-catalog.

Exposes :class:`AISDKCatalog`, which builds native ``ai.Model`` objects (the
Vercel AI SDK for Python, PyPI ``ai``) from a catalog config and routes them
through your gateway via the core ``GatewayTransport``. Importing this package
pulls in neither ``pydantic-ai`` nor ``litellm``.

import namespace: ``llm_catalog.ai_sdk``
"""

from .catalog import AISDKCatalog

__all__ = ["AISDKCatalog"]
