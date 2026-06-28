# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Pydantic AI adapter for llm-catalog.

Exposes :class:`PydanticAICatalog`, which builds native Pydantic AI ``Model``
objects from a ``catalog.yaml`` and routes them through your gateway via the core
``GatewayTransport``. Importing this package does not pull in ``litellm``.

import namespace: ``llm_catalog.pydantic_ai``
"""

from .catalog import PydanticAICatalog

__all__ = ["PydanticAICatalog"]
