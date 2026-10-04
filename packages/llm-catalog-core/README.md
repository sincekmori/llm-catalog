# llm-catalog-core

Runtime-agnostic core for [llm-catalog](https://github.com/sincekmori/llm-catalog).
It holds the config schema (shipped as `schema.json` for editor validation; parity with `ai-sdk-catalog` 0.13), validation, role resolution, the path-rewriting `GatewayTransport`, native LiteLLM codegen, and an embedded models.dev price snapshot that fills a model's missing `cost` by vendor and model id (an explicit `cost` in the config always wins).

This package knows nothing about any runtime adapter and never touches the filesystem, so installing it pulls in neither Pydantic AI nor LiteLLM.
Use it directly when you only need resolution or codegen.
Otherwise install one of the adapter distributions (`llm-catalog-pydantic-ai`, `llm-catalog-ai-sdk`, `llm-catalog-litellm`), which depend on this.

The config format is JSON, shared verbatim with [`ai-sdk-catalog`](https://github.com/sincekmori/ai-sdk-utils/tree/main/packages/catalog) — read the file yourself and hand the parsed mapping to `Catalog`, which validates it:

```python
import json
from pathlib import Path

from llm_catalog.core import Catalog

config = json.loads(Path("ai-sdk-catalog.json").read_text(encoding="utf-8"))
cat = Catalog(config)
rm = cat.resolve_role("fast")  # -> ResolvedModel (no adapter types)
print(rm.kind, rm.vendor, rm.base_url)  # "gateway" or "direct"
```

To keep using YAML, parse it yourself (e.g. `yaml.safe_load`) and pass the result the same way.

`GatewayTransport` comes in two flavours with identical behaviour, because vendor SDKs are split between `httpx` and its continuation `httpx2`:

- `llm_catalog.core.transport` (also re-exported from `llm_catalog.core`) for `httpx` clients, e.g. LiteLLM with `openai<3`.
- `llm_catalog.core.transport2` for `httpx2` clients (`openai>=3`, `anthropic>=1`, Pydantic AI, the AI SDK for Python); install the optional dependency with `pip install "llm-catalog-core[httpx2]"`.

Build one for a resolved model with `GatewayTransport.for_model(inner, rm)`.

See the [repository README](https://github.com/sincekmori/llm-catalog) for the full picture, the public/private boundary, and the verification notes.

import namespace: `llm_catalog.core`
