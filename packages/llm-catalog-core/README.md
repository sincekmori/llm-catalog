# llm-catalog-core

Gateway-agnostic core for [llm-catalog](https://github.com/sincekmori/llm-catalog).
It holds the config schema (shipped as `schema.json` for editor validation), validation, role resolution, the path-rewriting `GatewayTransport`, and native LiteLLM codegen.

This package knows nothing about any runtime adapter and never touches the filesystem, so installing it pulls in neither Pydantic AI nor LiteLLM.
Use it directly when you only need resolution or codegen.
Otherwise install one of the adapter distributions (`llm-catalog-pydantic-ai`, `llm-catalog-litellm`), which depend on this.

The config format is JSON, shared verbatim with [`ai-sdk-catalog`](https://github.com/sincekmori/ai-sdk-utils/tree/main/packages/catalog) — read the file yourself and hand the parsed mapping to `Catalog`, which validates it:

```python
import json
from pathlib import Path

from llm_catalog.core import Catalog

config = json.loads(Path("llm-catalog.json").read_text(encoding="utf-8"))
cat = Catalog(config)
rm = cat.resolve_role("fast")   # -> ResolvedModel (no adapter types)
print(rm.backend, rm.slug, rm.base_url)
```

To keep using YAML, parse it yourself (e.g. `yaml.safe_load`) and pass the result the same way.

See the [repository README](https://github.com/sincekmori/llm-catalog) for the full picture, the public/private boundary, and the verification notes.

import namespace: `llm_catalog.core`
