# llm-catalog-core

Gateway-agnostic core for [llm-catalog](https://github.com/sincekmori/llm-catalog).
It holds the config schema, loader/validation, role resolution, the path-rewriting `GatewayTransport`, and native LiteLLM codegen.

This package knows nothing about any runtime adapter, so installing it pulls in neither Pydantic AI nor LiteLLM.
Use it directly when you only need resolution or codegen.
Otherwise install one of the adapter distributions (`llm-catalog-pydantic-ai`, `llm-catalog-litellm`), which depend on this.

```python
from llm_catalog.core import Catalog

cat = Catalog.from_file("catalog.yaml")
rm = cat.resolve_role("fast")   # -> ResolvedModel (no adapter types)
print(rm.backend, rm.slug, rm.base_url)
```

See the [repository README](https://github.com/sincekmori/llm-catalog) for the full picture, the public/private boundary, and the verification notes.

import namespace: `llm_catalog.core`
