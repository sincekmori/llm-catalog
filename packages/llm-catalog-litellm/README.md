# llm-catalog-litellm

[LiteLLM](https://docs.litellm.ai) `CustomLLM` plugin for [llm-catalog](https://github.com/sincekmori/llm-catalog).
One implementation serves both in-process use and the central proxy.

For in-process use, call `register()` once to wire the handler into LiteLLM.

```python
from llm_catalog.litellm import register

register()   # reads llm-catalog.json (LLM_CATALOG_CONFIG or default path)

import litellm

resp = litellm.completion(
    model="examplegw/fast",
    messages=[{"role": "user", "content": "hi"}],
)
```

For the proxy, reference `llm_catalog.litellm.handler` from `config.yaml`'s `custom_provider_map` (no `register()` call needed).

The handler resolves each model from `llm-catalog.json` itself (the JSON config shared verbatim with `ai-sdk-catalog`), so the proxy config never needs gateway details in `litellm_params`, sidestepping LiteLLM issue #18216.

See the [repository README](https://github.com/sincekmori/llm-catalog) for the proxy operations guide and the verification notes (§9), including whether LiteLLM honours a custom httpx client (the route-1/route-2 decision in §6.2).

import namespace: `llm_catalog.litellm`
