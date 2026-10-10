# llm-catalog-ai-sdk

[AI SDK for Python](https://ai-python.dev/) adapter for [llm-catalog](https://github.com/sincekmori/llm-catalog).
It turns a role in your catalog config (`ai-sdk-catalog.json`, shared verbatim with `ai-sdk-catalog`) into a native `ai.Model` — a gateway model is routed through your gateway via the core `GatewayTransport`, a direct model calls the vendor's own endpoint.

```python
import json
from pathlib import Path

import ai
from llm_catalog.ai_sdk import AISDKCatalog

config = json.loads(Path("ai-sdk-catalog.json").read_text(encoding="utf-8"))

async with AISDKCatalog(config) as cat:  # closes its HTTP clients on exit
    model = cat.model_for_role("fast")
    params = cat.params_for_role("fast")
    async with ai.stream(model, [ai.user_message("hi")], params=params) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
```

`ai.Model` carries no default call settings, so the catalog's merged `settings` come back separately from `params_for_role()` / `params()` as an `ai.InferenceRequestParams` to pass (or refine) per call.
`providerOptions` is skipped: its entries are options of the TypeScript AI SDK providers and have no counterpart here.

## Vendors and call surfaces

`ai` speaks three wire protocols directly, which covers three of the shared schema's vendors.
Every other combination raises `LLMCatalogError` when the model is built; nothing is silently routed to a different API.

| Vendor | `api` omitted | `responses` | `chat` | `completion` |
|---|---|---|---|---|
| `anthropic` | Messages | error | Messages | error |
| `openai` | Responses | Responses | Chat Completions | error |
| `openai-compatible` | Chat Completions | error | Chat Completions | error |
| any other vendor | error | error | error | error |

`chat` on `anthropic` follows ai-sdk-catalog, where a single-surface vendor exposes `chat` as an alias of its one surface.
`google`, `mistral`, `groq`, and the other vendors of the shared schema have no native protocol in `ai`; declare an endpoint that speaks Chat Completions as an `openai-compatible` vendor (or backend) instead.
`stopSequences` has no stop parameter on the OpenAI Responses API, so it raises there rather than being dropped.

## Beta notice

`ai` is in public beta and may change its API in any 0.x minor release.
This adapter pins the `ai` minor line it is tested against (currently 0.8.x) and uses only `ai`'s documented surface: `ai.get_provider` with `base_url` / `api_key` / `client`, `ai.Model` with an explicit `protocol`, and `ai.InferenceRequestParams`.
Its tests send requests through `ai` to a mock transport, so an `ai` release that stops honouring the injected client fails CI instead of failing at runtime.

Requires Python 3.12+ (the floor of `ai`).
The HTTP client is [`httpx2`](https://github.com/pydantic/httpx2), which `ai` and the official OpenAI / Anthropic SDKs are built on; the `header_rewrite` / `body_rewrite` hooks receive `httpx2.Headers` / `httpx2.Request`.
Pass `transport_factory` to supply the underlying `httpx2` transport yourself (a proxy, mTLS, connection limits, or `httpx2.MockTransport` in tests).
Installing this package pulls in neither `pydantic-ai` nor `litellm`.

See the [repository README](https://github.com/sincekmori/llm-catalog) for the full picture and the verification notes (§9).

import namespace: `llm_catalog.ai_sdk`
