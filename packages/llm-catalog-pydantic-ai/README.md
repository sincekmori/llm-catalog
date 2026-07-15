# llm-catalog-pydantic-ai

[Pydantic AI](https://ai.pydantic.dev) adapter for [llm-catalog](https://github.com/sincekmori/llm-catalog).
It turns a role in your catalog config (`llm-catalog.json`, shared verbatim with `ai-sdk-catalog`) into a native Pydantic AI `Model` — a gateway model is routed through your gateway via the core `GatewayTransport`, a direct model calls the vendor's own endpoint (anthropic / openai / openai-compatible / google).

```python
import json
from pathlib import Path

from pydantic_ai import Agent
from llm_catalog.pydantic_ai import PydanticAICatalog

config = json.loads(Path("llm-catalog.json").read_text(encoding="utf-8"))
cat = PydanticAICatalog(config)  # validates the config itself
agent = Agent(cat.model_for_role("fast"))
```

Installing this package does not pull in `litellm`.
The distributions are deliberately split so a Pydantic AI user's lockfile never references it.

See the [repository README](https://github.com/sincekmori/llm-catalog) for the full picture and the verification notes (§9), including the google-genai / custom-httpx-client caveat.

import namespace: `llm_catalog.pydantic_ai`
