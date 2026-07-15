# llm-catalog

[![CI](https://github.com/sincekmori/llm-catalog/actions/workflows/ci.yml/badge.svg)](https://github.com/sincekmori/llm-catalog/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/sincekmori/llm-catalog)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/llm-catalog-core)](https://pypi.org/project/llm-catalog-core/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/types-ty-261230)](https://github.com/astral-sh/ty)

Drive multiple LLM runtimes from one declarative JSON config — implement nothing, configure once.
It is the Python counterpart of [`ai-sdk-catalog`](https://github.com/sincekmori/ai-sdk-utils) (Vercel AI SDK), reading the **same config file**: one `llm-catalog.json` drives TypeScript and Python alike (schema parity with ai-sdk-catalog 0.7).
You reference a **role** name (`fast`, `reasoning`, `search`, …) and it resolves to a concrete model — either a **direct** vendor endpoint (OpenAI, Anthropic, Google, any OpenAI-compatible server) or a model behind your own **gateway**.

It targets two runtimes from the same config and the same core.

- **Pydantic AI** — in-process, native fidelity.
- **LiteLLM** — OpenAI-compatible, both in-process (SDK) and as a central proxy.

Gateway quirks (path layout, action names, auth headers, query params, structured-output mode, grounding tools) are honoured as config values, never hardcoded.
The same generic code therefore works for any gateway that exposes native provider formats under a path prefix.

## Packages

Three independently versioned distributions form a uv workspace and share the PEP 420 namespace `llm_catalog`.

| Distribution | PyPI | Import | Depends on |
|---|---|---|---|
| [`llm-catalog-core`](packages/llm-catalog-core) | [![PyPI](https://img.shields.io/pypi/v/llm-catalog-core)](https://pypi.org/project/llm-catalog-core/) | `llm_catalog.core` | `pydantic`, `httpx` |
| [`llm-catalog-pydantic-ai`](packages/llm-catalog-pydantic-ai) | [![PyPI](https://img.shields.io/pypi/v/llm-catalog-pydantic-ai)](https://pypi.org/project/llm-catalog-pydantic-ai/) | `llm_catalog.pydantic_ai` | core + `pydantic-ai` |
| [`llm-catalog-litellm`](packages/llm-catalog-litellm) | [![PyPI](https://img.shields.io/pypi/v/llm-catalog-litellm)](https://pypi.org/project/llm-catalog-litellm/) | `llm_catalog.litellm` | core + `litellm>=1.90.0` |

`llm-catalog-core` holds the config schema, validation, resolver, `GatewayTransport`, and LiteLLM codegen, and knows no runtime (and touches no filesystem).
The two adapters depend only on core; core imports neither adapter.

**Why three distributions, not one with extras?**
Python has no tree-shaking, so the only way to guarantee "don't pay for what you don't use" is to split the distributions.
A Pydantic AI user's environment and lockfile never reference `litellm`, and vice versa.

## Install

```bash
pip install llm-catalog-pydantic-ai   # Pydantic AI (pulls in core)
pip install llm-catalog-litellm       # LiteLLM plugin (pulls in core)
pip install llm-catalog-core          # core only (config / resolve / codegen)
```

## The config: `llm-catalog.json`

The config format is **JSON**, shared verbatim with `ai-sdk-catalog` (0.7) — write one file and hand it to both runtimes.
A provider is either **direct** (its `vendor` — string shorthand or a block with `baseURL` / `apiKey` / `headers` / `query` — defaults to the provider `id`) or **gateway-routed** (a `gateway` block with free-form `backends`, each naming its `vendor`; every model then names its `backend` key).
Roles point at a `(provider, model)` pair, written either as an object or as the `"provider:model"` shorthand.
Secrets are a literal string only for local endpoints; otherwise `{"envVarName": "..."}` reads the environment lazily (a gateway with no `apiKey` falls back to `AI_GATEWAY_API_KEY`).
See [`examples/llm-catalog.example.json`](examples/llm-catalog.example.json) for the full placeholder file.

```json
{
  "$schema": "./node_modules/ai-sdk-catalog/schema.json",
  "providers": [
    {
      "id": "anthropic",
      "models": [{ "id": "claude-sonnet-5" }]
    },
    {
      "id": "examplegw",
      "gateway": {
        "baseURL": "https://gateway.example.invalid/base",
        "apiKey": { "envVarName": "EXAMPLEGW_API_KEY" },
        "headers": { "Authorization": "Bearer {apiKey}" },
        "query": { "api-version": "2026-01-01" },
        "backends": {
          "claude": { "vendor": "anthropic", "pathTemplate": "anthropic/{slug}" },
          "gpt": { "vendor": "openai", "pathTemplate": "gpt/{slug}" },
          "gemini": {
            "vendor": "google",
            "pathTemplate": "gemini/{slug}:{action}",
            "actionMap": { "streamGenerateContent": "customStreamGenerateContent" }
          }
        }
      },
      "models": [
        { "id": "light-openai", "backend": "gpt", "api": "chat" },
        { "id": "light-anthropic", "backend": "claude" },
        { "id": "search-google", "backend": "gemini" }
      ]
    }
  ],
  "roles": {
    "chat": "anthropic:claude-sonnet-5",
    "fast": { "provider": "examplegw", "model": "light-openai" },
    "reasoning": "examplegw:light-anthropic",
    "search": "examplegw:search-google"
  }
}
```

`llm-catalog-core` ships the schema as `schema.json` inside the package (`llm_catalog/core/schema.json`; also exposed as `llm_catalog.core.config_json_schema()`), so editors validate and autocomplete the file — point `"$schema"` at whichever side's schema fits your repo.
Validation is strict on both sides: an unknown key fails loudly instead of being silently dropped.
The one Python-only extension is the `capabilities` block on a model (structured-output mode, grounding tools); since `ai-sdk-catalog` 0.7 rejects unknown keys, keep `capabilities` out of a file shared with TypeScript (its defaults then apply here) and use it only in Python-only configs.

Loading is explicit and filesystem-free in every package: read the file yourself and hand the parsed mapping over.
To keep using YAML, parse it yourself and pass the result the same way (`Catalog(yaml.safe_load(text))`).

## Usage

### 1. Pydantic AI (in-process)

```python
import json
from pathlib import Path

from pydantic_ai import Agent
from llm_catalog.pydantic_ai import PydanticAICatalog

config = json.loads(Path("llm-catalog.json").read_text(encoding="utf-8"))
cat = PydanticAICatalog(config)  # validates the config itself
agent = Agent(cat.model_for_role("fast"))

# structured output mode follows capabilities.structuredOutput (native/tool/prompted)
out = cat.output_for("reasoning", MySchema)
# grounding tools mapped from capabilities.grounding
tools = cat.grounding_tools("search")
```

This environment never installs `litellm`.

### 2. LiteLLM (in-process)

Call `register()` once to wire the handler into LiteLLM, then use `litellm` as usual.

```python
from llm_catalog.litellm import register

register()   # reads llm-catalog.json (LLM_CATALOG_CONFIG or default path)

import litellm

resp = litellm.completion(
    model="examplegw/fast",
    messages=[{"role": "user", "content": "hi"}],
)
```

Each process uses its own per-user key (`EXAMPLEGW_API_KEY`), and no server is needed.

### 3. LiteLLM proxy (central, "config only" for everyone)

Run one LiteLLM proxy that references the plugin from `config.yaml`.
See [`examples/litellm.proxy.example.yaml`](examples/litellm.proxy.example.yaml).
The system key lives in the proxy's env, and each user gets a **virtual key** to call `model="fast"` from any language or tool (OpenAI SDK, Pydantic AI, curl), with OpenAI-form responses.

```bash
pip install llm-catalog-litellm
export EXAMPLEGW_API_KEY=...                 # gateway system key
export LLM_CATALOG_CONFIG=/path/llm-catalog.json
export LITELLM_MASTER_KEY=...
litellm --config litellm.proxy.example.yaml
```

The handler resolves each model from `llm-catalog.json` itself, so the proxy config's `litellm_params` needs no gateway details (sidesteps LiteLLM #18216).
The proxy references `llm_catalog.litellm.handler` directly, so it does not call `register()`.

### Alternative: generate a native LiteLLM config (no plugin)

`llm-catalog-core` alone can emit a plain LiteLLM proxy config using LiteLLM's built-in providers.
The output is JSON — a subset of YAML, so LiteLLM's config loader reads it directly.

```bash
llm-catalog-codegen llm-catalog.json -o litellm.config.json
# or: python -m llm_catalog.core.codegen llm-catalog.json -o litellm.config.json
litellm --config litellm.config.json
```

This lean alternative does not honour a custom `pathTemplate` or native grounding, because it relies on LiteLLM's built-in path construction.
Use the `llm-catalog-litellm` plugin when those matter.

## Public / private boundary (strict)

This (public) repository ships generic code, placeholder examples, and mock tests only — no real base URLs, model ids, or capability values.
Your real `llm-catalog.json` is private data, distributed inside your org, and is never committed here.
Secrets (API keys) live only in env or a secret manager; `.gitignore` excludes `*.local.json`, `llm-catalog.json`, and `.env*`.

## Verification notes (§9 — confirm against your real gateway)

These depend on your gateway and the installed library versions, and are not asserted by the (mock-only) test suite.
Confirm them before relying on them; the code documents the fallback at each point.

1. Each backend's exact `pathTemplate` and auth header (standard vendor header vs needing `GatewayTransport(header_rewrite=...)`).
2. Whether google-genai honours a custom httpx client/transport; if not, fall back to building a genai `Client` and passing it via `GoogleProvider(client=...)`.
3. Whether Pydantic AI's builtin grounding tool emits the exact variant your gateway expects, else drive the raw vendor client.
4. Whether LiteLLM honours a custom client and the path rewrite takes effect (the plugin's route-1 strategy), verified against the mock gateway here for all three backends; the documented fallback (route 2) is hand-written native→OpenAI conversion.
5. LiteLLM gotchas mitigated in code: provider-id/built-in collision (#23352, warned at `register()` — direct providers naturally named `openai`/`anthropic` route to LiteLLM's built-ins, not the handler) and per-model `litellm_params` not reaching the handler (#18216, the handler self-resolves).
6. Pydantic AI output-mode symbols `NativeOutput` / `ToolOutput` / `PromptedOutput` (confirmed against the installed version).
7. LiteLLM pinned `>=1.90.0` (the verified floor; also past the 1.82.7 / 1.82.8 supply-chain incident).
8. uv workspace × PEP 420 namespace × editable install: after `uv sync`, all three `llm_catalog.*` import in one venv (covered by the test suite and CI).

## Development

Python 3.10+, [uv](https://docs.astral.sh/uv/) workspace.

```bash
uv sync                      # one venv, all three members editable
uv run pytest                # mock-only tests; no real gateway/keys
uv run ruff check . && uv run ruff format --check .
uv run ty check packages     # strict type check
uv build --all-packages      # build all three distributions
```

Local dev runs on Python 3.10 (the floor of the supported range, pinned in `.python-version`), so 3.10-incompatible code is caught immediately; CI runs the full 3.10–3.14 matrix.

## Releases

Releases are automated with [release-please](https://github.com/googleapis/release-please) (GitHub only).
Commits on `main` follow [Conventional Commits](https://www.conventionalcommits.org), and `feat:` / `fix:` entries drive each package's version bump.
release-please maintains a release PR that bumps the affected versions and updates their `CHANGELOG.md`; merging it tags each changed package (e.g. `llm-catalog-core-v0.2.0`) and publishes it to PyPI via Trusted Publishing (OIDC, no API token).
The three distributions are versioned and released independently.
