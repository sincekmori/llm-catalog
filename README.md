# llm-catalog

[![CI](https://github.com/sincekmori/llm-catalog/actions/workflows/ci.yml/badge.svg)](https://github.com/sincekmori/llm-catalog/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/sincekmori/llm-catalog)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/llm-catalog-core)](https://pypi.org/project/llm-catalog-core/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/types-ty-261230)](https://github.com/astral-sh/ty)

Drive multiple LLM runtimes from one declarative `catalog.yaml` — implement nothing, configure once.
It is the Python counterpart of [`ai-sdk-catalog`](https://github.com/sincekmori/ai-sdk-utils) (Vercel AI SDK), reading the same config structure.
You reference a **role** name (`fast`, `reasoning`, `search`, …) and it resolves to a concrete model behind your own LLM gateway.

It targets two runtimes from the same config and the same core.

- **Pydantic AI** — in-process, native fidelity.
- **LiteLLM** — OpenAI-compatible, both in-process (SDK) and as a central proxy.

Gateway quirks (path layout, action names, auth headers, structured-output mode, grounding tools) are honoured as config values, never hardcoded.
The same generic code therefore works for any gateway that exposes native provider formats under a path prefix.

## Packages

Three independently versioned distributions form a uv workspace and share the PEP 420 namespace `llm_catalog`.

| Distribution | PyPI | Import | Depends on |
|---|---|---|---|
| [`llm-catalog-core`](packages/llm-catalog-core) | [![PyPI](https://img.shields.io/pypi/v/llm-catalog-core)](https://pypi.org/project/llm-catalog-core/) | `llm_catalog.core` | `pydantic`, `pyyaml`, `httpx` |
| [`llm-catalog-pydantic-ai`](packages/llm-catalog-pydantic-ai) | [![PyPI](https://img.shields.io/pypi/v/llm-catalog-pydantic-ai)](https://pypi.org/project/llm-catalog-pydantic-ai/) | `llm_catalog.pydantic_ai` | core + `pydantic-ai` |
| [`llm-catalog-litellm`](packages/llm-catalog-litellm) | [![PyPI](https://img.shields.io/pypi/v/llm-catalog-litellm)](https://pypi.org/project/llm-catalog-litellm/) | `llm_catalog.litellm` | core + `litellm>=1.90.0` |

`llm-catalog-core` holds the config schema, loader, resolver, `GatewayTransport`, and LiteLLM codegen, and knows no runtime.
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

## `catalog.yaml`

Roles point at a `(provider, model)`, and each model names the gateway `backend` that serves it.
camelCase keys (`baseURL`, `apiKeyEnvVarName`, `pathTemplate`, `actionMap`) are accepted as-is for parity with `ai-sdk-catalog`.
See [`examples/catalog.example.yaml`](examples/catalog.example.yaml) for the full placeholder file.

```yaml
providers:
  - id: examplegw
    gateway:
      baseURL: https://gateway.example.invalid/base
      apiKeyEnvVarName: EXAMPLEGW_API_KEY      # env var NAME (value stays in env)
      backends:
        anthropic: { pathTemplate: "anthropic/{slug}" }
        openai:    { pathTemplate: "gpt/{slug}" }
        google:
          pathTemplate: "gemini/{slug}:{action}"
          actionMap: { streamGenerateContent: customStreamGenerateContent }
    models:
      - { id: light-openai, backend: openai, api: chat }
      - { id: light-anthropic, backend: anthropic }
      - { id: search-google, backend: google }
roles:
  fast:      { provider: examplegw, model: light-openai }
  reasoning: { provider: examplegw, model: light-anthropic }
  search:    { provider: examplegw, model: search-google }
```

## Usage

### 1. Pydantic AI (in-process)

```python
from pydantic_ai import Agent
from llm_catalog.pydantic_ai import PydanticAICatalog

cat = PydanticAICatalog.from_file("catalog.yaml")
agent = Agent(cat.model_for_role("fast"))

# structured output mode follows capabilities.structured_output (native/tool/prompted)
out = cat.output_for("reasoning", MySchema)
# grounding tools mapped from capabilities.grounding
tools = cat.grounding_tools("search")
```

This environment never installs `litellm`.

### 2. LiteLLM (in-process)

Call `register()` once to wire the handler into LiteLLM, then use `litellm` as usual.

```python
from llm_catalog.litellm import register

register()   # reads catalog.yaml (LLM_CATALOG_CONFIG or default path)

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
export LLM_CATALOG_CONFIG=/path/catalog.yaml
export LITELLM_MASTER_KEY=...
litellm --config litellm.proxy.example.yaml
```

The handler resolves each model from `catalog.yaml` itself, so `config.yaml`'s `litellm_params` needs no gateway details (sidesteps LiteLLM #18216).
The proxy references `llm_catalog.litellm.handler` directly, so it does not call `register()`.

### Alternative: generate a native LiteLLM config (no plugin)

`llm-catalog-core` alone can emit a plain LiteLLM `config.yaml` using LiteLLM's built-in providers.

```bash
python -m llm_catalog.core.codegen catalog.yaml -o litellm.config.yaml
# or: llm-catalog-codegen catalog.yaml -o litellm.config.yaml
```

This lean alternative does not honour a custom `pathTemplate` or native grounding, because it relies on LiteLLM's built-in path construction.
Use the `llm-catalog-litellm` plugin when those matter.

## Public / private boundary (strict)

This (public) repository ships generic code, placeholder examples, and mock tests only — no real base URLs, model ids, or capability values.
Your real `catalog.yaml` is private data, distributed inside your org, and is never committed here.
Secrets (API keys) live only in env or a secret manager; `.gitignore` excludes `*.local.yaml`, `catalog.yaml`, and `.env*`.

## Verification notes (§9 — confirm against your real gateway)

These depend on your gateway and the installed library versions, and are not asserted by the (mock-only) test suite.
Confirm them before relying on them; the code documents the fallback at each point.

1. Each backend's exact `pathTemplate` and auth header (standard vendor header vs needing `GatewayTransport(header_rewrite=...)`).
2. Whether google-genai honours a custom httpx client/transport; if not, fall back to building a genai `Client` and passing it via `GoogleProvider(client=...)`.
3. Whether Pydantic AI's builtin grounding tool emits the exact variant your gateway expects, else drive the raw vendor client.
4. Whether LiteLLM honours a custom client and the path rewrite takes effect (the plugin's route-1 strategy), verified against the mock gateway here for all three backends; the documented fallback (route 2) is hand-written native→OpenAI conversion.
5. LiteLLM gotchas mitigated in code: provider-id/built-in collision (#23352, warned at load) and per-model `litellm_params` not reaching the handler (#18216, the handler self-resolves).
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
