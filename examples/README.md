# llm-catalog examples

**Placeholders only.** Every value here is a dummy. Real base URLs, real model
ids, and real capability values are confidential "data" that belong in your
private environment — never commit them to this (public) repository. Secrets
(API keys) live in environment variables; a config stores only the *name* of the
env var via `"apiKey": { "envVarName": "..." }` (or, for a local endpoint, a
literal `apiKey`).

| File                                                         | What it shows                                                                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| [`llm-catalog.example.json`](llm-catalog.example.json)       | The full placeholder catalog: direct providers (bare and with a vendor block), one gateway provider with three backends, per-model `capabilities`, and roles in both spellings. |
| [`litellm.proxy.example.yaml`](litellm.proxy.example.yaml)   | A LiteLLM **proxy** config referencing the `llm-catalog-litellm` plugin (LiteLLM's own file, hence YAML).                |

The catalog config format is JSON, shared verbatim with
[`ai-sdk-catalog`](https://github.com/sincekmori/ai-sdk-utils/tree/main/packages/catalog)
— the same file drives both ecosystems. `"$schema"` points at the JSON Schema
shipped inside `llm-catalog-core`
([`schema.json`](../packages/llm-catalog-core/src/llm_catalog/core/schema.json),
here as a repo-relative path; from an installed environment use
`.venv/lib/pythonX.Y/site-packages/llm_catalog/core/schema.json`, or the raw
GitHub URL) so editors validate and autocomplete it. It is generated from the
Pydantic models by
[`scripts/generate_schema.py`](../packages/llm-catalog-core/scripts/generate_schema.py),
and a test fails if it drifts. Regenerate it after changing `config.py`:

```bash
uv run python packages/llm-catalog-core/scripts/generate_schema.py
```

One field is Python-only: `capabilities` (structured-output mode, multi-step
tool support, grounding tools) drives the Python adapters. `ai-sdk-catalog`
0.7+ validates strictly and **rejects** unknown keys, so a file that sets
`capabilities` no longer drives the TypeScript side. Keep `capabilities` out of
a shared file (its defaults then apply in Python) and use it only in
Python-only configs — like this example, which is not consumed by TypeScript.
