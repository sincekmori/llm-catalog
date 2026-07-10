# Changelog

## [0.3.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-litellm-v0.2.0...llm-catalog-litellm-v0.3.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* YAML support is dropped along with the pyyaml dependency; parse YAML yourself and pass the mapping to Catalog. load_config and parse_config are removed — Catalog(config) accepts the parsed mapping (or a CatalogConfig) and validates it. Catalog.from_file and PydanticAICatalog.from_file are removed; read the file yourself with json.loads. The LiteLLM handler's default config path changes from catalog.yaml to llm-catalog.json (JSON only). The api-on-non-openai config error is gone (adapters reject unsupported surfaces at use time), and gateway apiKeyEnvVarName is now optional.

### Features

* adopt the ai-sdk-catalog 0.5.0 config contract ([f8762fb](https://github.com/sincekmori/llm-catalog/commit/f8762fb7431b866f6f2bdd4147fe75610b4a245c))

## [0.2.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-litellm-v0.1.0...llm-catalog-litellm-v0.2.0) (2026-06-28)


### Features

* config-driven Pydantic AI and LiteLLM behind your own LLM gateway ([14a07c4](https://github.com/sincekmori/llm-catalog/commit/14a07c416ac9fea19d8ec4f467186d954391e483))


### Bug Fixes

* mark packages OS-independent; set per-package release components ([7270924](https://github.com/sincekmori/llm-catalog/commit/7270924c849e01f1308c0f46c4872b25c370ee50))

## 0.1.0 (2026-06-28)


### Features

* config-driven Pydantic AI and LiteLLM behind your own LLM gateway ([14a07c4](https://github.com/sincekmori/llm-catalog/commit/14a07c416ac9fea19d8ec4f467186d954391e483))
