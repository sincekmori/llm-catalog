# Changelog

## [0.6.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-core-v0.5.1...llm-catalog-core-v0.6.0) (2026-07-29)


### Features

* add the declarative per-model cost field (models.dev vocabulary, USD per 1M tokens) for schema parity with ai-sdk-catalog 0.8, exposed via ResolvedModel.cost ([2d95ebf](https://github.com/sincekmori/llm-catalog/commit/2d95ebf932a5d4ffb0004fcb8620a305ffea6f1d))

## [0.5.1](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-core-v0.5.0...llm-catalog-core-v0.5.1) (2026-07-27)


### Documentation

* rename config file references from llm-catalog.json to ai-sdk-catalog.json across READMEs, docstrings, examples, and test fixtures ([2e5c7b6](https://github.com/sincekmori/llm-catalog/commit/2e5c7b622163a0d28a2b94d3766906f19612e2f9))

## [0.5.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-core-v0.4.0...llm-catalog-core-v0.5.0) (2026-07-15)


### ⚠ BREAKING CHANGES

* restructure the config schema for parity with ai-sdk-catalog 0.7

### Features

* restructure the config schema for parity with ai-sdk-catalog 0.7 ([8050cc2](https://github.com/sincekmori/llm-catalog/commit/8050cc28f91afe583d352d2a65f7293705a85da5))

## [0.4.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-core-v0.3.0...llm-catalog-core-v0.4.0) (2026-07-10)


### Features

* add a body_rewrite hook to GatewayTransport and both adapters ([0682c8b](https://github.com/sincekmori/llm-catalog/commit/0682c8b9ded2212f1a1051db9904361a02ad92ae))

## [0.3.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-core-v0.2.0...llm-catalog-core-v0.3.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* YAML support is dropped along with the pyyaml dependency; parse YAML yourself and pass the mapping to Catalog. load_config and parse_config are removed — Catalog(config) accepts the parsed mapping (or a CatalogConfig) and validates it. Catalog.from_file and PydanticAICatalog.from_file are removed; read the file yourself with json.loads. The LiteLLM handler's default config path changes from catalog.yaml to llm-catalog.json (JSON only). The api-on-non-openai config error is gone (adapters reject unsupported surfaces at use time), and gateway apiKeyEnvVarName is now optional.

### Features

* adopt the ai-sdk-catalog 0.5.0 config contract ([f8762fb](https://github.com/sincekmori/llm-catalog/commit/f8762fb7431b866f6f2bdd4147fe75610b4a245c))

## [0.2.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-core-v0.1.0...llm-catalog-core-v0.2.0) (2026-06-28)


### Features

* config-driven Pydantic AI and LiteLLM behind your own LLM gateway ([14a07c4](https://github.com/sincekmori/llm-catalog/commit/14a07c416ac9fea19d8ec4f467186d954391e483))


### Bug Fixes

* mark packages OS-independent; set per-package release components ([7270924](https://github.com/sincekmori/llm-catalog/commit/7270924c849e01f1308c0f46c4872b25c370ee50))

## 0.1.0 (2026-06-28)


### Features

* config-driven Pydantic AI and LiteLLM behind your own LLM gateway ([14a07c4](https://github.com/sincekmori/llm-catalog/commit/14a07c416ac9fea19d8ec4f467186d954391e483))
