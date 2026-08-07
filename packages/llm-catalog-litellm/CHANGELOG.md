# Changelog

## [0.6.1](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-litellm-v0.6.0...llm-catalog-litellm-v0.6.1) (2026-08-07)


### Bug Fixes

* constrain llm-catalog-core to &gt;=0.8.1,&lt;0.9 in both adapters so published wheels resolve a core that exports BodyRewrite instead of the incompatible 0.3 line ([215032a](https://github.com/sincekmori/llm-catalog/commit/215032a366c1579d3b85f8d49a97f28119778fa9))

## [0.6.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-litellm-v0.5.0...llm-catalog-litellm-v0.6.0) (2026-07-27)


### Features

* default the config filename to ai-sdk-catalog.json, keeping llm-catalog.json as a deprecated fallback after LLM_CATALOG_CONFIG ([c44b7a3](https://github.com/sincekmori/llm-catalog/commit/c44b7a35f6db68cd13782b71445321973cf9eaa1))

## [0.5.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-litellm-v0.4.0...llm-catalog-litellm-v0.5.0) (2026-07-15)


### ⚠ BREAKING CHANGES

* restructure the config schema for parity with ai-sdk-catalog 0.7

### Features

* restructure the config schema for parity with ai-sdk-catalog 0.7 ([8050cc2](https://github.com/sincekmori/llm-catalog/commit/8050cc28f91afe583d352d2a65f7293705a85da5))

## [0.4.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-litellm-v0.3.0...llm-catalog-litellm-v0.4.0) (2026-07-10)


### Features

* add a body_rewrite hook to GatewayTransport and both adapters ([0682c8b](https://github.com/sincekmori/llm-catalog/commit/0682c8b9ded2212f1a1051db9904361a02ad92ae))

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
