# Changelog

## [0.7.1](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.7.0...llm-catalog-pydantic-ai-v0.7.1) (2026-10-10)


### Documentation

* move every example, test and comment to the current model generation (claude-sonnet-5-5, claude-opus-5-5, gpt-6-astra, gpt-6-luna, gemini-3.8-flash, qwen3.8) to match ai-sdk-catalog ([e0b5e2f](https://github.com/sincekmori/llm-catalog/commit/e0b5e2f22084da57333942d7a5fa43c5f019384c))

## [0.7.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.6.0...llm-catalog-pydantic-ai-v0.7.0) (2026-10-10)


### Features

* support Python 3.15 in llm-catalog-core, llm-catalog-pydantic-ai, and llm-catalog-ai-sdk, and cap llm-catalog-litellm at Python &lt;3.15 to mirror LiteLLM's own requires-python ([1a9126b](https://github.com/sincekmori/llm-catalog/commit/1a9126bb4238f84f683f04db625003760c06390b))


### Bug Fixes

* require llm-catalog-core 0.10 in llm-catalog-pydantic-ai, llm-catalog-ai-sdk, and llm-catalog-litellm so the adapters released alongside core 0.10.0 actually install it instead of staying on the 0.9 line ([61d1253](https://github.com/sincekmori/llm-catalog/commit/61d12539003b8dfebae6305339ce429290ae3b0a))

## [0.6.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.5.2...llm-catalog-pydantic-ai-v0.6.0) (2026-10-04)


### ⚠ BREAKING CHANGES

* add the llm-catalog-ai-sdk adapter for the AI SDK for Python, move to httpx2 with a second GatewayTransport flavour that fixes Anthropic models in the Pydantic AI adapter, reach schema parity with ai-sdk-catalog 0.13, require Python 3.12+, and split llm-catalog-litellm into its own uv project

### Features

* add the llm-catalog-ai-sdk adapter for the AI SDK for Python, move to httpx2 with a second GatewayTransport flavour that fixes Anthropic models in the Pydantic AI adapter, reach schema parity with ai-sdk-catalog 0.13, require Python 3.12+, and split llm-catalog-litellm into its own uv project ([f07e226](https://github.com/sincekmori/llm-catalog/commit/f07e22674cb1cced755d3a859b4fb810b355a1fc))
* **pydantic-ai:** build each model once and reuse it so repeated lookups share a connection pool, and add aclose() and async-with support to close the HTTP clients the catalog opened ([09b4943](https://github.com/sincekmori/llm-catalog/commit/09b49430aa46d555ed88735e803edf9d2a0ae6d6))

## [0.5.2](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.5.1...llm-catalog-pydantic-ai-v0.5.2) (2026-08-07)


### Bug Fixes

* constrain llm-catalog-core to &gt;=0.8.1,&lt;0.9 in both adapters so published wheels resolve a core that exports BodyRewrite instead of the incompatible 0.3 line ([215032a](https://github.com/sincekmori/llm-catalog/commit/215032a366c1579d3b85f8d49a97f28119778fa9))

## [0.5.1](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.5.0...llm-catalog-pydantic-ai-v0.5.1) (2026-07-27)


### Documentation

* rename config file references from llm-catalog.json to ai-sdk-catalog.json across READMEs, docstrings, examples, and test fixtures ([2e5c7b6](https://github.com/sincekmori/llm-catalog/commit/2e5c7b622163a0d28a2b94d3766906f19612e2f9))

## [0.5.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.4.0...llm-catalog-pydantic-ai-v0.5.0) (2026-07-15)


### ⚠ BREAKING CHANGES

* restructure the config schema for parity with ai-sdk-catalog 0.7

### Features

* restructure the config schema for parity with ai-sdk-catalog 0.7 ([8050cc2](https://github.com/sincekmori/llm-catalog/commit/8050cc28f91afe583d352d2a65f7293705a85da5))

## [0.4.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.3.0...llm-catalog-pydantic-ai-v0.4.0) (2026-07-10)


### Features

* add a body_rewrite hook to GatewayTransport and both adapters ([0682c8b](https://github.com/sincekmori/llm-catalog/commit/0682c8b9ded2212f1a1051db9904361a02ad92ae))

## [0.3.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.2.0...llm-catalog-pydantic-ai-v0.3.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* YAML support is dropped along with the pyyaml dependency; parse YAML yourself and pass the mapping to Catalog. load_config and parse_config are removed — Catalog(config) accepts the parsed mapping (or a CatalogConfig) and validates it. Catalog.from_file and PydanticAICatalog.from_file are removed; read the file yourself with json.loads. The LiteLLM handler's default config path changes from catalog.yaml to llm-catalog.json (JSON only). The api-on-non-openai config error is gone (adapters reject unsupported surfaces at use time), and gateway apiKeyEnvVarName is now optional.

### Features

* adopt the ai-sdk-catalog 0.5.0 config contract ([f8762fb](https://github.com/sincekmori/llm-catalog/commit/f8762fb7431b866f6f2bdd4147fe75610b4a245c))

## [0.2.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-pydantic-ai-v0.1.0...llm-catalog-pydantic-ai-v0.2.0) (2026-06-28)


### Features

* config-driven Pydantic AI and LiteLLM behind your own LLM gateway ([14a07c4](https://github.com/sincekmori/llm-catalog/commit/14a07c416ac9fea19d8ec4f467186d954391e483))


### Bug Fixes

* mark packages OS-independent; set per-package release components ([7270924](https://github.com/sincekmori/llm-catalog/commit/7270924c849e01f1308c0f46c4872b25c370ee50))

## 0.1.0 (2026-06-28)


### Features

* config-driven Pydantic AI and LiteLLM behind your own LLM gateway ([14a07c4](https://github.com/sincekmori/llm-catalog/commit/14a07c416ac9fea19d8ec4f467186d954391e483))
