# Changelog

## [0.2.0](https://github.com/sincekmori/llm-catalog/compare/llm-catalog-ai-sdk-v0.1.0...llm-catalog-ai-sdk-v0.2.0) (2026-10-10)


### Features

* support Python 3.15 in llm-catalog-core, llm-catalog-pydantic-ai, and llm-catalog-ai-sdk, and cap llm-catalog-litellm at Python &lt;3.15 to mirror LiteLLM's own requires-python ([1a9126b](https://github.com/sincekmori/llm-catalog/commit/1a9126bb4238f84f683f04db625003760c06390b))


### Bug Fixes

* allow ai 0.8.x in llm-catalog-ai-sdk, whose 0.8.0 breaking changes (stream buffering, evaluate typing, ops namespace) do not touch the provider, Model, protocol, or InferenceRequestParams surface this adapter uses ([d49f358](https://github.com/sincekmori/llm-catalog/commit/d49f3588f7d5d1689f184368998221ed0e7b569a))
* require llm-catalog-core 0.10 in llm-catalog-pydantic-ai, llm-catalog-ai-sdk, and llm-catalog-litellm so the adapters released alongside core 0.10.0 actually install it instead of staying on the 0.9 line ([61d1253](https://github.com/sincekmori/llm-catalog/commit/61d12539003b8dfebae6305339ce429290ae3b0a))

## 0.1.0 (2026-10-04)


### ⚠ BREAKING CHANGES

* add the llm-catalog-ai-sdk adapter for the AI SDK for Python, move to httpx2 with a second GatewayTransport flavour that fixes Anthropic models in the Pydantic AI adapter, reach schema parity with ai-sdk-catalog 0.13, require Python 3.12+, and split llm-catalog-litellm into its own uv project

### Features

* add the llm-catalog-ai-sdk adapter for the AI SDK for Python, move to httpx2 with a second GatewayTransport flavour that fixes Anthropic models in the Pydantic AI adapter, reach schema parity with ai-sdk-catalog 0.13, require Python 3.12+, and split llm-catalog-litellm into its own uv project ([f07e226](https://github.com/sincekmori/llm-catalog/commit/f07e22674cb1cced755d3a859b4fb810b355a1fc))
