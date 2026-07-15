# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM config generation (route 2a) — runs without litellm installed."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_catalog.core import Catalog, ConfigError, to_litellm_config
from llm_catalog.core.codegen import main


def generate(config_dict: dict[str, Any]) -> dict[str, Any]:
    return to_litellm_config(Catalog(config_dict).config)


def test_model_list_uses_builtin_providers(config_dict: dict[str, Any]) -> None:
    out = generate(config_dict)
    by_name = {m["model_name"]: m["litellm_params"] for m in out["model_list"]}

    assert by_name["examplegw/light-anthropic"]["model"] == "anthropic/light-anthropic"
    # openai backend, slug override applied
    assert by_name["examplegw/light-openai"]["model"] == "openai/oai-light"
    # google backend -> gemini built-in
    assert by_name["examplegw/search-google"]["model"] == "gemini/search-google"


def test_api_base_and_env_ref_no_secret(config_dict: dict[str, Any]) -> None:
    out = generate(config_dict)
    params = next(
        m["litellm_params"]
        for m in out["model_list"]
        if m["model_name"] == "examplegw/light-openai"
    )
    assert params["api_base"] == "https://gateway.example.invalid/base"
    # env reference, never the value
    assert params["api_key"] == "os.environ/EXAMPLEGW_API_KEY"


def test_gateway_default_env_when_no_api_key(config_dict: dict[str, Any]) -> None:
    del config_dict["providers"][0]["gateway"]["apiKey"]
    out = generate(config_dict)
    assert (
        out["model_list"][0]["litellm_params"]["api_key"]
        == "os.environ/AI_GATEWAY_API_KEY"
    )


def test_api_key_literal_carried_over(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["gateway"]["apiKey"] = "local-dummy"
    out = generate(config_dict)
    assert out["model_list"][0]["litellm_params"]["api_key"] == "local-dummy"


def test_settings_merged_and_translated(config_dict: dict[str, Any]) -> None:
    config_dict["providers"][0]["models"][1]["settings"]["maxOutputTokens"] = 4096
    out = generate(config_dict)
    params = next(
        m["litellm_params"]
        for m in out["model_list"]
        if m["model_name"] == "examplegw/light-openai"
    )
    assert params["temperature"] == 0.5  # model wins over provider default
    # camelCase config settings become LiteLLM parameter names
    assert params["max_tokens"] == 4096
    assert "maxOutputTokens" not in params


def test_role_aliases(config_dict: dict[str, Any]) -> None:
    out = generate(config_dict)
    aliases = out["router_settings"]["model_group_alias"]
    assert aliases["fast"] == "examplegw/light-openai"
    # the "provider:model" shorthand is resolved too
    assert aliases["search"] == "examplegw/search-google"


def test_unsupported_vendor_rejected(config_dict: dict[str, Any]) -> None:
    gw = config_dict["providers"][0]["gateway"]
    gw["backends"]["mistral"] = {"vendor": "mistral", "pathTemplate": "mistral/{slug}"}
    config_dict["providers"][0]["models"].append(
        {"id": "m-large", "backend": "mistral"}
    )
    with pytest.raises(ConfigError, match="no LiteLLM built-in"):
        generate(config_dict)


def test_headers_and_query_rejected(config_dict: dict[str, Any]) -> None:
    # The codegen cannot express transport extras; it must not drop them silently.
    config_dict["providers"][0]["gateway"]["headers"] = {"x-tenant": "acme"}
    with pytest.raises(ConfigError, match="llm-catalog-litellm"):
        generate(config_dict)


def test_direct_providers(direct_config_dict: dict[str, Any]) -> None:
    # headers/query on the fireworks block are rejected; drop them for this test
    fw_block = direct_config_dict["providers"][2]["vendor"]
    del fw_block["headers"]
    del fw_block["query"]
    out = generate(direct_config_dict)
    by_name = {m["model_name"]: m["litellm_params"] for m in out["model_list"]}

    plain = by_name["anthropic/claude-sonnet-5"]
    assert plain["model"] == "anthropic/claude-sonnet-5"
    # no endpoint/key overrides -> LiteLLM's own defaults apply
    assert "api_base" not in plain
    assert "api_key" not in plain

    fw = by_name["fireworks/accounts/fireworks/models/gpt-oss-120b"]
    # openai-compatible goes through LiteLLM's openai provider + api_base
    assert fw["model"] == "openai/accounts/fireworks/models/gpt-oss-120b"
    assert fw["api_base"] == "https://api.fireworks.example.invalid/inference/v1"
    assert fw["api_key"] == "os.environ/FIREWORKS_API_KEY"


def test_direct_headers_rejected(direct_config_dict: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="llm-catalog-litellm"):
        generate(direct_config_dict)  # the fireworks block carries headers/query


def test_json_output_is_valid_yaml(config_dict: dict[str, Any]) -> None:
    # The CLI writes JSON; LiteLLM's proxy loads its config with a YAML parser,
    # and JSON is a subset of YAML, so the generated file must round-trip.
    out = generate(config_dict)
    rendered = json.dumps(out, indent=2)
    assert yaml.safe_load(rendered) == out


def test_cli_writes_file(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    src = tmp_path / "llm-catalog.json"
    src.write_text(json.dumps(config_dict), encoding="utf-8")
    dst = tmp_path / "litellm.config.json"

    rc = main([str(src), "-o", str(dst)])
    assert rc == 0

    generated = json.loads(dst.read_text(encoding="utf-8"))
    assert "model_list" in generated
    assert (
        generated["router_settings"]["model_group_alias"]["fast"]
        == "examplegw/light-openai"
    )


def test_cli_rejects_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "llm-catalog.json"
    src.write_text("providers: []", encoding="utf-8")  # YAML, not JSON

    rc = main([str(src)])
    assert rc == 1
    assert "llm-catalog.json" in capsys.readouterr().err


def test_cli_rejects_invalid_config(
    tmp_path: Path, config_dict: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    config_dict["roles"]["fast"]["model"] = "ghost"
    src = tmp_path / "llm-catalog.json"
    src.write_text(json.dumps(config_dict), encoding="utf-8")

    rc = main([str(src)])
    assert rc == 1
    assert "ghost" in capsys.readouterr().err
