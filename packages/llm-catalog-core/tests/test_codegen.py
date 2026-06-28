# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM config generation (route 2a) — runs without litellm installed."""

from pathlib import Path
from typing import Any

import yaml

from llm_catalog.core import parse_config, to_litellm_config
from llm_catalog.core.codegen import main


def test_model_list_uses_builtin_providers(config_dict: dict[str, Any]) -> None:
    out = to_litellm_config(parse_config(config_dict))
    by_name = {m["model_name"]: m["litellm_params"] for m in out["model_list"]}

    assert by_name["examplegw/light-anthropic"]["model"] == "anthropic/light-anthropic"
    # openai backend, slug override applied
    assert by_name["examplegw/light-openai"]["model"] == "openai/oai-light"
    # google backend -> gemini built-in
    assert by_name["examplegw/search-google"]["model"] == "gemini/search-google"


def test_api_base_and_env_ref_no_secret(config_dict: dict[str, Any]) -> None:
    out = to_litellm_config(parse_config(config_dict))
    params = next(
        m["litellm_params"]
        for m in out["model_list"]
        if m["model_name"] == "examplegw/light-openai"
    )
    assert params["api_base"] == "https://gateway.example.invalid/base"
    # env reference, never the value
    assert params["api_key"] == "os.environ/EXAMPLEGW_API_KEY"


def test_settings_merged_into_params(config_dict: dict[str, Any]) -> None:
    out = to_litellm_config(parse_config(config_dict))
    params = next(
        m["litellm_params"]
        for m in out["model_list"]
        if m["model_name"] == "examplegw/light-openai"
    )
    assert params["temperature"] == 0.5  # model wins over provider default


def test_role_aliases(config_dict: dict[str, Any]) -> None:
    out = to_litellm_config(parse_config(config_dict))
    aliases = out["router_settings"]["model_group_alias"]
    assert aliases["fast"] == "examplegw/light-openai"
    assert aliases["search"] == "examplegw/search-google"


def test_output_is_valid_yaml(config_dict: dict[str, Any]) -> None:
    out = to_litellm_config(parse_config(config_dict))
    rendered = yaml.safe_dump(out, sort_keys=False)
    assert yaml.safe_load(rendered) == out


def test_cli_writes_file(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    src = tmp_path / "catalog.yaml"
    src.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    dst = tmp_path / "litellm.config.yaml"

    rc = main([str(src), "-o", str(dst)])
    assert rc == 0

    generated = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert "model_list" in generated
    assert (
        generated["router_settings"]["model_group_alias"]["fast"]
        == "examplegw/light-openai"
    )
