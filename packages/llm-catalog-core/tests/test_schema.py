# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""The shipped schema.json and example config stay in sync with the models."""

import json
from importlib import resources
from pathlib import Path

from llm_catalog.core import Catalog, config_json_schema


def test_shipped_schema_matches_models() -> None:
    # Regenerate with: uv run python packages/*/scripts/generate_schema.py
    shipped = json.loads(
        resources.files("llm_catalog.core").joinpath("schema.json").read_text("utf-8")
    )
    assert shipped == config_json_schema()


def test_schema_declares_dialect_and_closes_objects() -> None:
    schema = config_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    # extra="forbid" must surface as closed objects, so editors catch typos.
    assert schema["additionalProperties"] is False


def test_example_config_is_valid() -> None:
    # The shipped placeholder example must always validate.
    example = (
        Path(__file__).resolve().parents[3] / "examples" / "llm-catalog.example.json"
    )
    catalog = Catalog(json.loads(example.read_text(encoding="utf-8")))
    assert set(catalog.roles) == {"fast", "reasoning", "search"}
