# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the snapshot generator itself (bucket renaming,
modality filtering, ordering) against a small fixture — the committed
``costs_gen.py`` is data, so its shape is checked separately in
``test_costs.py``."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_costs.py"
_spec = importlib.util.spec_from_file_location("generate_costs", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["generate_costs"] = _mod
_spec.loader.exec_module(_mod)
build_costs_module = _mod.build_costs_module


def fixture(anthropic_models: dict[str, Any]) -> dict[str, Any]:
    """A models.dev-shaped dump covering every bundled vendor."""
    empty: dict[str, Any] = {"models": {}}
    return {
        "anthropic": {"models": anthropic_models},
        "openai": empty,
        "mistral": empty,
        "cohere": empty,
        "groq": empty,
        "xai": empty,
        "deepseek": empty,
        "perplexity": empty,
        "google": empty,
    }


@pytest.fixture
def source() -> str:
    return build_costs_module(
        fixture(
            {
                "model-b": {
                    "cost": {
                        "input": 3,
                        "output": 15,
                        "cache_read": 0.3,
                        "cache_write": 3.75,
                    },
                    "modalities": {"output": ["text"]},
                },
                "model-a": {
                    "cost": {"input": 1, "output": 5},
                    "modalities": {"output": ["text"]},
                },
                "model-tts": {
                    "cost": {"input": 1, "output": 5},
                    "modalities": {"output": ["audio"]},
                },
                "model-free": {"modalities": {"output": ["text"]}},
            }
        )
    )


def test_renames_buckets_to_camelcase_and_sorts_models_by_id(source: str) -> None:
    assert "\"model-a\": {'input': 1, 'output': 5}," in source
    assert (
        "\"model-b\": {'input': 3, 'output': 15, 'cacheRead': 0.3, 'cacheWrite': 3.75},"
        in source
    )
    assert source.index('"model-a"') < source.index('"model-b"')


def test_skips_non_text_models_unpriced_models_and_empty_vendors(source: str) -> None:
    assert "model-tts" not in source  # no text output -> skipped
    assert "model-free" not in source  # no price -> skipped
    assert '"openai"' not in source  # no sheets at all -> vendor omitted


def test_fails_on_a_missing_vendor_and_on_an_unrealistic_price() -> None:
    with pytest.raises(ValueError, match=r"models\.dev has no"):
        build_costs_module({"anthropic": {"models": {}}})
    with pytest.raises(ValidationError, match="less than or equal to 1000"):
        build_costs_module(
            fixture(
                {
                    "pricey": {  # a per-1K unit mix-up
                        "cost": {"input": 1200},
                        "modalities": {"output": ["text"]},
                    }
                }
            )
        )
