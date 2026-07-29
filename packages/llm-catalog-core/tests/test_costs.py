# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Cost fallback: a model whose config omits ``cost`` gets the embedded
models.dev sheet for its vendor and id. Assertions compare against the
committed snapshot itself, so a price refresh never breaks them."""

from typing import Any

from llm_catalog.core import Catalog, ModelCost
from llm_catalog.core.config import VendorName
from llm_catalog.core.costs_gen import MODEL_COSTS


def _catalog(providers: list[dict[str, Any]]) -> Catalog:
    return Catalog({"providers": providers, "roles": {}})


def test_fills_direct_provider_cost_from_snapshot() -> None:
    cat = _catalog([{"id": "anthropic", "models": [{"id": "claude-sonnet-5"}]}])
    sheet = MODEL_COSTS["anthropic"]["claude-sonnet-5"]
    rm = cat.resolve_key("anthropic:claude-sonnet-5")
    assert rm.cost is not None
    assert rm.cost.as_dict() == sheet


def test_resolves_vendor_through_vendor_block_not_provider_id() -> None:
    cat = _catalog(
        [{"id": "my-proxy", "vendor": "openai", "models": [{"id": "gpt-4o"}]}]
    )
    rm = cat.resolve_key("my-proxy:gpt-4o")
    assert rm.cost is not None
    assert rm.cost.as_dict() == MODEL_COSTS["openai"]["gpt-4o"]


def test_explicit_cost_in_config_wins() -> None:
    cost = {"input": 1, "output": 2}
    cat = _catalog(
        [{"id": "anthropic", "models": [{"id": "claude-sonnet-5", "cost": cost}]}]
    )
    rm = cat.resolve_key("anthropic:claude-sonnet-5")
    assert rm.cost is not None
    assert rm.cost.as_dict() == {"input": 1.0, "output": 2.0}


def test_leaves_cost_none_for_an_unlisted_model_id() -> None:
    cat = _catalog([{"id": "anthropic", "models": [{"id": "claude-imaginary-9"}]}])
    assert cat.resolve_key("anthropic:claude-imaginary-9").cost is None


def test_leaves_cost_none_for_a_non_vendor_provider_id() -> None:
    # A direct provider whose id names no bundled vendor still resolves; the
    # snapshot has no sheet for it even when the model id itself is known.
    cat = _catalog([{"id": "local-llm", "models": [{"id": "gpt-4o"}]}])
    assert cat.resolve_key("local-llm:gpt-4o").cost is None


def test_leaves_cost_none_for_the_openai_compatible_vendor() -> None:
    # openai-compatible names a protocol, not an upstream with a price list.
    cat = _catalog(
        [
            {
                "id": "local",
                "vendor": {
                    "id": "openai-compatible",
                    "baseURL": "http://localhost:1234/v1",
                },
                "models": [{"id": "gpt-4o"}],
            }
        ]
    )
    assert cat.resolve_key("local:gpt-4o").cost is None


def test_fills_gateway_model_cost_from_its_backend_vendor(
    config_dict: dict[str, Any],
) -> None:
    config_dict["providers"][0]["models"].append(
        {"id": "claude-sonnet-5", "backend": "anthropic"}
    )
    rm = Catalog(config_dict).resolve_key("examplegw:claude-sonnet-5")
    assert rm.cost is not None
    assert rm.cost.as_dict() == MODEL_COSTS["anthropic"]["claude-sonnet-5"]


def test_snapshot_sanity() -> None:
    # Bundled vendors only, every sheet non-empty and a valid ModelCost.
    from typing import get_args

    vendors = set(get_args(VendorName))
    assert MODEL_COSTS
    for vendor, sheets in MODEL_COSTS.items():
        assert vendor in vendors
        assert vendor != "openai-compatible"
        assert sheets
        for sheet in sheets.values():
            assert sheet == ModelCost.model_validate(sheet).as_dict()
