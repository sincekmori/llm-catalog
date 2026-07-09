# Copyright 2026 Shinsuke Mori
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the shipped ``schema.json`` from the Pydantic config models.

Run from the repository root after changing ``config.py``:

    uv run python packages/llm-catalog-core/scripts/generate_schema.py

A test (``tests/test_schema.py``) fails if the shipped file drifts from the
models.
"""

import json
from pathlib import Path

from llm_catalog.core import config_json_schema


def main() -> None:
    out = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llm_catalog"
        / "core"
        / "schema.json"
    )
    out.write_text(json.dumps(config_json_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
