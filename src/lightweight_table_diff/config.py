"""YAML config loading and deep-merge expansion."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    _merge_in_place(result, override)
    return result


def _merge_in_place(target: dict, source: dict) -> None:
    for k, v in source.items():
        if k in target and isinstance(v, dict) and isinstance(target[k], dict):
            _merge_in_place(target[k], v)
        else:
            target[k] = copy.deepcopy(v)


def expand_comparisons(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Split shared defaults + 'comparisons' list into per-job dicts."""
    base = {k: v for k, v in raw.items() if k != "comparisons"}
    items = raw.get("comparisons", [{}])
    if not isinstance(items, list):
        raise TypeError(
            f"'comparisons' must be a list, got {type(items).__name__}"
        )
    return [deep_merge(base, item) for item in items]


def load_config(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return expand_comparisons(raw)
