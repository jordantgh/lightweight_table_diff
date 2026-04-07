from __future__ import annotations

from typing import Any, Callable

import polars as pl

AdapterFn = Callable[..., pl.LazyFrame]

_registry: dict[str, AdapterFn] = {}


def register(name: str, adapter: AdapterFn) -> None:
    _registry[name] = adapter


def load(source_def: dict, **context: Any) -> pl.LazyFrame:
    adapter_type = source_def.get("type", "parquet")
    if adapter_type not in _registry:
        raise ValueError(
            f"Unknown adapter {adapter_type!r}. Registered: {sorted(_registry)}"
        )
    return _registry[adapter_type](source_def, **context)


from .csv import load_csv      # noqa: E402
from .parquet import load_parquet  # noqa: E402
from .sav import load_sav      # noqa: E402

register("parquet", load_parquet)
register("csv", load_csv)
register("sav", load_sav)