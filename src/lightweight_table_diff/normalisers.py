"""
Normalisation transforms applied *before* diffing to suppress irrelevant
format differences between disparate sources.

Each normaliser has signature:  (lf, keys, cols) -> lf
"""

from __future__ import annotations

from typing import Callable

import polars as pl

NormaliserFn = Callable[[pl.LazyFrame, list[str], list[str]], pl.LazyFrame]

NULLISH = ["", "nan", "none", "<na>"]


def _norm_expr(expr: pl.Expr) -> pl.Expr:
    s = expr.cast(pl.String).str.strip_chars()
    s_lower = s.str.to_lowercase()
    s = (
        pl.when(s.is_null() | s_lower.is_in(NULLISH))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(s)
    )
    return s.str.replace(r"^(-?\d+)\.0+$", "${1}")


def normalise_float_strings(
    lf: pl.LazyFrame, keys: list[str], cols: list[str]
) -> pl.LazyFrame:
    """Cast everything to string, unify nulls, strip whitespace/trailing .0."""
    return lf.select(
        *[_norm_expr(pl.col(k)).alias(k) for k in keys],
        *[_norm_expr(pl.col(c)).alias(c) for c in cols],
    )


REGISTRY: dict[str, NormaliserFn] = {
    "float_strings": normalise_float_strings,
}
