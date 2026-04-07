"""
Cell-level table differ for Polars.

Produces long-form (keys…, col_name, before_val, after_val) for every
cell that changed between two LazyFrames.
"""

from __future__ import annotations

import logging

import polars as pl
from polars_checkpoint import CheckpointSession, checkpoint

logger = logging.getLogger(__name__)


def diff_tbls(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    id_cols: list[str],
    compare_cols: list[str] | None = None,
    join_type: str = "full",
) -> pl.LazyFrame:
    if compare_cols is None:
        compare_cols = list(before.drop(*id_cols).collect_schema().keys())

    bef_cols = [pl.col(c).alias(f"b__{c}") for c in compare_cols]
    aft_cols = [pl.col(c).alias(f"a__{c}") for c in compare_cols]
    before = before.select(*id_cols, *bef_cols)
    after = after.select(*id_cols, *aft_cols)

    joined = before.join(after, on=id_cols, how=join_type, coalesce=True)

    diff_structs = [
        pl.when(~pl.col(f"b__{c}").eq_missing(pl.col(f"a__{c}")))
        .then(
            pl.struct(
                pl.col(f"b__{c}").alias("before_val"),
                pl.col(f"a__{c}").alias("after_val"),
            )
        )
        .otherwise(None)
        .alias(c)
        for c in compare_cols
    ]

    return (
        joined.select(*id_cols, *diff_structs)
        .unpivot(
            on=compare_cols,
            index=id_cols,
            variable_name="col_name",
            value_name="diff",
        )
        .drop_nulls("diff")
        .select(
            *id_cols,
            "col_name",
            pl.col("diff").struct.field("before_val"),
            pl.col("diff").struct.field("after_val"),
        )
    )


def batch_diff_tbls(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    id_cols: list[str],
    compare_cols: list[str] | None = None,
    batch_size: int = 50,
    join_type: str = "full",
) -> pl.LazyFrame:
    if compare_cols is None:
        compare_cols = list(before.drop(*id_cols).collect_schema().keys())

    parts = []
    n = len(compare_cols)
    for i in range(0, n, batch_size):
        batch = compare_cols[i : i + batch_size]
        logger.info("  batch %d-%d of %d columns", i + 1, min(i + len(batch), n), n)
        diff = diff_tbls(before, after, id_cols, batch, join_type=join_type)
        parts.append(checkpoint(diff))

    return pl.concat(parts)
