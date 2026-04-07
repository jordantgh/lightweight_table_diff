"""Column/row checking and key-uniqueness validation."""

from __future__ import annotations

import polars as pl


def get_cols_to_compare(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    keys: list[str],
    include_cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """Work out which non-key columns to compare, from schema intersection."""
    before_names = set(before.collect_schema().names())
    after_names = set(after.collect_schema().names())
    key_set = set(keys)

    missing = key_set - (before_names & after_names)
    if missing:
        raise ValueError(
            f"Join key(s) missing from one or both sides: {missing}"
        )

    if include_cols:
        cols = [
            c
            for c in include_cols
            if c in before_names and c in after_names and c not in key_set
        ]
    else:
        cols = sorted((before_names & after_names) - key_set)

    if exclude_cols:
        exclude_set = set(exclude_cols)
        cols = [c for c in cols if c not in exclude_set]

    if not cols:
        raise ValueError("No columns to compare after applying filters")

    return cols


def validate_key_uniqueness(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    keys: list[str],
    sample_limit: int = 20,
) -> None:
    """Raise if either frame has duplicate key combinations."""
    problems: list[str] = []
    for label, lf in [("before", before), ("after", after)]:
        dupes = (
            lf.select(*keys)
            .group_by(keys)
            .len()
            .filter(pl.col("len") > 1)
            .limit(sample_limit)
            .collect()
        )
        if dupes.height:
            problems.append(
                f"  {label}: {dupes.height} duplicate key group(s)\n{dupes}"
            )

    if problems:
        raise ValueError(
            "Duplicate keys (would cause row explosion):\n"
            + "\n".join(problems)
        )


def column_indels(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    keys: list[str],
) -> tuple[list[str], list[str]]:
    """Return (removed_cols, added_cols) relative to before → after."""
    key_set = set(keys)
    before_cols = set(before.collect_schema().names()) - key_set
    after_cols = set(after.collect_schema().names()) - key_set
    return sorted(before_cols - after_cols), sorted(after_cols - before_cols)


def row_indels(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    id_cols: list[str],
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Return (removed_rows, added_rows) as key-only LazyFrames."""
    before_keys = before.select(*id_cols)
    after_keys = after.select(*id_cols)
    removed = before_keys.join(after_keys, on=id_cols, how="anti")
    added = after_keys.join(before_keys, on=id_cols, how="anti")
    return removed, added
