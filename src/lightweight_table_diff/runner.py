from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from . import adapters
from .dimensions import (
    validate_key_uniqueness,
    get_cols_to_compare,
    column_indels,
    row_indels,
)
from .config import load_config
from .core import batch_diff_tbls
from .normalisers import REGISTRY as NORMALISER_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    name: str
    diff: pl.LazyFrame
    n_diffs: int
    removed_rows: pl.LazyFrame
    added_rows: pl.LazyFrame
    removed_cols: list[str]
    added_cols: list[str]


def run_comparison(job: dict, **context) -> ComparisonResult:
    job = copy.deepcopy(job)
    name = job.get("name", "unnamed")
    keys = job["join_keys"]
    logger.info("Running comparison: %s", name)

    before = adapters.load(job["before"], **context)
    after = adapters.load(job["after"], **context)

    removed_cols, added_cols = column_indels(before, after, keys)
    if removed_cols:
        logger.info("  %d column(s) removed: %s", len(removed_cols), removed_cols)
    if added_cols:
        logger.info("  %d column(s) added: %s", len(added_cols), added_cols)

    cols = get_cols_to_compare(
        before,
        after,
        keys,
        include_cols=job.get("compare_cols"),
        exclude_cols=job.get("exclude_cols"),
    )
    logger.info("  %d column(s) to compare", len(cols))
    validate_key_uniqueness(before, after, keys)

    removed, added = row_indels(before, after, keys)

    if norm_name := job.get("normalisation"):
        if norm_name not in NORMALISER_REGISTRY:
            raise ValueError(f"Unknown normaliser: {norm_name!r}")
        norm_fn = NORMALISER_REGISTRY[norm_name]
        before = norm_fn(before, keys, cols)
        after = norm_fn(after, keys, cols)

    diff = batch_diff_tbls(
        before,
        after,
        keys,
        cols,
        batch_size=job.get("batch_size", 50),
        join_type=job.get("join_type", "full"),
    )

    n = diff.select(pl.len()).collect().item()
    logger.info("  %d difference(s) found", n)

    return ComparisonResult(
        name=name,
        diff=diff,
        n_diffs=n,
        removed_rows=removed,
        added_rows=added,
        removed_cols=removed_cols,
        added_cols=added_cols,
    )


def write_results(result: ComparisonResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- column indels (just a small text/csv file) ----------------------------
    if result.removed_cols or result.added_cols:
        col_indels_path = out / f"{result.name}_column_indels.csv"
        max_len = max(len(result.removed_cols), len(result.added_cols))
        pl.DataFrame(
            {
                "removed_columns": result.removed_cols + [""] * (max_len - len(result.removed_cols)),
                "added_columns": result.added_cols + [""] * (max_len - len(result.added_cols)),
            }
        ).write_csv(col_indels_path)
        logger.info("  Wrote %s", col_indels_path)

    # -- row indels ------------------------------------------------------------
    for label, lf in [
        ("removed", result.removed_rows),
        ("added", result.added_rows),
    ]:
        n = lf.select(pl.len()).collect().item()
        if n > 0:
            path = out / f"{result.name}_{label}_rows.csv"
            lf.sink_csv(path)
            logger.info("  %d %s row(s) → %s", n, label, path)
        else:
            logger.info("  No %s rows", label)

    # -- cell-level diff -------------------------------------------------------
    if result.n_diffs == 0:
        logger.info("  %s: no cell differences", result.name)
        return

    detail_path = out / f"{result.name}_detailed.csv"
    result.diff.sink_csv(detail_path)
    logger.info("  Wrote %s", detail_path)

    summary_path = out / f"{result.name}_summary.csv"
    (
        result.diff.group_by("col_name", "before_val", "after_val")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .sink_csv(summary_path)
    )
    logger.info("  Wrote %s", summary_path)


def run_config(config_path: str | Path, **context) -> list[ComparisonResult]:
    jobs = load_config(config_path)
    results: list[ComparisonResult] = []
    for job in jobs:
        result = run_comparison(job, **context)
        write_results(result, job.get("output_dir", "./diff_output"))
        results.append(result)
    return results