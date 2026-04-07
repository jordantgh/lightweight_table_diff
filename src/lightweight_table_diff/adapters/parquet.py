from pathlib import Path

import polars as pl


def load_parquet(source_def: dict) -> pl.LazyFrame:
    path = Path(source_def["path"])
    glob = source_def.get("glob", "*.parquet")
    scan_path = str(path / glob) if path.is_dir() else str(path)
    return pl.scan_parquet(
        scan_path,
        hive_partitioning=source_def.get("hive_partitioning", False),
    )