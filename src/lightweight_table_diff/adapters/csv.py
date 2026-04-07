from pathlib import Path

import polars as pl


def load_csv(source_def: dict) -> pl.LazyFrame:
    path = Path(source_def["path"])
    glob = source_def.get("glob", "*.csv")
    scan_path = str(path / glob) if path.is_dir() else str(path)
    return pl.scan_csv(
        scan_path,
        infer_schema_length=source_def.get("infer_schema_length", 10_000),
        ignore_errors=source_def.get("ignore_errors", True),
    )