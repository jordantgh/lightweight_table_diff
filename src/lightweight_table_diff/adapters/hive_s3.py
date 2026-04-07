from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

logger = logging.getLogger(__name__)


def load_hive(source_def: dict, **context) -> pl.LazyFrame:
    import boto3

    spark = context.get("spark")
    if spark is None:
        raise RuntimeError("Hive adapter requires spark= to be passed")

    ssl_cert = context.get("ssl_cert")
    table_name = source_def["table"]
    cache_dir = Path(source_def.get("cache_dir", f"/tmp/hive_{table_name}"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = spark.sql(f"DESCRIBE FORMATTED {table_name}").collect()
    location = next(
        (r[1].strip() for r in rows if r[0] and "Location" in r[0]), None
    )
    if not location:
        raise RuntimeError(f"Could not resolve S3 location for '{table_name}'")

    parsed = urlparse(str(location).replace("s3a://", "s3://"))
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    client = boto3.client("s3")
    try:
        import raz_client

        if ssl_cert:
            raz_client.configure_ranger_raz(client, ssl_file=ssl_cert)
    except ImportError:
        pass

    logger.info("Downloading %s → %s", location, cache_dir)
    n_files = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".parquet"):
                continue
            n_files += 1
            relative = (
                obj["Key"][len(prefix) :].lstrip("/")
                if obj["Key"].startswith(prefix)
                else Path(obj["Key"]).name
            )
            dest = cache_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(dest))

    if not n_files:
        raise FileNotFoundError(f"No parquet files found at {location}")

    return pl.scan_parquet(
        str(cache_dir / "**/*.parquet"), hive_partitioning=True
    )
