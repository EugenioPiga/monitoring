#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession, DataFrame

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)

def create_spark(app_name: str, shuffle_partitions: int = 800, tmpdir: Optional[str] = None) -> SparkSession:
    builder = (SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true"))
    if tmpdir:
        builder = builder.config("spark.local.dir", tmpdir)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def write_parquet(df: DataFrame, path: str, coalesce: int | None = None) -> None:
    out = df.coalesce(int(coalesce)) if coalesce and coalesce > 0 else df
    out.write.mode("overwrite").option("compression", "snappy").parquet(path)
