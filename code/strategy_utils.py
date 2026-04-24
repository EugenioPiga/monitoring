#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, Optional
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)

def create_spark(app_name: str, threads: int, shuffle_partitions: int, tmpdir: Optional[str] = None) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )
    if tmpdir:
        builder = builder.config("spark.local.dir", tmpdir)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def write_parquet(df: DataFrame, path: str, coalesce: Optional[int] = None) -> None:
    out = df
    if coalesce and coalesce > 0:
        out = out.coalesce(max(1, int(coalesce)))
    out.write.mode("overwrite").option("compression", "snappy").parquet(path)

def safe_divide(num, den):
    return F.when(den.isNotNull() & (den != 0), num / den).otherwise(F.lit(None))

def weighted_mean(value: str, weight: str, alias: str):
    return safe_divide(F.sum(F.col(value) * F.col(weight)), F.sum(F.col(weight))).alias(alias)

def add_naics3(df: DataFrame, source_col: str = "naics_code") -> DataFrame:
    if source_col not in df.columns:
        return df.withColumn("naics3", F.lit(None).cast("string"))
    return df.withColumn(
        "naics3",
        F.when(F.col(source_col).isNull(), F.lit(None).cast("string"))
        .otherwise(F.substring(F.regexp_replace(F.col(source_col).cast("string"), r"[^0-9]", ""), 1, 3))
    )

def get_existing(df: DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def add_forward_outcomes(panel: DataFrame, id_cols: list[str], order_col: str = "year") -> DataFrame:
    w = Window.partitionBy(*id_cols).orderBy(order_col)
    base_cols = [
        "n_workers", "workforce_weighted", "avg_salary", "hire_rate", "exit_rate",
        "skill_count_sd", "skill_bundle_dispersion", "skill_hhi_mean", "specialist_share"
    ]
    for col in base_cols:
        if col in panel.columns:
            panel = panel.withColumn(f"F5_{col}", F.lead(col, 5).over(w))
            panel = panel.withColumn(f"L1_{col}", F.lag(col, 1).over(w))
    if "n_workers" in panel.columns:
        panel = panel.withColumn("d5_log_workers", F.when((F.col("n_workers") > 0) & (F.col("F5_n_workers") > 0), 100.0 * (F.log(F.col("F5_n_workers")) - F.log(F.col("n_workers")))))
    if "workforce_weighted" in panel.columns:
        panel = panel.withColumn("d5_log_workforce", F.when((F.col("workforce_weighted") > 0) & (F.col("F5_workforce_weighted") > 0), 100.0 * (F.log(F.col("F5_workforce_weighted")) - F.log(F.col("workforce_weighted")))))
    if "avg_salary" in panel.columns:
        panel = panel.withColumn("d5_log_avg_salary", F.when((F.col("avg_salary") > 0) & (F.col("F5_avg_salary") > 0), 100.0 * (F.log(F.col("F5_avg_salary")) - F.log(F.col("avg_salary")))))
    for col in ["hire_rate", "exit_rate", "skill_count_sd", "skill_bundle_dispersion", "skill_hhi_mean", "specialist_share"]:
        if col in panel.columns and f"F5_{col}" in panel.columns:
            panel = panel.withColumn(f"d5_{col}", 100.0 * (F.col(f"F5_{col}") - F.col(col)))
    return panel

