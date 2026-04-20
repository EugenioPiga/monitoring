#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from pyspark.sql import SparkSession, DataFrame, functions as F

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive audit for final Parquet panel datasets.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the main Parquet dataset to audit (e.g. firm_year_panel).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where diagnostics will be written.",
    )
    parser.add_argument(
        "--key-cols",
        nargs="+",
        default=["firm_key", "year"],
        help="Key columns that should uniquely identify rows.",
    )
    parser.add_argument(
        "--year-col",
        default="year",
        help="Year column name.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="How many top duplicate keys / heavy hitters to save.",
    )
    parser.add_argument(
        "--master",
        default=None,
        help="Optional Spark master override, e.g. local[4]. Leave empty on cluster.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=None,
        help="Optional Spark shuffle partitions override.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_spark(app_name: str, master: str | None = None, shuffle_partitions: int | None = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    if shuffle_partitions is not None:
        spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
    spark.sparkContext.setLogLevel("WARN")
    return spark


def file_inventory(path: str) -> Dict[str, object]:
    root = Path(path)
    info = {
        "path_exists": root.exists(),
        "is_dir": root.is_dir(),
        "parquet_files": 0,
        "success_markers": 0,
        "temporary_dirs": 0,
        "total_bytes": 0,
    }
    if not root.exists():
        return info

    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if d == "_temporary":
                info["temporary_dirs"] += 1
        for f in filenames:
            fp = Path(dirpath) / f
            if f.endswith(".parquet"):
                info["parquet_files"] += 1
                try:
                    info["total_bytes"] += fp.stat().st_size
                except FileNotFoundError:
                    pass
            if f == "_SUCCESS":
                info["success_markers"] += 1
    return info


def bytes_to_gb(x: int) -> float:
    return round(x / (1024 ** 3), 3)


def write_json(obj: Dict[str, object], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_text(text: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def write_csv(df: DataFrame, path: str) -> None:
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(path)
    )


def spark_dtype_map(df: DataFrame) -> Dict[str, str]:
    return {name: dtype for name, dtype in df.dtypes}


def is_float_like(dtype: str) -> bool:
    return dtype in {"double", "float"}


def is_numeric(dtype: str) -> bool:
    return dtype in {
        "byte", "short", "int", "bigint", "long",
        "float", "double", "decimal", "decimal(38,18)"
    } or dtype.startswith("decimal")


def null_expr(col_name: str, dtype: str):
    c = F.col(col_name)
    if is_float_like(dtype):
        return F.sum(F.when(c.isNull() | F.isnan(c), 1).otherwise(0)).alias(col_name)
    return F.sum(F.when(c.isNull(), 1).otherwise(0)).alias(col_name)


def approx_distinct_expr(col_name: str):
    return F.approx_count_distinct(F.col(col_name)).alias(col_name)


def pretty_schema(df: DataFrame) -> str:
    return df._jdf.schema().treeString()

def to_float_or_none(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    spark = build_spark(
        app_name="panel_audit",
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
    )

    dataset_path = args.dataset_path
    output_dir = args.output_dir
    key_cols = args.key_cols
    year_col = args.year_col
    top_n = args.top_n

    # ------------------------------------------------------------------
    # 1. File-system level checks
    # ------------------------------------------------------------------
    fs_info = file_inventory(dataset_path)
    fs_info["total_gb"] = bytes_to_gb(fs_info["total_bytes"])
    write_json(fs_info, os.path.join(output_dir, "00_filesystem_inventory.json"))

    if not fs_info["path_exists"] or fs_info["parquet_files"] == 0:
        write_json(
            {"status": "FAIL", "reason": "Dataset path missing or contains zero parquet files."},
            os.path.join(output_dir, "00_status.json"),
        )
        spark.stop()
        return

    # ------------------------------------------------------------------
    # 2. Read dataset and basic structure
    # ------------------------------------------------------------------
    df = spark.read.parquet(dataset_path)

    basic = {
        "rows": df.count(),
        "columns": len(df.columns),
        "key_cols": key_cols,
        "year_col": year_col,
    }
    write_json(basic, os.path.join(output_dir, "01_basic_counts.json"))
    write_text(pretty_schema(df), os.path.join(output_dir, "01_schema.txt"))

    # Save a tiny sample
    write_csv(df.limit(100), os.path.join(output_dir, "01_sample_rows"))

    dtype_map = spark_dtype_map(df)

    # ------------------------------------------------------------------
    # 3. Key uniqueness and duplication checks
    # ------------------------------------------------------------------
    missing_key_cols = [c for c in key_cols if c not in df.columns]
    key_summary = {
        "missing_key_cols": missing_key_cols,
    }

    if not missing_key_cols:
        total_rows = basic["rows"]
        distinct_keys = df.select(*key_cols).distinct().count()

        dup_keys = (
            df.groupBy(*key_cols)
            .count()
            .where(F.col("count") > 1)
        )

        duplicate_key_cells = dup_keys.count()
        dup_stats_row = dup_keys.agg(
            F.max("count").alias("max_dup_count"),
            F.avg("count").alias("avg_dup_count_among_duplicate_keys"),
            F.sum("count").alias("rows_in_duplicate_key_cells"),
        ).collect()[0]

        key_summary.update({
            "total_rows": total_rows,
            "distinct_keys": distinct_keys,
            "duplicate_key_cells": duplicate_key_cells,
            "excess_rows_over_distinct_keys": total_rows - distinct_keys,
            "max_dup_count": dup_stats_row["max_dup_count"],
            "avg_dup_count_among_duplicate_keys": float(dup_stats_row["avg_dup_count_among_duplicate_keys"]) if dup_stats_row["avg_dup_count_among_duplicate_keys"] is not None else None,
            "rows_in_duplicate_key_cells": dup_stats_row["rows_in_duplicate_key_cells"],
        })

        write_csv(
            dup_keys.orderBy(F.col("count").desc()).limit(top_n),
            os.path.join(output_dir, "02_top_duplicate_keys"),
        )

        worst_dup_keys = dup_keys.orderBy(F.col("count").desc()).limit(min(top_n, 25))
        dup_sample = df.join(F.broadcast(worst_dup_keys.select(*key_cols)), on=key_cols, how="inner")
        write_csv(
            dup_sample.limit(5000),
            os.path.join(output_dir, "02_duplicate_key_row_samples"),
        )

    write_json(key_summary, os.path.join(output_dir, "02_key_uniqueness.json"))

    # ------------------------------------------------------------------
    # 4. Null counts + approx distinct by column
    # ------------------------------------------------------------------
    null_aggs = [null_expr(c, dtype_map[c]) for c in df.columns]
    approx_aggs = [approx_distinct_expr(c) for c in df.columns]

    null_row = df.agg(*null_aggs).collect()[0].asDict()
    approx_row = df.agg(*approx_aggs).collect()[0].asDict()

    rows_total = basic["rows"]
    profile_rows = []
    for c in df.columns:
        null_count = int(null_row[c]) if null_row[c] is not None else None
        approx_distinct = int(approx_row[c]) if approx_row[c] is not None else None
        profile_rows.append((
            c,
            dtype_map[c],
            null_count,
            (null_count / rows_total) if (null_count is not None and rows_total > 0) else None,
            approx_distinct,
        ))

    column_profile = spark.createDataFrame(
        profile_rows,
        schema=["column_name", "dtype", "null_count", "null_share", "approx_distinct_count"],
    )
    write_csv(column_profile.orderBy(F.col("null_share").desc()), os.path.join(output_dir, "03_column_profile"))

    # ------------------------------------------------------------------
    # 5. Numeric column summary stats
    # ------------------------------------------------------------------
    numeric_cols = [c for c, dt in dtype_map.items() if is_numeric(dt)]
    if numeric_cols:
        numeric_rows = []
        for c in numeric_cols:
            stats = df.select(
                F.min(F.col(c)).alias("min_value"),
                F.max(F.col(c)).alias("max_value"),
                F.mean(F.col(c)).alias("mean_value"),
                F.stddev(F.col(c)).alias("stddev_value"),
            ).collect()[0].asDict()

            numeric_rows.append((
                str(c),
                str(dtype_map[c]),
                to_float_or_none(stats.get("min_value")),
                to_float_or_none(stats.get("max_value")),
                to_float_or_none(stats.get("mean_value")),
                to_float_or_none(stats.get("stddev_value")),
            ))

        numeric_schema = StructType([
            StructField("column_name", StringType(), True),
            StructField("dtype", StringType(), True),
            StructField("min_value", DoubleType(), True),
            StructField("max_value", DoubleType(), True),
            StructField("mean_value", DoubleType(), True),
            StructField("stddev_value", DoubleType(), True),
        ])

        numeric_profile = spark.createDataFrame(numeric_rows, schema=numeric_schema)
        write_csv(numeric_profile, os.path.join(output_dir, "04_numeric_profile"))

    # ------------------------------------------------------------------
    # 6. Year-level sanity checks
    # ------------------------------------------------------------------
    if year_col in df.columns:
        year_aggs = [F.count(F.lit(1)).alias("rows")]
        if "firm_key" in df.columns:
            year_aggs.append(F.countDistinct("firm_key").alias("distinct_firms"))
        if "parent_rcid" in df.columns:
            year_aggs.append(F.countDistinct("parent_rcid").alias("distinct_parents"))

        by_year = df.groupBy(year_col).agg(*year_aggs).orderBy(F.col(year_col).asc())
        write_csv(by_year, os.path.join(output_dir, "05_rows_by_year"))

        year_summary_row = df.agg(
            F.min(F.col(year_col)).alias("min_year"),
            F.max(F.col(year_col)).alias("max_year"),
            F.countDistinct(F.col(year_col)).alias("distinct_years"),
        ).collect()[0].asDict()

        write_json(year_summary_row, os.path.join(output_dir, "05_year_summary.json"))

    # ------------------------------------------------------------------
    # 7. Firm-level frequency checks
    # ------------------------------------------------------------------
    if "firm_key" in df.columns:
        rows_per_firm = df.groupBy("firm_key").count()
        top_firms = rows_per_firm.orderBy(F.col("count").desc()).limit(top_n)
        write_csv(top_firms, os.path.join(output_dir, "06_top_firms_by_row_count"))

        firm_dist = rows_per_firm.agg(
            F.count(F.lit(1)).alias("distinct_firms"),
            F.min("count").alias("min_rows_per_firm"),
            F.expr("percentile_approx(count, 0.5)").alias("p50_rows_per_firm"),
            F.expr("percentile_approx(count, 0.9)").alias("p90_rows_per_firm"),
            F.expr("percentile_approx(count, 0.99)").alias("p99_rows_per_firm"),
            F.max("count").alias("max_rows_per_firm"),
        ).collect()[0].asDict()
        write_json(firm_dist, os.path.join(output_dir, "06_firm_row_distribution.json"))

    # ------------------------------------------------------------------
    # 8. Common indicator sanity checks
    # ------------------------------------------------------------------
    indicator_summary = {}
    for c in [
        "has_position_data",
        "has_posting_data",
        "has_people_analytics_firm_any_enriched_by_year",
        "is_first_people_analytics_firm_year_any_enriched",
        "is_first_people_analytics_position_year_any_enriched",
        "is_first_people_analytics_posting_year_any_enriched",
    ]:
        if c in df.columns:
            vals = (
                df.groupBy(c)
                .count()
                .orderBy(F.col(c).asc_nulls_last())
            )
            write_csv(vals, os.path.join(output_dir, f"07_indicator_distribution__{c}"))
            indicator_summary[c] = True
        else:
            indicator_summary[c] = False
    write_json(indicator_summary, os.path.join(output_dir, "07_indicator_summary.json"))

    if "has_position_data" in df.columns and "has_posting_data" in df.columns:
        cov = (
            df.groupBy("has_position_data", "has_posting_data")
            .count()
            .orderBy("has_position_data", "has_posting_data")
        )
        write_csv(cov, os.path.join(output_dir, "07_position_posting_coverage_crosstab"))

    # ------------------------------------------------------------------
    # 9. Overall status file
    # ------------------------------------------------------------------
    status = {
        "filesystem_ok": fs_info["parquet_files"] > 0 and fs_info["success_markers"] >= 1 and fs_info["temporary_dirs"] == 0,
        "main_dataset_rows": basic["rows"],
        "main_dataset_columns": basic["columns"],
        "key_cols_present": len(missing_key_cols) == 0,
        "duplicate_key_problem_detected": key_summary.get("duplicate_key_cells", 0) > 0 if not missing_key_cols else None,
    }
    write_json(status, os.path.join(output_dir, "99_status.json"))

    print("[INFO] Audit complete")
    print(json.dumps(status, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()
