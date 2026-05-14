#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pyspark.sql import SparkSession, functions as F


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Safely augment an existing parent-year first-pass panel with AI/HR/manager "
            "variables from the new firm-year AI/HR/manager panel. Writes to a new output dir."
        )
    )
    p.add_argument("--parent-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass")
    p.add_argument("--firm-year-ai-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/firm_year_panel_ai_hr_manager_safe_v3")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass_ai_hr_manager_safe_v3")
    p.add_argument("--diagnostics-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/diagnostics/parent_year_ai_hr_manager_safe_v3")
    p.add_argument("--shuffle-partitions", type=int, default=600)
    p.add_argument("--coalesce", type=int, default=200)
    return p.parse_args()


def safe_divide(num, den):
    return F.when(den.isNull() | (den == 0), F.lit(None)).otherwise(num / den)


def first_existing(df, names):
    return [c for c in names if c in df.columns]


def write_parquet(df, path: str, coalesce: int | None = None):
    out = df
    if coalesce and coalesce > 0:
        out = out.coalesce(max(1, int(coalesce)))
    out.write.mode("overwrite").option("compression", "snappy").parquet(path)


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.diagnostics_dir).mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("augment_parent_year_with_ai_hr_manager_safe_v3")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    py = spark.read.parquet(args.parent_year_dir)
    fy = spark.read.parquet(args.firm_year_ai_dir)

    if "parent_rcid" not in fy.columns or "year" not in fy.columns:
        raise ValueError("firm-year AI panel must contain parent_rcid and year")
    if "parent_rcid" not in py.columns or "year" not in py.columns:
        raise ValueError("parent-year panel must contain parent_rcid and year")

    fy = fy.where(F.col("parent_rcid").isNotNull()).where(F.col("year").isNotNull())

    sum_cols = first_existing(fy, [
        "n_employees", "n_hr_positions", "n_managers",
        "ai_positions_title_strict_weighted", "ai_positions_description_strict_weighted", "ai_positions_any_strict_weighted",
        "ai_positions_title_broad_weighted", "ai_positions_description_broad_weighted", "ai_positions_any_broad_weighted",
        "ai_postings_title_strict", "ai_postings_description_strict", "ai_postings_any_strict",
        "ai_postings_title_broad", "ai_postings_description_broad", "ai_postings_any_broad",
        "posting_count_for_ai_flags",
    ])
    min_cols = first_existing(fy, [
        "first_ai_position_year_any_strict", "first_ai_posting_year_any_strict", "first_ai_firm_year_any_strict",
        "first_ai_position_year_any_broad", "first_ai_posting_year_any_broad", "first_ai_firm_year_any_broad",
        "first_ai_position_date_any_strict", "first_ai_posting_date_any_strict", "first_ai_firm_date_any_strict",
        "first_ai_position_date_any_broad", "first_ai_posting_date_any_broad", "first_ai_firm_date_any_broad",
    ])
    max_cols = first_existing(fy, [
        "has_ai_position_any_strict_by_year", "has_ai_posting_any_strict_by_year", "has_ai_firm_any_strict_by_year",
        "has_ai_position_any_broad_by_year", "has_ai_posting_any_broad_by_year", "has_ai_firm_any_broad_by_year",
        "is_first_ai_position_year_any_strict", "is_first_ai_posting_year_any_strict", "is_first_ai_firm_year_any_strict",
        "is_first_ai_position_year_any_broad", "is_first_ai_posting_year_any_broad", "is_first_ai_firm_year_any_broad",
    ])

    agg_exprs = []
    for c in sum_cols:
        agg_exprs.append(F.sum(F.coalesce(F.col(c).cast("double"), F.lit(0.0))).alias(c))
    for c in min_cols:
        agg_exprs.append(F.min(F.col(c)).alias(c))
    for c in max_cols:
        agg_exprs.append(F.max(F.coalesce(F.col(c).cast("double"), F.lit(0.0))).alias(c))

    if not agg_exprs:
        raise ValueError("No AI/HR/manager columns found in firm-year AI panel. Check the previous job output schema.")

    aug = fy.groupBy("parent_rcid", "year").agg(*agg_exprs)

    # Recompute shares and logs at the parent-year level.
    if "n_employees" in aug.columns:
        if "n_hr_positions" in aug.columns:
            aug = aug.withColumn("hr_to_employee_ratio", safe_divide(F.col("n_hr_positions"), F.col("n_employees")))
        if "n_managers" in aug.columns:
            aug = aug.withColumn("managers_to_employee_ratio", safe_divide(F.col("n_managers"), F.col("n_employees")))
        if "ai_positions_any_strict_weighted" in aug.columns:
            aug = aug.withColumn("ai_positions_any_strict_share", safe_divide(F.col("ai_positions_any_strict_weighted"), F.col("n_employees")))
        if "ai_positions_any_broad_weighted" in aug.columns:
            aug = aug.withColumn("ai_positions_any_broad_share", safe_divide(F.col("ai_positions_any_broad_weighted"), F.col("n_employees")))
    if "posting_count_for_ai_flags" in aug.columns:
        if "ai_postings_any_strict" in aug.columns:
            aug = aug.withColumn("ai_postings_any_strict_share", safe_divide(F.col("ai_postings_any_strict"), F.col("posting_count_for_ai_flags")))
        if "ai_postings_any_broad" in aug.columns:
            aug = aug.withColumn("ai_postings_any_broad_share", safe_divide(F.col("ai_postings_any_broad"), F.col("posting_count_for_ai_flags")))
    if "ai_positions_any_strict_weighted" in aug.columns:
        aug = aug.withColumn("ai_position_log1p", F.log1p(F.coalesce(F.col("ai_positions_any_strict_weighted"), F.lit(0.0))))
    if "ai_postings_any_strict" in aug.columns:
        aug = aug.withColumn("ai_posting_log1p", F.log1p(F.coalesce(F.col("ai_postings_any_strict"), F.lit(0.0))))

    # Drop any same-named augmented variables from the old parent-year panel so the join is unambiguous.
    drop_cols = [c for c in aug.columns if c not in ["parent_rcid", "year"] and c in py.columns]
    py_base = py.drop(*drop_cols) if drop_cols else py
    out = py_base.join(aug, on=["parent_rcid", "year"], how="left")

    write_parquet(out, args.out_dir, args.coalesce)

    written = spark.read.parquet(args.out_dir)
    meta = {
        "parent_year_dir": args.parent_year_dir,
        "firm_year_ai_dir": args.firm_year_ai_dir,
        "out_dir": args.out_dir,
        "rows": written.count(),
        "parents": written.select("parent_rcid").distinct().count(),
        "years": written.select("year").distinct().count(),
        "augmented_columns": [c for c in aug.columns if c not in ["parent_rcid", "year"]],
        "dropped_existing_columns_before_join": drop_cols,
    }
    with open(os.path.join(args.diagnostics_dir, "00_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)

    yearly_exprs = [F.count("*").alias("n_parent_year")]
    for c in ["ai_positions_any_strict_weighted", "ai_postings_any_strict", "n_hr_positions", "n_managers", "n_employees"]:
        if c in written.columns:
            yearly_exprs.append(F.sum(F.coalesce(F.col(c).cast("double"), F.lit(0.0))).alias(c))
    for c in ["hr_to_employee_ratio", "managers_to_employee_ratio", "ai_positions_any_strict_share", "ai_postings_any_strict_share"]:
        if c in written.columns:
            yearly_exprs.append(F.avg(F.col(c).cast("double")).alias(f"mean_{c}"))

    (written.groupBy("year").agg(*yearly_exprs).orderBy("year")
        .coalesce(1).write.mode("overwrite").option("header", True)
        .csv(os.path.join(args.diagnostics_dir, "01_yearly_summary_csv")))

    print(json.dumps(meta, indent=2, sort_keys=True, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
