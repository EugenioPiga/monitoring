#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from revelio_people_analytics_utils import (
    build_default_paths,
    create_spark,
    parquet_reader,
    safe_divide,
)

SKILL_DOMAIN_COLS = [
    "user_has_data_skill",
    "user_has_software_skill",
    "user_has_management_skill",
    "user_has_hr_skill",
    "user_has_sales_marketing_skill",
    "user_has_finance_skill",
    "user_has_operations_skill",
    "user_has_employee_feedback_tool_skill",
    "user_has_hr_technology_skill",
]

PARENT_KEEP_COLS = [
    "parent_rcid",
    "year",
    "firm_name",
    "naics_code",
    "is_public_company",
    "firm_age",
    "has_position_data",
    "has_posting_data",
    "workforce_weighted",
    "unique_sampled_users",
    "avg_salary",
    "hire_rate",
    "exit_rate",
    "female_share",
    "workers_with_data_skill_share",
    "workers_with_hr_skill_share",
    "workers_with_hr_technology_skill_share",
    "workers_with_employee_feedback_tool_skill_share",
    "hr_people_role_share",
    "data_analytics_role_share",
    "us_position_share",
    "people_analytics_postings_any_enriched",
    "people_analytics_postings_any_enriched_share",
    "people_analytics_postings_any_study",
    "people_analytics_postings_any_study_share",
    "first_people_analytics_posting_year_any_enriched",
    "is_first_people_analytics_posting_year_any_enriched",
    "has_people_analytics_posting_any_enriched_by_year",
]


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    p = argparse.ArgumentParser(description="Build Revelio-only parent-year first-pass panel.")
    p.add_argument("--project-root", default=defaults["project_root"])
    p.add_argument("--firm-year-dir", default=defaults["firm_year_output"])
    p.add_argument("--worker-year-dir", default=defaults["worker_year_output"])
    p.add_argument("--out-dir", default=os.path.join(defaults["processed_root"], "diagnostics", "parent_first_pass"))
    p.add_argument("--panel-out-dir", default=os.path.join(defaults["processed_root"], "final", "parent_year_first_pass"))
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "24")))
    p.add_argument("--shuffle-partitions", type=int, default=800)
    p.add_argument("--tmpdir", default=None)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--min-workforce", type=float, default=5.0)
    p.add_argument("--restrict-us", action="store_true")
    p.add_argument("--coalesce", type=int, default=80)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def write_parquet(df: DataFrame, path: str, coalesce: Optional[int] = None) -> None:
    writer = df
    if coalesce and coalesce > 0:
        writer = writer.coalesce(coalesce)
    writer.write.mode("overwrite").option("compression", "snappy").parquet(path)


def add_naics3(frame: DataFrame, source_col: str = "naics_code") -> DataFrame:
    return frame.withColumn(
        "naics3",
        F.when(F.col(source_col).isNull(), F.lit(None))
        .otherwise(F.substring(F.regexp_replace(F.col(source_col).cast("string"), r"[^0-9]", ""), 1, 3))
    )


def safe_weighted_mean(value: str, weight: str, alias: str) -> F.Column:
    return safe_divide(F.sum(F.col(value) * F.col(weight)), F.sum(F.col(weight))).alias(alias)


def build_parent_panel_from_firm_year(spark: SparkSession, args: argparse.Namespace) -> DataFrame:
    fy = parquet_reader(spark, args.firm_year_dir)
    keep = [c for c in PARENT_KEEP_COLS if c in fy.columns]
    df = fy.select(*keep).where(F.col("parent_rcid").isNotNull())

    if args.restrict_us and "us_position_share" in df.columns:
        df = df.where(F.coalesce(F.col("us_position_share"), F.lit(0.0)) > 0)

    agg_exprs = [
        F.countDistinct("firm_name").alias("distinct_firm_name_count"),
        F.sum(F.coalesce(F.col("workforce_weighted"), F.lit(0.0))).alias("workforce_weighted"),
        F.sum(F.coalesce(F.col("unique_sampled_users"), F.lit(0.0))).alias("unique_sampled_users_sum"),
        F.sum(F.coalesce(F.col("people_analytics_postings_any_enriched"), F.lit(0.0))).alias("people_analytics_postings_any_enriched"),
        F.sum(F.coalesce(F.col("people_analytics_postings_any_study"), F.lit(0.0))).alias("people_analytics_postings_any_study"),
        F.max(F.col("is_first_people_analytics_posting_year_any_enriched")).alias("is_first_people_analytics_posting_year_any_enriched"),
        F.max(F.col("has_people_analytics_posting_any_enriched_by_year")).alias("has_people_analytics_posting_any_enriched_by_year"),
        F.min(F.col("first_people_analytics_posting_year_any_enriched")).alias("first_people_analytics_posting_year_any_enriched"),
        F.max(F.col("is_public_company")).alias("is_public_company"),
        F.first("naics_code", ignorenulls=True).alias("naics_code"),
        F.first("firm_name", ignorenulls=True).alias("firm_name"),
        F.first("firm_age", ignorenulls=True).alias("firm_age"),
        F.max("has_position_data").alias("has_position_data"),
        F.max("has_posting_data").alias("has_posting_data"),
    ]

    if "avg_salary" in df.columns:
        agg_exprs.append(safe_weighted_mean("avg_salary", "workforce_weighted", "avg_salary"))
    if "hire_rate" in df.columns:
        agg_exprs.append(safe_weighted_mean("hire_rate", "workforce_weighted", "hire_rate"))
    if "exit_rate" in df.columns:
        agg_exprs.append(safe_weighted_mean("exit_rate", "workforce_weighted", "exit_rate"))
    if "people_analytics_postings_any_enriched_share" in df.columns:
        agg_exprs.append(
            safe_weighted_mean("people_analytics_postings_any_enriched_share", "workforce_weighted", "people_analytics_postings_any_enriched_share")
        )
    if "people_analytics_postings_any_study_share" in df.columns:
        agg_exprs.append(
            safe_weighted_mean("people_analytics_postings_any_study_share", "workforce_weighted", "people_analytics_postings_any_study_share")
        )

    for extra in [
        "female_share",
        "workers_with_data_skill_share",
        "workers_with_hr_skill_share",
        "workers_with_hr_technology_skill_share",
        "workers_with_employee_feedback_tool_skill_share",
        "hr_people_role_share",
        "data_analytics_role_share",
        "us_position_share",
    ]:
        if extra in df.columns:
            agg_exprs.append(safe_weighted_mean(extra, "workforce_weighted", extra))

    parent = df.groupBy("parent_rcid", "year").agg(*agg_exprs)
    parent = parent.withColumn("pa_posting_log1p", F.log1p(F.coalesce(F.col("people_analytics_postings_any_enriched"), F.lit(0.0))))
    parent = add_naics3(parent)
    parent = parent.withColumn(
        "analysis_sample",
        F.when(
            (F.col("workforce_weighted") >= F.lit(float(args.min_workforce)))
            & (F.col("has_posting_data") == 1)
            & F.col("naics3").isNotNull(),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    return parent


def build_parent_skill_dispersion(spark: SparkSession, args: argparse.Namespace) -> DataFrame:
    wy = parquet_reader(spark, args.worker_year_dir)

    needed = ["year", "primary_parent_rcid", "worker_weight_in_firm_year", "user_distinct_skills"] + [
        c for c in SKILL_DOMAIN_COLS if c in wy.columns
    ]
    needed = [c for c in needed if c in wy.columns]

    df = wy.select(*needed).where(F.col("primary_parent_rcid").isNotNull())

    if "worker_weight_in_firm_year" not in df.columns:
        df = df.withColumn("worker_weight_in_firm_year", F.lit(1.0))

    df = df.withColumn(
        "weight",
        F.greatest(F.coalesce(F.col("worker_weight_in_firm_year").cast("double"), F.lit(1.0)), F.lit(0.0)),
    )
    df = df.withColumn("user_distinct_skills", F.col("user_distinct_skills").cast("double"))

    for col in SKILL_DOMAIN_COLS:
        if col in df.columns:
            df = df.withColumn(col, F.coalesce(F.col(col).cast("double"), F.lit(0.0)))

    means = (
        df.groupBy(F.col("primary_parent_rcid").alias("parent_rcid"), "year")
        .agg(
            safe_divide(F.sum(F.col("weight") * F.col("user_distinct_skills")), F.sum(F.col("weight"))).alias("mean_skill"),
            F.sum("weight").alias("worker_weight_total"),
            safe_divide(F.sum(F.col("weight") * F.col("user_distinct_skills")), F.sum(F.col("weight"))).alias("avg_distinct_skills_worker"),
            *[
                safe_divide(F.sum(F.col("weight") * F.col(col)), F.sum(F.col("weight"))).alias(f"mean_{col}")
                for col in SKILL_DOMAIN_COLS if col in df.columns
            ],
        )
    )

    df2 = (
        df.alias("d")
        .join(
            means.select("parent_rcid", "year", "mean_skill").alias("m"),
            (F.col("d.primary_parent_rcid") == F.col("m.parent_rcid")) & (F.col("d.year") == F.col("m.year")),
            how="left",
        )
        .select("d.*", F.col("m.mean_skill"))
    )

    df2 = df2.withColumn("sqdev", (F.col("user_distinct_skills") - F.col("mean_skill")) ** 2)

    skill_var = (
        df2.groupBy(F.col("primary_parent_rcid").alias("parent_rcid"), "year")
        .agg(
            safe_divide(F.sum(F.col("weight") * F.col("sqdev")), F.sum(F.col("weight"))).alias("skill_count_var")
        )
    )

    out = means.join(skill_var, on=["parent_rcid", "year"], how="left")
    out = out.withColumn("skill_count_sd", F.sqrt(F.col("skill_count_var")))

    bundle_expr = None
    for col in SKILL_DOMAIN_COLS:
        mean_col = f"mean_{col}"
        if mean_col in out.columns:
            this_var = F.col(mean_col) * (F.lit(1.0) - F.col(mean_col))
            bundle_expr = this_var if bundle_expr is None else (bundle_expr + this_var)

    out = out.withColumn(
        "skill_bundle_dispersion",
        bundle_expr if bundle_expr is not None else F.lit(None),
    )

    return out.select(
        "parent_rcid",
        "year",
        "worker_weight_total",
        "skill_count_var",
        "skill_count_sd",
        "skill_bundle_dispersion",
        "avg_distinct_skills_worker",
    )


def build_final_parent_panel(spark: SparkSession, args: argparse.Namespace) -> DataFrame:
    parent_panel = build_parent_panel_from_firm_year(spark, args)
    skill_panel = build_parent_skill_dispersion(spark, args)

    # Write/reload to break lineage and reduce memory pressure
    tmp_parent = args.panel_out_dir + "_tmp_parent"
    tmp_skill = args.panel_out_dir + "_tmp_skill"

    write_parquet(parent_panel, tmp_parent, coalesce=40)
    write_parquet(skill_panel, tmp_skill, coalesce=40)

    parent_panel = spark.read.parquet(tmp_parent)
    skill_panel = spark.read.parquet(tmp_skill)

    panel = parent_panel.join(skill_panel, on=["parent_rcid", "year"], how="left")

    panel = panel.withColumn(
        "ever_adopter_posting",
        F.when(F.col("first_people_analytics_posting_year_any_enriched").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
    )

    panel = panel.withColumn(
        "event_time_posting",
        F.when(
            F.col("first_people_analytics_posting_year_any_enriched").isNotNull(),
            F.col("year") - F.col("first_people_analytics_posting_year_any_enriched"),
        ).otherwise(F.lit(None)),
    )

    panel = panel.withColumn("log_workforce", F.log(F.greatest(F.col("workforce_weighted"), F.lit(1.0))))
    panel = panel.withColumn("log_avg_salary", F.when(F.col("avg_salary") > 0, F.log(F.col("avg_salary"))).otherwise(F.lit(None)))

    parent_window = Window.partitionBy("parent_rcid").orderBy("year")
    for col in ["workforce_weighted", "avg_salary", "hire_rate", "exit_rate", "skill_count_sd", "skill_bundle_dispersion"]:
        if col in panel.columns:
            panel = panel.withColumn(f"L1_{col}", F.lag(col, 1).over(parent_window))
            panel = panel.withColumn(f"F5_{col}", F.lead(col, 5).over(parent_window))

    panel = panel.withColumn(
        "d5_log_workforce",
        F.when(
            (F.col("workforce_weighted") > 0) & (F.col("F5_workforce_weighted") > 0),
            100.0 * (F.log(F.col("F5_workforce_weighted")) - F.log(F.col("workforce_weighted"))),
        ).otherwise(F.lit(None)),
    )

    panel = panel.withColumn(
        "d5_log_avg_salary",
        F.when(
            (F.col("avg_salary") > 0) & (F.col("F5_avg_salary") > 0),
            100.0 * (F.log(F.col("F5_avg_salary")) - F.log(F.col("avg_salary"))),
        ).otherwise(F.lit(None)),
    )

    panel = panel.withColumn("d5_hire_rate", 100.0 * (F.col("F5_hire_rate") - F.col("hire_rate")))
    panel = panel.withColumn("d5_exit_rate", 100.0 * (F.col("F5_exit_rate") - F.col("exit_rate")))
    panel = panel.withColumn("d5_skill_count_sd", 100.0 * (F.col("F5_skill_count_sd") - F.col("skill_count_sd")))
    panel = panel.withColumn("d5_skill_bundle_dispersion", 100.0 * (F.col("F5_skill_bundle_dispersion") - F.col("skill_bundle_dispersion")))

    panel = panel.where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year))
    return panel


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.panel_out_dir)

    spark = create_spark(
        app_name="revelio_parent_first_pass_panel_only",
        threads=args.threads,
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    panel = build_final_parent_panel(spark, args)
    write_parquet(panel, args.panel_out_dir, coalesce=args.coalesce)

    written = spark.read.parquet(args.panel_out_dir)

    meta = {
        "panel_out_dir": args.panel_out_dir,
        "n_parent_year_rows": written.count(),
        "n_parent_firms": written.select("parent_rcid").distinct().count(),
        "n_analysis_sample_rows": written.where(F.col("analysis_sample") == 1).count(),
        "n_adopter_parents": written.where(F.col("ever_adopter_posting") == 1).select("parent_rcid").distinct().count(),
        "years": [r[0] for r in written.select("year").distinct().orderBy("year").collect()],
    }
    save_json(meta, os.path.join(args.out_dir, "00_metadata.json"))

    yearly = (
        written.groupBy("year")
        .agg(
            F.countDistinct("parent_rcid").alias("n_parents"),
            F.sum("analysis_sample").alias("n_analysis_sample"),
            F.avg("pa_posting_log1p").alias("mean_pa_posting_log1p"),
            F.avg("has_people_analytics_posting_any_enriched_by_year").alias("share_adopted_by_year"),
            F.sum("is_first_people_analytics_posting_year_any_enriched").alias("first_adoption_count"),
            F.avg("workforce_weighted").alias("mean_workforce"),
            F.avg("skill_count_sd").alias("mean_skill_count_sd"),
            F.avg("skill_bundle_dispersion").alias("mean_skill_bundle_dispersion"),
        )
        .orderBy("year")
    )

    yearly.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(args.out_dir, "01_yearly_summary_csv"))

    sample = (
        written.where(F.col("analysis_sample") == 1)
        .select(
            "parent_rcid",
            "year",
            "firm_name",
            "naics3",
            "pa_posting_log1p",
            "people_analytics_postings_any_enriched",
            "people_analytics_postings_any_enriched_share",
            "is_first_people_analytics_posting_year_any_enriched",
            "has_people_analytics_posting_any_enriched_by_year",
            "workforce_weighted",
            "avg_salary",
            "hire_rate",
            "exit_rate",
            "skill_count_sd",
            "skill_bundle_dispersion",
            "analysis_sample",
        )
        .limit(5000)
    )

    sample.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(args.out_dir, "02_sample_head_csv"))

    spark.stop()


if __name__ == "__main__":
    main()
