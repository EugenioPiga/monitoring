#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pyspark.sql import functions as F

from strategy_utils import (
    create_spark, ensure_dir, save_json, write_parquet,
    safe_divide, weighted_mean, add_naics3, add_forward_outcomes, get_existing
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

def parse_args():
    p = argparse.ArgumentParser(description="Build parent_rcid x occupation x year panel.")
    p.add_argument("--project-root", default="/labs/khanna/predictive_capital/revelio_people_analytics")
    p.add_argument("--worker-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/worker_year_panel")
    p.add_argument("--parent-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_occupation_year_panel")
    p.add_argument("--diagnostics-dir", default=None)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "24")))
    p.add_argument("--shuffle-partitions", type=int, default=1200)
    p.add_argument("--coalesce", type=int, default=200)
    p.add_argument("--tmpdir", default=None)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--occupation-col", default="primary_role_k150")
    p.add_argument("--min-cell-workers", type=float, default=5.0)
    return p.parse_args()

def main():
    args = parse_args()
    diag_dir = args.diagnostics_dir or os.path.join(args.project_root, "processed", "diagnostics", "parent_occupation_year")
    ensure_dir(args.out_dir)
    ensure_dir(diag_dir)
    spark = create_spark("build_parent_occupation_year", args.threads, args.shuffle_partitions, args.tmpdir)

    wy = spark.read.parquet(args.worker_year_dir)
    py = spark.read.parquet(args.parent_year_dir)

    occ_col = args.occupation_col if args.occupation_col in wy.columns else (
        "primary_mapped_role" if "primary_mapped_role" in wy.columns else "primary_job_category"
    )

    needed = [
        "user_id", "year", "primary_parent_rcid", occ_col,
        "worker_weight_in_firm_year", "avg_salary", "hires_weighted", "exits_weighted",
        "avg_seniority", "user_distinct_skills", "user_predicted_skill_share",
    ] + [c for c in SKILL_DOMAIN_COLS if c in wy.columns]

    df = (
        wy.select(*get_existing(wy, needed))
        .where(F.col("primary_parent_rcid").isNotNull())
        .where(F.col(occ_col).isNotNull())
        .where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year))
        .withColumnRenamed("primary_parent_rcid", "parent_rcid")
        .withColumnRenamed(occ_col, "occupation")
    )

    if "worker_weight_in_firm_year" not in df.columns:
        df = df.withColumn("worker_weight_in_firm_year", F.lit(1.0))
    df = df.withColumn("weight", F.greatest(F.coalesce(F.col("worker_weight_in_firm_year").cast("double"), F.lit(1.0)), F.lit(0.0)))

    for c in ["user_distinct_skills", "user_predicted_skill_share", "avg_salary", "avg_seniority"]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))

    skill_cols = [c for c in SKILL_DOMAIN_COLS if c in df.columns]
    for c in skill_cols:
        df = df.withColumn(c, F.coalesce(F.col(c).cast("double"), F.lit(0.0)))

    if skill_cols:
        total = None
        sq = None
        for c in skill_cols:
            total = F.col(c) if total is None else total + F.col(c)
            sq = F.col(c) * F.col(c) if sq is None else sq + F.col(c) * F.col(c)
        df = df.withColumn("skill_domain_count", total)
        df = df.withColumn("skill_hhi_worker", F.when(F.col("skill_domain_count") > 0, sq / (F.col("skill_domain_count") * F.col("skill_domain_count"))))
        df = df.withColumn("specialist_worker", F.when(F.col("skill_hhi_worker") >= 0.50, F.lit(1.0)).otherwise(F.lit(0.0)))
    else:
        df = df.withColumn("skill_hhi_worker", F.lit(None).cast("double"))
        df = df.withColumn("specialist_worker", F.lit(None).cast("double"))

    agg = [
        F.sum("weight").alias("n_workers"),
        F.countDistinct("user_id").alias("n_unique_workers"),
        weighted_mean("skill_hhi_worker", "weight", "skill_hhi_mean"),
        weighted_mean("specialist_worker", "weight", "specialist_share"),
    ]
    if "avg_salary" in df.columns:
        agg.append(weighted_mean("avg_salary", "weight", "avg_salary"))
    if "avg_seniority" in df.columns:
        agg.append(weighted_mean("avg_seniority", "weight", "avg_seniority"))
    if "user_distinct_skills" in df.columns:
        agg.append(weighted_mean("user_distinct_skills", "weight", "avg_distinct_skills"))
    if "hires_weighted" in df.columns:
        agg.append(F.sum(F.coalesce(F.col("hires_weighted"), F.lit(0.0))).alias("hires_weighted"))
    else:
        agg.append(F.lit(None).cast("double").alias("hires_weighted"))
    if "exits_weighted" in df.columns:
        agg.append(F.sum(F.coalesce(F.col("exits_weighted"), F.lit(0.0))).alias("exits_weighted"))
    else:
        agg.append(F.lit(None).cast("double").alias("exits_weighted"))

    panel = df.groupBy("parent_rcid", "occupation", "year").agg(*agg)
    panel = panel.withColumn("hire_rate", safe_divide(F.col("hires_weighted"), F.col("n_workers")))
    panel = panel.withColumn("exit_rate", safe_divide(F.col("exits_weighted"), F.col("n_workers")))

    if "user_distinct_skills" in df.columns:
        mean = panel.select("parent_rcid", "occupation", "year", F.col("avg_distinct_skills").alias("mean_skill_count"))
        df2 = df.join(mean, on=["parent_rcid", "occupation", "year"], how="left")
        df2 = df2.withColumn("sqdev_skill_count", (F.col("user_distinct_skills") - F.col("mean_skill_count")) ** 2)
        skill_var = (
            df2.groupBy("parent_rcid", "occupation", "year")
            .agg(safe_divide(F.sum(F.col("weight") * F.col("sqdev_skill_count")), F.sum("weight")).alias("skill_count_var"))
            .withColumn("skill_count_sd", F.sqrt(F.col("skill_count_var")))
        )
        panel = panel.join(skill_var, on=["parent_rcid", "occupation", "year"], how="left")

    for c in skill_cols:
        tmp = df.groupBy("parent_rcid", "occupation", "year").agg(weighted_mean(c, "weight", f"mean_{c}"))
        panel = panel.join(tmp, on=["parent_rcid", "occupation", "year"], how="left")

    bundle = None
    for c in skill_cols:
        mc = f"mean_{c}"
        if mc in panel.columns:
            this = F.col(mc) * (F.lit(1.0) - F.col(mc))
            bundle = this if bundle is None else bundle + this
    panel = panel.withColumn("skill_bundle_dispersion", bundle if bundle is not None else F.lit(None).cast("double"))

    py_keep = get_existing(py, [
        "parent_rcid", "year", "firm_name", "naics3", "naics_code", "workforce_weighted",
        "pa_posting_log1p", "people_analytics_postings_any_enriched",
        "people_analytics_postings_any_enriched_share",
        "first_people_analytics_posting_year_any_enriched",
        "has_people_analytics_posting_any_enriched_by_year",
        "is_first_people_analytics_posting_year_any_enriched",
        "analysis_sample",
    ])
    panel = panel.join(py.select(*py_keep).dropDuplicates(["parent_rcid", "year"]), on=["parent_rcid", "year"], how="left")
    panel = add_naics3(panel, "naics_code" if "naics_code" in panel.columns else "naics3")
    panel = panel.withColumn("event_time_posting", F.when(F.col("first_people_analytics_posting_year_any_enriched").isNotNull(), F.col("year") - F.col("first_people_analytics_posting_year_any_enriched")))

    panel = panel.withColumn(
        "occupation_analysis_sample",
        F.when((F.col("n_workers") >= F.lit(args.min_cell_workers)) & (F.col("analysis_sample") == 1) & F.col("naics3").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )

    panel = add_forward_outcomes(panel, ["parent_rcid", "occupation"], "year")
    write_parquet(panel, args.out_dir, args.coalesce)

    written = spark.read.parquet(args.out_dir)
    meta = {
        "out_dir": args.out_dir,
        "rows": written.count(),
        "parents": written.select("parent_rcid").distinct().count(),
        "occupations": written.select("occupation").distinct().count(),
        "analysis_rows": written.where(F.col("occupation_analysis_sample") == 1).count(),
        "occupation_col_used": occ_col,
    }
    save_json(meta, os.path.join(diag_dir, "00_parent_occupation_year_metadata.json"))

    (
        written.where(F.col("occupation_analysis_sample") == 1)
        .groupBy("year")
        .agg(
            F.count("*").alias("n_cells"),
            F.countDistinct("parent_rcid").alias("n_parents"),
            F.countDistinct("occupation").alias("n_occupations"),
            F.avg("pa_posting_log1p").alias("mean_pa_posting_log1p"),
            F.avg("n_workers").alias("mean_workers_per_cell"),
            F.avg("exit_rate").alias("mean_exit_rate"),
            F.avg("skill_bundle_dispersion").alias("mean_skill_bundle_dispersion"),
        )
        .orderBy("year")
        .coalesce(1).write.mode("overwrite").option("header", True)
        .csv(os.path.join(diag_dir, "01_yearly_summary_csv"))
    )

    print("[INFO] Done parent-occupation-year panel")
    print(meta)
    spark.stop()

if __name__ == "__main__":
    main()
