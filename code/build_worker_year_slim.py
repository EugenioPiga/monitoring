#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pyspark.sql import functions as F
from pyspark.sql import Window

from strategy_utils import create_spark, ensure_dir, save_json, write_parquet, get_existing

def parse_args():
    p = argparse.ArgumentParser(description="Build slim worker-year analysis panel from huge worker-year panel.")
    p.add_argument("--project-root", default="/labs/khanna/predictive_capital/revelio_people_analytics")
    p.add_argument("--worker-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/worker_year_panel")
    p.add_argument("--parent-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/worker_year_slim")
    p.add_argument("--diagnostics-dir", default=None)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "24")))
    p.add_argument("--shuffle-partitions", type=int, default=1200)
    p.add_argument("--coalesce", type=int, default=400)
    p.add_argument("--tmpdir", default=None)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--analysis-sample-only", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    diag_dir = args.diagnostics_dir or os.path.join(args.project_root, "processed", "diagnostics", "worker_year_slim")
    ensure_dir(args.out_dir)
    ensure_dir(diag_dir)

    spark = create_spark("build_worker_year_slim", args.threads, args.shuffle_partitions, args.tmpdir)
    wy = spark.read.parquet(args.worker_year_dir)
    py = spark.read.parquet(args.parent_year_dir)

    cols = [
        "user_id", "year",
        "primary_parent_rcid", "primary_firm_key", "primary_firm_name",
        "primary_role_k150", "primary_role_k50", "primary_mapped_role", "primary_job_category",
        "primary_state", "primary_country", "primary_metro_area",
        "avg_salary", "avg_seniority", "salary_coverage_share",
        "hire_rate", "exit_rate", "hires_weighted", "exits_weighted",
        "worker_weight_in_firm_year", "multi_firm_worker",
        "user_highest_degree", "user_highest_degree_score",
        "user_has_bachelor_plus", "user_has_advanced_degree",
        "user_has_stem_education", "user_has_business_education",
        "user_distinct_skills", "user_predicted_skill_share",
        "user_has_data_skill", "user_has_software_skill", "user_has_management_skill",
        "user_has_hr_skill", "user_has_sales_marketing_skill", "user_has_finance_skill",
        "user_has_operations_skill", "user_has_employee_feedback_tool_skill", "user_has_hr_technology_skill",
        "employer_workforce_weighted", "employer_avg_salary",
        "employer_people_analytics_postings_any_enriched",
        "employer_first_people_analytics_posting_year_any_enriched",
        "employer_has_people_analytics_firm_any_enriched_by_year",
    ]
    df = (
        wy.select(*get_existing(wy, cols))
        .where(F.col("primary_parent_rcid").isNotNull())
        .where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year))
        .withColumnRenamed("primary_parent_rcid", "parent_rcid")
    )

    if "primary_role_k150" in df.columns:
        df = df.withColumn("occupation", F.col("primary_role_k150"))
    elif "primary_mapped_role" in df.columns:
        df = df.withColumn("occupation", F.col("primary_mapped_role"))
    else:
        df = df.withColumn("occupation", F.col("primary_job_category"))

    py_keep = get_existing(py, [
        "parent_rcid", "year", "naics3", "firm_name", "analysis_sample",
        "pa_posting_log1p", "first_people_analytics_posting_year_any_enriched",
        "has_people_analytics_posting_any_enriched_by_year",
        "is_first_people_analytics_posting_year_any_enriched",
    ])
    df = df.join(py.select(*py_keep).dropDuplicates(["parent_rcid", "year"]), on=["parent_rcid", "year"], how="left")

    if args.analysis_sample_only and "analysis_sample" in df.columns:
        df = df.where(F.col("analysis_sample") == 1)

    w = Window.partitionBy("user_id").orderBy("year")
    df = df.withColumn("next_parent_rcid", F.lead("parent_rcid").over(w))
    df = df.withColumn("next_occupation", F.lead("occupation").over(w))
    df = df.withColumn("next_salary", F.lead("avg_salary").over(w))
    df = df.withColumn("prev_parent_rcid", F.lag("parent_rcid").over(w))
    df = df.withColumn("prev_occupation", F.lag("occupation").over(w))

    df = df.withColumn("external_move_next", F.when(F.col("next_parent_rcid").isNotNull() & (F.col("next_parent_rcid") != F.col("parent_rcid")), F.lit(1)).otherwise(F.lit(0)))
    df = df.withColumn("internal_occ_move_next", F.when(F.col("next_parent_rcid") == F.col("parent_rcid"), F.when(F.col("next_occupation") != F.col("occupation"), F.lit(1)).otherwise(F.lit(0))).otherwise(F.lit(0)))
    df = df.withColumn("salary_growth_next", F.when((F.col("avg_salary") > 0) & (F.col("next_salary") > 0), 100.0 * (F.log(F.col("next_salary")) - F.log(F.col("avg_salary")))))

    write_parquet(df, args.out_dir, args.coalesce)

    written = spark.read.parquet(args.out_dir)
    meta = {
        "out_dir": args.out_dir,
        "rows": written.count(),
        "workers": written.select("user_id").distinct().count(),
        "parents": written.select("parent_rcid").distinct().count(),
        "analysis_sample_only": args.analysis_sample_only,
    }
    save_json(meta, os.path.join(diag_dir, "00_worker_year_slim_metadata.json"))

    (
        written.groupBy("year")
        .agg(
            F.count("*").alias("n_worker_years"),
            F.countDistinct("user_id").alias("n_workers"),
            F.countDistinct("parent_rcid").alias("n_parents"),
            F.avg("external_move_next").alias("mean_external_move_next"),
            F.avg("internal_occ_move_next").alias("mean_internal_occ_move_next"),
            F.avg("salary_growth_next").alias("mean_salary_growth_next"),
        )
        .orderBy("year")
        .coalesce(1).write.mode("overwrite").option("header", True)
        .csv(os.path.join(diag_dir, "01_yearly_summary_csv"))
    )

    print("[INFO] Done worker-year slim")
    print(meta)
    spark.stop()

if __name__ == "__main__":
    main()
