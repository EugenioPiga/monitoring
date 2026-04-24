#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pyspark.sql import functions as F

from strategy_utils import create_spark, ensure_dir, save_json, write_parquet, safe_divide


def parse_args():
    p = argparse.ArgumentParser(
        description="Build parent_rcid × occupation × year posting panel from available Revelio postings."
    )
    p.add_argument("--project-root", default="/labs/khanna/predictive_capital/revelio_people_analytics")
    p.add_argument("--postings-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/intermediate/postings_extracted")
    p.add_argument("--parent-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/posting_parent_occ_year")
    p.add_argument("--diagnostics-dir", default=None)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "24")))
    p.add_argument("--shuffle-partitions", type=int, default=1200)
    p.add_argument("--coalesce", type=int, default=200)
    p.add_argument("--tmpdir", default=None)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2023)
    return p.parse_args()


def pa_text_flag(text_col):
    """
    Conservative people-analytics / workforce-monitoring text flag.
    This is deliberately broad but still tied to worker analytics / HR analytics.
    """
    pattern = (
        r"(people analytics|workforce analytics|hr analytics|human resources analytics|"
        r"talent analytics|employee analytics|employee engagement analytics|"
        r"workforce planning|workforce intelligence|organizational analytics|"
        r"performance analytics|performance dashboard|employee performance|"
        r"attrition prediction|retention analytics|turnover prediction|"
        r"employee monitoring|productivity monitoring|productivity analytics|"
        r"workforce management system|human capital analytics|"
        r"succession analytics|promotion analytics|employee listening)"
    )
    return F.when(F.lower(F.coalesce(text_col, F.lit(""))).rlike(pattern), F.lit(1)).otherwise(F.lit(0))


def main():
    args = parse_args()
    diag_dir = args.diagnostics_dir or os.path.join(
        args.project_root, "processed", "diagnostics", "posting_parent_occ_year"
    )
    ensure_dir(args.out_dir)
    ensure_dir(diag_dir)

    spark = create_spark(
        "build_posting_parent_occ_year",
        args.threads,
        args.shuffle_partitions,
        args.tmpdir,
    )

    postings = spark.read.parquet(args.postings_dir)
    py = spark.read.parquet(args.parent_year_dir)

    # Core available columns from the actual postings schema.
    df = (
        postings
        .select(
            F.col("job_id"),
            F.col("ultimate_parent_rcid").cast("string").alias("parent_rcid"),
            F.coalesce(F.col("role_k150"), F.col("role_k50"), F.col("mapped_role"), F.col("job_category")).alias("occupation"),
            F.col("job_category"),
            F.col("salary").cast("double").alias("posting_salary"),
            F.to_date(F.col("post_date")).alias("post_date"),
            F.to_date(F.col("remove_date")).alias("remove_date"),
            F.col("company"),
            F.col("ultimate_parent_company_name").alias("parent_name"),
            F.col("jobtitle_raw"),
            F.col("jobtitle"),
            F.col("jobtitle_translated"),
            F.col("description"),
            F.col("country"),
            F.col("state"),
            F.col("city"),
            F.col("metro_area"),
        )
        .where(F.col("parent_rcid").isNotNull())
        .where(F.col("occupation").isNotNull())
        .where(F.col("post_date").isNotNull())
        .withColumn("year", F.year("post_date"))
        .where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year))
    )

    text = F.concat_ws(
        " ",
        F.coalesce(F.col("jobtitle_raw"), F.lit("")),
        F.coalesce(F.col("jobtitle"), F.lit("")),
        F.coalesce(F.col("jobtitle_translated"), F.lit("")),
        F.coalesce(F.col("description"), F.lit("")),
    )

    df = df.withColumn("posting_text", text)
    df = df.withColumn("pa_posting_flag", pa_text_flag(F.col("posting_text")))
    df = df.withColumn(
        "has_salary",
        F.when(F.col("posting_salary").isNotNull() & (F.col("posting_salary") > 0), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "posting_duration_days",
        F.when(
            F.col("remove_date").isNotNull(),
            F.datediff(F.col("remove_date"), F.col("post_date")),
        ),
    )

    panel = (
        df.groupBy("parent_rcid", "occupation", "year")
        .agg(
            F.countDistinct("job_id").alias("posting_count"),
            F.sum("pa_posting_flag").alias("pa_posting_count"),
            F.avg("pa_posting_flag").alias("pa_posting_share"),
            F.avg("posting_salary").alias("avg_posting_salary"),
            F.avg("has_salary").alias("salary_coverage_share"),
            F.avg("posting_duration_days").alias("avg_posting_duration_days"),
            F.countDistinct("job_category").alias("n_job_categories"),
            F.countDistinct("metro_area").alias("n_metros"),
            F.first("parent_name", ignorenulls=True).alias("parent_name"),
        )
        .withColumn("log_posting_count", F.log1p(F.col("posting_count")))
        .withColumn("log_pa_posting_count", F.log1p(F.col("pa_posting_count")))
    )

    # Attach parent-year adoption context.
    py_keep = [
        c for c in [
            "parent_rcid",
            "year",
            "pa_posting_log1p",
            "people_analytics_postings_any_enriched",
            "people_analytics_postings_any_enriched_share",
            "first_people_analytics_posting_year_any_enriched",
            "has_people_analytics_posting_any_enriched_by_year",
            "is_first_people_analytics_posting_year_any_enriched",
            "naics3",
            "analysis_sample",
            "workforce_weighted",
        ]
        if c in py.columns
    ]

    panel = panel.join(
        py.select(*py_keep).dropDuplicates(["parent_rcid", "year"]),
        on=["parent_rcid", "year"],
        how="left",
    )

    panel = panel.withColumn(
        "event_time_posting",
        F.when(
            F.col("first_people_analytics_posting_year_any_enriched").isNotNull(),
            F.col("year") - F.col("first_people_analytics_posting_year_any_enriched"),
        ),
    )

    panel = panel.withColumn(
        "posting_occ_analysis_sample",
        F.when(
            (F.col("analysis_sample") == 1)
            & F.col("naics3").isNotNull()
            & (F.col("posting_count") > 0),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    write_parquet(panel, args.out_dir, args.coalesce)

    written = spark.read.parquet(args.out_dir)
    meta = {
        "out_dir": args.out_dir,
        "rows": written.count(),
        "parents": written.select("parent_rcid").distinct().count(),
        "occupations": written.select("occupation").distinct().count(),
        "analysis_rows": written.where(F.col("posting_occ_analysis_sample") == 1).count(),
        "note": "This is a posting occupation-year panel. The source postings file has no skill column, so this is not a posting-skill panel.",
    }
    save_json(meta, os.path.join(diag_dir, "00_posting_parent_occ_year_metadata.json"))

    by_year = (
        written.where(F.col("posting_occ_analysis_sample") == 1)
        .groupBy("year")
        .agg(
            F.count("*").alias("n_parent_occ_year"),
            F.countDistinct("parent_rcid").alias("n_parents"),
            F.countDistinct("occupation").alias("n_occupations"),
            F.sum("posting_count").alias("posting_count"),
            F.sum("pa_posting_count").alias("pa_posting_count"),
            F.avg("pa_posting_share").alias("mean_pa_posting_share"),
            F.avg("avg_posting_salary").alias("mean_posting_salary"),
        )
        .orderBy("year")
    )

    by_year.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(diag_dir, "01_yearly_summary_csv")
    )

    print("[INFO] Done posting parent-occupation-year panel")
    print(meta)
    spark.stop()


if __name__ == "__main__":
    main()
