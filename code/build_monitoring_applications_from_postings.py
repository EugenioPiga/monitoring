#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pyspark.sql import functions as F
from monitoring_exposure_utils import create_spark, ensure_dir, save_json, write_parquet

APPLICATION_PATTERNS = {
    "attrition_retention_prediction": r"(attrition|turnover|retention|flight risk|churn).{0,80}(predict|model|forecast|analytic|dashboard|risk)|(?:predict|model|forecast).{0,80}(attrition|turnover|retention|flight risk|churn)",
    "performance_dashboards": r"(performance|productivity|kpi|metric|goal|okr).{0,80}(dashboard|scorecard|report|analytics|measurement|tracking)|(?:dashboard|scorecard).{0,80}(performance|productivity|kpi|metric|goal|okr)",
    "employee_engagement_listening": r"(employee engagement|engagement survey|employee listening|pulse survey|sentiment|experience survey|voice of employee|culture analytics)",
    "workforce_planning_forecasting": r"(workforce planning|headcount planning|capacity planning|labor planning|staffing forecast|workforce forecast|demand forecast|scheduling optimization)",
    "productivity_monitoring": r"(employee monitoring|productivity monitoring|activity tracking|digital activity|work monitoring|time tracking|workflow tracking|workforce productivity)",
    "performance_review_management": r"(performance review|performance management|talent review|calibration|review cycle|360 feedback|continuous feedback)",
    "promotion_succession_talent": r"(promotion|succession|talent marketplace|internal mobility|career path|high potential|hipo).{0,80}(analytic|model|dashboard|prediction|planning)",
    "compliance_quality_monitoring": r"(compliance|quality assurance|quality monitoring|call quality|audit).{0,80}(employee|agent|workforce|performance|dashboard|monitoring|analytics)|(?:monitor|track).{0,80}(compliance|quality)",
    "organizational_network_analysis": r"(organizational network analysis|people network|collaboration analytics|network analytics|relationship analytics)",
    "learning_skills_analytics": r"(skill analytics|skills intelligence|learning analytics|training analytics|capability analytics|competency analytics)",
}
BROAD_PA_PATTERN = r"(people analytics|workforce analytics|hr analytics|human resources analytics|talent analytics|employee analytics|human capital analytics|organizational analytics|workforce intelligence|employee monitoring|productivity analytics|performance analytics|employee engagement analytics|workforce planning|attrition prediction|retention analytics|turnover prediction)"

def parse_args():
    p = argparse.ArgumentParser(description="Extract monitoring application categories from Revelio postings.")
    p.add_argument("--postings-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/intermediate/postings_extracted")
    p.add_argument("--parent-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_applications_parent_year")
    p.add_argument("--diagnostics-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/diagnostics/monitoring_applications_parent_year")
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--shuffle-partitions", type=int, default=800)
    p.add_argument("--coalesce", type=int, default=120)
    p.add_argument("--tmpdir", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out_dir); ensure_dir(args.diagnostics_dir)
    spark = create_spark("build_monitoring_applications_from_postings", args.shuffle_partitions, args.tmpdir)
    postings = spark.read.parquet(args.postings_dir)
    py = spark.read.parquet(args.parent_year_dir)

    df = (postings
        .select(F.col("job_id"), F.col("ultimate_parent_rcid").cast("string").alias("parent_rcid"),
                F.to_date(F.col("post_date")).alias("post_date"), F.col("role_k150").alias("posting_role_k150"),
                F.col("jobtitle_raw"), F.col("jobtitle"), F.col("jobtitle_translated"), F.col("description"))
        .where(F.col("parent_rcid").isNotNull()).where(F.col("post_date").isNotNull())
        .withColumn("year", F.year("post_date"))
        .where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year)))
    text = F.lower(F.concat_ws(" ", F.coalesce(F.col("jobtitle_raw"), F.lit("")), F.coalesce(F.col("jobtitle"), F.lit("")),
                             F.coalesce(F.col("jobtitle_translated"), F.lit("")), F.coalesce(F.col("description"), F.lit(""))))
    df = df.withColumn("posting_text", text).withColumn("broad_pa_flag", F.when(F.col("posting_text").rlike(BROAD_PA_PATTERN), F.lit(1)).otherwise(F.lit(0)))
    for app, pattern in APPLICATION_PATTERNS.items():
        df = df.withColumn(f"flag_{app}", F.when(F.col("posting_text").rlike(pattern), F.lit(1)).otherwise(F.lit(0)))
    flag_cols = [f"flag_{a}" for a in APPLICATION_PATTERNS]
    any_concrete = sum([F.col(c) for c in flag_cols])
    df = df.withColumn("any_concrete_monitoring_app", F.when(any_concrete > 0, F.lit(1)).otherwise(F.lit(0)))

    app_array = F.array(*[F.when(F.col(f"flag_{app}") == 1, F.lit(app)).otherwise(F.lit(None)) for app in APPLICATION_PATTERNS])
    apps = (df.where((F.col("broad_pa_flag") == 1) | (F.col("any_concrete_monitoring_app") == 1))
        .withColumn("application_category", F.explode(app_array))
        .where(F.col("application_category").isNotNull())
        .groupBy("parent_rcid", "year", "application_category")
        .agg(F.countDistinct("job_id").alias("application_posting_count")))
    parent_totals = (df.groupBy("parent_rcid", "year")
        .agg(F.countDistinct("job_id").alias("all_posting_count"),
             F.sum("broad_pa_flag").alias("broad_pa_posting_count"),
             F.sum("any_concrete_monitoring_app").alias("concrete_monitoring_posting_count")))
    apps = (apps.join(parent_totals, ["parent_rcid", "year"], "left")
        .withColumn("application_share_all_postings", F.col("application_posting_count") / F.col("all_posting_count"))
        .withColumn("application_share_pa_postings", F.col("application_posting_count") / F.col("broad_pa_posting_count"))
        .withColumn("application_log_count", F.log1p(F.col("application_posting_count"))))
    py_keep = [c for c in ["parent_rcid", "year", "analysis_sample", "naics3", "pa_posting_log1p", "first_people_analytics_posting_year_any_enriched"] if c in py.columns]
    if py_keep:
        apps = apps.join(py.select(*py_keep).dropDuplicates(["parent_rcid", "year"]), ["parent_rcid", "year"], "left")
    write_parquet(apps, args.out_dir, args.coalesce)
    written = spark.read.parquet(args.out_dir)
    meta = {"out_dir": args.out_dir, "rows": written.count(), "parents": written.select("parent_rcid").distinct().count(), "application_categories": list(APPLICATION_PATTERNS.keys())}
    save_json(meta, os.path.join(args.diagnostics_dir, "00_metadata.json"))
    (written.groupBy("year", "application_category")
        .agg(F.countDistinct("parent_rcid").alias("n_parents"), F.sum("application_posting_count").alias("application_posting_count"))
        .orderBy("year", "application_category")
        .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(args.diagnostics_dir, "01_application_summary_csv")))
    print(meta, flush=True)
    spark.stop()

if __name__ == "__main__":
    main()
