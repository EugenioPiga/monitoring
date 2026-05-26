#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np
from pyspark.sql import functions as F

from monitoring_exposure_utils import create_spark, ensure_dir, save_json, write_parquet
from build_monitoring_exposure_parent_occ_year import build_app_task, build_crosswalk

OUTCOME_COLS = [
    "d5_log_workers", "d5_exit_rate", "d5_hire_rate", "d5_skill_count_sd", "d5_skill_bundle_dispersion",
    "d5_skill_hhi_mean", "d5_specialist_share", "exit_rate", "hire_rate", "skill_count_sd",
    "skill_bundle_dispersion", "skill_hhi_mean", "specialist_share",
    "hr_to_employee_ratio", "managers_to_employee_ratio", "n_hr_positions", "n_managers",
    "n_promotions", "promotion_rate", "promotion_rate_continuers", "n_continuing_workers",
    "avg_salary", "log_avg_salary", "F5_avg_salary", "L1_avg_salary", "d5_log_avg_salary",
]

PARENT_YEAR_KEEP = [
    "parent_rcid", "year", "people_analytics_postings_any_enriched", "pa_posting_log1p",
    "ai_postings_any_strict", "ai_positions_any_strict_weighted", "ai_posting_log1p", "ai_position_log1p",
    "n_employees", "n_hr_positions", "n_managers", "hr_to_employee_ratio", "managers_to_employee_ratio",
]


def parse_args():
    p = argparse.ArgumentParser(description="Build parent-occupation-year exposure with static internal/external visibility and log-inside formula.")
    p.add_argument("--parent-occ-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_occupation_year_panel")
    p.add_argument("--parent-year-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass")
    p.add_argument("--applications-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_applications_parent_year")
    p.add_argument("--onet-task-weights-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_task_weights")
    p.add_argument("--visibility-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_task_visibility_static")
    p.add_argument("--onet-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_30_2_text")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_exposure_parent_occ_year")
    p.add_argument("--diagnostics-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/diagnostics/monitoring_exposure_parent_occ_year")
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--similarity-threshold-quantile", type=float, default=0.95)
    p.add_argument("--min-similarity", type=float, default=0.05)
    p.add_argument("--shuffle-partitions", type=int, default=1000)
    p.add_argument("--coalesce", type=int, default=250)
    p.add_argument("--tmpdir", default=None)
    return p.parse_args()


def existing(df, cols):
    return [c for c in cols if c in df.columns]


def first_existing_count_expr(candidates: list[str]) -> F.Column:
    expr = None
    for c in candidates:
        term = F.col(c).cast("double") if c.endswith("_log1p") is False else F.expm1(F.col(c).cast("double"))
        expr = term if expr is None else F.coalesce(expr, term)
    return F.coalesce(expr, F.lit(0.0)) if expr is not None else F.lit(0.0)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.diagnostics_dir)
    spark = create_spark("build_monitoring_exposure_parent_occ_year_v2", args.shuffle_partitions, args.tmpdir)

    parent_occ = spark.read.parquet(args.parent_occ_dir)
    apps = spark.read.parquet(args.applications_dir)
    onet_tasks = spark.read.parquet(args.onet_task_weights_dir)
    visibility = spark.read.parquet(args.visibility_dir)

    keep = ["parent_rcid", "occupation", "year", "n_workers", "pa_posting_log1p", "event_time_posting"] + [c for c in OUTCOME_COLS if c in parent_occ.columns]
    poc = (
        parent_occ.where(F.col("occupation_analysis_sample") == 1)
        .where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year))
        .select(*existing(parent_occ, keep))
        .cache()
    )
    if "avg_salary" in poc.columns and "log_avg_salary" not in poc.columns:
        poc = poc.withColumn(
            "log_avg_salary",
            F.when(F.col("avg_salary") > 0, F.log(F.col("avg_salary"))).otherwise(F.lit(None))
        )

    _ = poc.count()

    # Add parent-year adoption intensities if the parent-occupation panel does not already carry them.
    if os.path.exists(args.parent_year_dir):
        py = spark.read.parquet(args.parent_year_dir)
        py_keep = existing(py, PARENT_YEAR_KEEP)
        if "parent_rcid" in py_keep and "year" in py_keep:
            py_small = py.select(*py_keep).dropDuplicates(["parent_rcid", "year"])
            for c in py_small.columns:
                if c in poc.columns and c not in ["parent_rcid", "year"]:
                    py_small = py_small.withColumnRenamed(c, f"py_{c}")
            poc = poc.join(py_small, on=["parent_rcid", "year"], how="left")

    # Recover PA and AI adoption counts. Preference order: explicit counts, then log1p variables.
    pa_candidates = [c for c in ["people_analytics_postings_any_enriched", "py_people_analytics_postings_any_enriched", "pa_posting_log1p", "py_pa_posting_log1p"] if c in poc.columns]
    ai_candidates = [c for c in ["ai_postings_any_strict", "py_ai_postings_any_strict", "ai_positions_any_strict_weighted", "py_ai_positions_any_strict_weighted", "ai_posting_log1p", "py_ai_posting_log1p", "ai_position_log1p", "py_ai_position_log1p"] if c in poc.columns]
    poc = poc.withColumn("pa_adoption_count_for_exposure", first_existing_count_expr(pa_candidates))
    poc = poc.withColumn("ai_adoption_count_for_exposure", first_existing_count_expr(ai_candidates))

    onet_pdf = onet_tasks.select("onet_soc_code", "onet_title", "onet_description", "task_id", "task_text", "task_weight", "task_importance").toPandas()
    onet_pdf["task_id"] = onet_pdf["task_id"].astype(str)
    task_unique = onet_pdf[["task_id", "task_text"]].drop_duplicates("task_id")
    sim_pdf, tau = build_app_task(task_unique, args.similarity_threshold_quantile, args.min_similarity)
    if sim_pdf.empty:
        raise RuntimeError("No app-task matches. Lower threshold.")
    sim_sdf = spark.createDataFrame(sim_pdf)
    write_parquet(sim_sdf, os.path.join(args.diagnostics_dir, "02_application_task_similarity"), 1)

    occ_pdf = poc.select("occupation").distinct().toPandas()
    cw_pdf = build_crosswalk(occ_pdf, onet_pdf[["onet_soc_code", "onet_title", "onet_description"]].drop_duplicates("onet_soc_code"), args.onet_dir)
    cw_pdf.to_csv(os.path.join(args.diagnostics_dir, "01_revelio_occupation_to_onet_crosswalk.csv"), index=False)
    cw = spark.createDataFrame(cw_pdf)

    app_counts = (
        apps.where((F.col("year") >= args.start_year) & (F.col("year") <= args.end_year))
        .groupBy("parent_rcid", "year", "application_category")
        .agg(F.sum("application_posting_count").alias("application_posting_count"))
    )
    total = (
        app_counts.groupBy("parent_rcid", "year")
        .agg(F.sum("application_posting_count").alias("monitoring_application_count"))
        .withColumn("monitoring_application_log1p", F.log1p("monitoring_application_count"))
    )

    task_py = (
        app_counts.join(sim_sdf, "application_category", "inner")
        .groupBy("parent_rcid", "year", "task_id")
        .agg(
            F.sum(F.col("application_posting_count") * F.col("task_exposed")).alias("task_exposed_weighted_count"),
            F.sum(F.col("application_posting_count") * F.col("monitoring_task_similarity")).alias("task_similarity_weighted_sum"),
        )
        .join(total, ["parent_rcid", "year"], "left")
        .withColumn("xi_task_parent_year", F.col("task_exposed_weighted_count") / F.col("monitoring_application_count"))
        .withColumn("xi_similarity_parent_year", F.col("task_similarity_weighted_sum") / F.col("monitoring_application_count"))
    )

    weights = onet_tasks.select("onet_soc_code", F.col("task_id").cast("string").alias("task_id"), "task_weight")
    vis = visibility.select(
        F.col("task_id").cast("string").alias("task_id"),
        "visibility_internal_static",
        "visibility_external_static",
        "visibility_internal_static_z",
        "visibility_external_static_z",
    ).dropDuplicates(["task_id"])

    po_tasks = poc.join(cw, "occupation", "left").join(weights, "onet_soc_code", "left").join(vis, "task_id", "left")
    joined = (
        po_tasks.join(
            task_py.select("parent_rcid", "year", "task_id", "xi_task_parent_year", "xi_similarity_parent_year", "monitoring_application_count", "monitoring_application_log1p"),
            ["parent_rcid", "year", "task_id"],
            "left",
        )
        .fillna({
            "xi_task_parent_year": 0.0,
            "xi_similarity_parent_year": 0.0,
            "monitoring_application_count": 0.0,
            "monitoring_application_log1p": 0.0,
            "task_weight": 0.0,
            "visibility_internal_static": 0.0,
            "visibility_external_static": 0.0,
            "visibility_internal_static_z": 0.0,
            "visibility_external_static_z": 0.0,
            "pa_adoption_count_for_exposure": 0.0,
            "ai_adoption_count_for_exposure": 0.0,
        })
    )

    base_first = [
        F.first("onet_soc_code", ignorenulls=True).alias("onet_soc_code"),
        F.first("onet_title", ignorenulls=True).alias("onet_title"),
        F.first("occupation_onet_similarity", ignorenulls=True).alias("occupation_onet_similarity"),
        F.first("crosswalk_method", ignorenulls=True).alias("crosswalk_method"),
        F.first("n_workers", ignorenulls=True).alias("n_workers"),
        F.first("pa_posting_log1p", ignorenulls=True).alias("pa_posting_log1p") if "pa_posting_log1p" in joined.columns else F.lit(None).cast("double").alias("pa_posting_log1p"),
        F.first("event_time_posting", ignorenulls=True).alias("event_time_posting") if "event_time_posting" in joined.columns else F.lit(None).cast("double").alias("event_time_posting"),
        F.first("pa_adoption_count_for_exposure", ignorenulls=True).alias("pa_adoption_count_for_exposure"),
        F.first("ai_adoption_count_for_exposure", ignorenulls=True).alias("ai_adoption_count_for_exposure"),
        F.first("monitoring_application_count", ignorenulls=True).alias("monitoring_application_count"),
        F.first("monitoring_application_log1p", ignorenulls=True).alias("monitoring_application_log1p"),
    ]
    outcome_first = [F.first(c, ignorenulls=True).alias(c) for c in OUTCOME_COLS if c in joined.columns]
    exposure_aggs = [
        # Existing monitoring exposure formula preserved.
        F.sum(F.col("task_weight") * F.col("xi_task_parent_year")).alias("monitoring_exposure_average_raw"),
        F.sum(F.col("task_weight") * F.col("xi_similarity_parent_year")).alias("monitoring_similarity_average_raw"),
        # Static visibility levels.
        F.sum(F.col("task_weight") * F.col("visibility_internal_static")).alias("occ_visibility_internal_static"),
        F.sum(F.col("task_weight") * F.col("visibility_external_static")).alias("occ_visibility_external_static"),
        # Old outside-log formula: log(1+N_pt) * sum_j w_oj visibility_j.
        F.sum(F.col("task_weight") * F.log1p(F.col("pa_adoption_count_for_exposure")) * F.col("visibility_internal_static")).alias("pa_visibility_internal_oldformula"),
        F.sum(F.col("task_weight") * F.log1p(F.col("pa_adoption_count_for_exposure")) * F.col("visibility_external_static")).alias("pa_visibility_external_oldformula"),
        F.sum(F.col("task_weight") * F.log1p(F.col("ai_adoption_count_for_exposure")) * F.col("visibility_internal_static")).alias("ai_visibility_internal_oldformula"),
        F.sum(F.col("task_weight") * F.log1p(F.col("ai_adoption_count_for_exposure")) * F.col("visibility_external_static")).alias("ai_visibility_external_oldformula"),
        # New requested formula: sum_j w_oj log(1 + N_pt * visibility_j).
        F.sum(F.col("task_weight") * F.log1p(F.col("pa_adoption_count_for_exposure") * F.col("visibility_internal_static"))).alias("pa_visibility_internal_loginside"),
        F.sum(F.col("task_weight") * F.log1p(F.col("pa_adoption_count_for_exposure") * F.col("visibility_external_static"))).alias("pa_visibility_external_loginside"),
        F.sum(F.col("task_weight") * F.log1p(F.col("ai_adoption_count_for_exposure") * F.col("visibility_internal_static"))).alias("ai_visibility_internal_loginside"),
        F.sum(F.col("task_weight") * F.log1p(F.col("ai_adoption_count_for_exposure") * F.col("visibility_external_static"))).alias("ai_visibility_external_loginside"),
    ]

    avg = joined.groupBy("parent_rcid", "occupation", "year").agg(*(base_first + outcome_first + exposure_aggs))
    joined2 = joined.join(avg.select("parent_rcid", "occupation", "year", "monitoring_exposure_average_raw"), ["parent_rcid", "occupation", "year"], "left") \
        .withColumn("weighted_sq_dev", F.col("task_weight") * (F.col("xi_task_parent_year") - F.col("monitoring_exposure_average_raw")) ** 2)
    conc = joined2.groupBy("parent_rcid", "occupation", "year").agg(F.sum("weighted_sq_dev").alias("monitoring_exposure_concentration_raw"))

    final = (
        avg.join(conc, ["parent_rcid", "occupation", "year"], "left")
        .withColumn("monitoring_exposure_average", F.col("monitoring_exposure_average_raw") * F.col("monitoring_application_log1p"))
        .withColumn("monitoring_exposure_concentration", F.col("monitoring_exposure_concentration_raw") * F.col("monitoring_application_log1p"))
        .withColumn("monitoring_similarity_average", F.col("monitoring_similarity_average_raw") * F.col("monitoring_application_log1p"))
        .withColumn("log_n_workers", F.when(F.col("n_workers") > 0, F.log("n_workers")))
    )

    write_parquet(final, args.out_dir, args.coalesce)
    written = spark.read.parquet(args.out_dir)
    meta = {
        "out_dir": args.out_dir,
        "rows": written.count(),
        "parents": written.select("parent_rcid").distinct().count(),
        "occupations": written.select("occupation").distinct().count(),
        "similarity_threshold": float(tau),
        "new_formula": "sum_j w_oj * log(1 + N_pt * visibility_j_static)",
        "visibility_dir": args.visibility_dir,
    }
    save_json(meta, os.path.join(args.diagnostics_dir, "00_metadata.json"))

    (
        written.groupBy("year")
        .agg(
            F.count("*").alias("n_parent_occ_year"),
            F.countDistinct("parent_rcid").alias("n_parents"),
            F.avg("monitoring_exposure_average").alias("mean_monitoring_exposure_average"),
            F.avg("pa_visibility_internal_loginside").alias("mean_pa_visibility_internal_loginside"),
            F.avg("pa_visibility_external_loginside").alias("mean_pa_visibility_external_loginside"),
            F.avg("ai_visibility_internal_loginside").alias("mean_ai_visibility_internal_loginside"),
            F.avg("ai_visibility_external_loginside").alias("mean_ai_visibility_external_loginside"),
        )
        .orderBy("year")
        .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(args.diagnostics_dir, "03_yearly_summary_csv"))
    )
    print(meta, flush=True)
    poc.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
