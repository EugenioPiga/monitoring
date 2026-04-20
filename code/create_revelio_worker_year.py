#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

from pyspark.sql import Window
from pyspark.sql import functions as F

from revelio_people_analytics_utils import (
    DEFAULT_RAW_PATHS,
    build_company_lookup,
    build_default_paths,
    build_user_features,
    choose_primary_position,
    create_spark,
    detect_analysis_end_year,
    ensure_directory,
    parquet_reader,
    prefix_columns,
    prepare_positions,
    prepare_weighted_columns,
    add_weighted_means,
    expand_positions_to_years,
    safe_divide,
)


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Build Revelio worker-year panel with employer context.")
    parser.add_argument("--project-root", default=defaults["project_root"])
    parser.add_argument("--company-ref-dir", default=DEFAULT_RAW_PATHS["company_ref"])
    parser.add_argument("--education-dir", default=DEFAULT_RAW_PATHS["education"])
    parser.add_argument("--positions-dir", default=DEFAULT_RAW_PATHS["position"])
    parser.add_argument("--skills-dir", default=DEFAULT_RAW_PATHS["skill"])
    parser.add_argument("--users-dir", default=DEFAULT_RAW_PATHS["user"])
    parser.add_argument("--firm-year-dir", default=defaults["firm_year_output"])
    parser.add_argument("--intermediate-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    parser.add_argument("--shuffle-partitions", type=int, default=1200)
    parser.add_argument("--coalesce", type=int, default=200)
    parser.add_argument("--tmpdir", default=None)
    return parser.parse_args()


def write_parquet(df, path: str, coalesce: int | None = None) -> None:
    writer = df
    if coalesce is not None and coalesce > 0:
        writer = writer.coalesce(max(1, coalesce))
    (
        writer.write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(path)
    )

def main() -> None:
    args = parse_args()
    paths = build_default_paths(args.project_root)
    intermediate_dir = args.intermediate_dir or paths["worker_year_intermediate"]
    out_dir = args.out_dir or paths["worker_year_output"]

    for path in [args.project_root, paths["processed_root"], paths["intermediate_root"], paths["final_root"], intermediate_dir, out_dir]:
        ensure_directory(path)

    spark = create_spark(
        app_name="revelio_people_analytics_worker_year",
        threads=args.threads,
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    print("[1/7] Detecting analysis horizon")
    analysis_end_year = detect_analysis_end_year(args.positions_dir, None, args.users_dir, spark)
    print(f"[INFO] Analysis end year: {analysis_end_year}")

    print("[2/7] Building company and user features")
    entity_lookup, _ = build_company_lookup(spark, args.company_ref_dir)
    user_features = build_user_features(spark, args.users_dir, args.education_dir, args.skills_dir)

    print("[3/7] Preparing and expanding positions")
    positions = prepare_positions(spark, args.positions_dir, user_features, entity_lookup, analysis_end_year)
    expanded = expand_positions_to_years(positions)

    print("[4/7] Aggregating worker-year positions")
    weighted_columns = [
        "remote_suitability",
        "salary",
        "start_mean_sampled_salary",
        "end_mean_sampled_salary",
        "seniority",
        "is_us_position",
        "is_data_analytics_role",
        "is_hr_people_role",
        "people_analytics_title_study",
        "people_analytics_description_study",
        "people_analytics_any_study",
        "people_analytics_title_enriched",
        "people_analytics_description_enriched",
        "people_analytics_any_enriched",
        "start_year_imputed",
        "end_year_imputed",
        "both_dates_missing",
        "is_hire_year_observed",
        "is_exit_year_observed",
        "is_censored_current_year",
        "jobcat_admin",
        "jobcat_engineer",
        "jobcat_sales",
        "jobcat_scientist",
        "jobcat_marketing",
        "jobcat_finance",
        "jobcat_operations",
        "salary_observed",
    ]
    expanded = prepare_weighted_columns(expanded, weighted_columns)

    group_exprs = [
        F.count(F.lit(1)).alias("active_position_records"),
        F.countDistinct("firm_key").alias("active_firms_count"),
        F.countDistinct("parent_rcid").alias("active_parent_firms_count"),
        F.countDistinct("state").alias("active_states_count"),
        F.countDistinct("metro_area").alias("active_metros_count"),
        F.countDistinct("country").alias("active_countries_count"),
        F.sum("weight").alias("worker_portfolio_weight"),
    ]
    for column_name in weighted_columns:
        group_exprs.append(F.sum(f"w_{column_name}").alias(f"w_{column_name}"))
        group_exprs.append(F.sum(f"wobs_{column_name}").alias(f"wobs_{column_name}"))

    worker_year = expanded.groupBy("user_id", "year").agg(*group_exprs)
    rename_map = {
        "remote_suitability": "avg_remote_suitability",
        "salary": "avg_salary",
        "start_mean_sampled_salary": "avg_start_salary",
        "end_mean_sampled_salary": "avg_end_salary",
        "seniority": "avg_seniority",
        "is_us_position": "us_position_share",
        "is_data_analytics_role": "data_analytics_role_share",
        "is_hr_people_role": "hr_people_role_share",
        "people_analytics_title_study": "worker_people_analytics_title_study_share",
        "people_analytics_description_study": "worker_people_analytics_description_study_share",
        "people_analytics_any_study": "worker_people_analytics_any_study_share",
        "people_analytics_title_enriched": "worker_people_analytics_title_enriched_share",
        "people_analytics_description_enriched": "worker_people_analytics_description_enriched_share",
        "people_analytics_any_enriched": "worker_people_analytics_any_enriched_share",
        "start_year_imputed": "share_start_year_imputed",
        "end_year_imputed": "share_end_year_imputed",
        "both_dates_missing": "share_both_dates_missing",
        "is_censored_current_year": "share_censored_current_positions",
        "jobcat_admin": "admin_role_share",
        "jobcat_engineer": "engineer_role_share",
        "jobcat_sales": "sales_role_share",
        "jobcat_scientist": "scientist_role_share",
        "jobcat_marketing": "marketing_role_share",
        "jobcat_finance": "finance_role_share",
        "jobcat_operations": "operations_role_share",
        "salary_observed": "salary_coverage_share",
    }
    worker_year = add_weighted_means(worker_year, weighted_columns, rename_map)
    worker_year = (
        worker_year
        .withColumn("hires_weighted", F.col("w_is_hire_year_observed"))
        .withColumn("exits_weighted", F.col("w_is_exit_year_observed"))
        .withColumn("hire_rate", safe_divide(F.col("w_is_hire_year_observed"), F.col("worker_portfolio_weight")))
        .withColumn("exit_rate", safe_divide(F.col("w_is_exit_year_observed"), F.col("worker_portfolio_weight")))
        .withColumn("worker_people_analytics_title_study_weighted", F.col("w_people_analytics_title_study"))
        .withColumn("worker_people_analytics_description_study_weighted", F.col("w_people_analytics_description_study"))
        .withColumn("worker_people_analytics_any_study_weighted", F.col("w_people_analytics_any_study"))
        .withColumn("worker_people_analytics_title_enriched_weighted", F.col("w_people_analytics_title_enriched"))
        .withColumn("worker_people_analytics_description_enriched_weighted", F.col("w_people_analytics_description_enriched"))
        .withColumn("worker_people_analytics_any_enriched_weighted", F.col("w_people_analytics_any_enriched"))
    )
    worker_year = worker_year.drop(*([f"w_{c}" for c in weighted_columns] + [f"wobs_{c}" for c in weighted_columns]))

    print("[5/7] Identifying primary employer-year and primary role")
    user_firm_year = (
        expanded.groupBy("user_id", "year", "firm_key")
        .agg(
            F.first("firm_name", ignorenulls=True).alias("primary_firm_name"),
            F.first("parent_rcid", ignorenulls=True).alias("primary_parent_rcid"),
            F.sum("weight").alias("worker_weight_in_firm_year"),
            F.countDistinct("position_id").alias("worker_positions_in_firm_year"),
            F.max("start_date").alias("latest_start_date_in_firm"),
        )
    )
    firm_order = Window.partitionBy("user_id", "year").orderBy(
        F.col("worker_weight_in_firm_year").desc(),
        F.col("worker_positions_in_firm_year").desc(),
        F.col("latest_start_date_in_firm").desc_nulls_last(),
        F.col("firm_key").asc(),
    )
    primary_firm = (
        user_firm_year
        .withColumn("firm_rank", F.row_number().over(firm_order))
        .where(F.col("firm_rank") == 1)
        .drop("firm_rank")
    )

    primary_position = (
        choose_primary_position(expanded)
        .select(
            "user_id",
            "year",
            F.col("position_id").alias("primary_position_id"),
            F.col("jobtitle_raw").alias("primary_jobtitle_raw"),
            F.col("mapped_role").alias("primary_mapped_role"),
            F.col("role_k50").alias("primary_role_k50"),
            F.col("role_k150").alias("primary_role_k150"),
            F.col("job_category").alias("primary_job_category"),
            F.col("state").alias("primary_state"),
            F.col("country").alias("primary_country"),
            F.col("metro_area").alias("primary_metro_area"),
            F.col("company_name").alias("primary_company_name"),
            F.col("firm_key").alias("primary_firm_key_from_position"),
        )
    )

    worker_year = worker_year.join(primary_firm, on=["user_id", "year"], how="left").join(primary_position, on=["user_id", "year"], how="left")
    worker_year = worker_year.withColumn("primary_firm_key", F.coalesce(F.col("firm_key"), F.col("primary_firm_key_from_position")))
    worker_year = worker_year.drop("firm_key", "primary_firm_key_from_position")
    worker_year = worker_year.withColumn("multi_firm_worker", F.when(F.col("active_firms_count") > 1, F.lit(1)).otherwise(F.lit(0)))

    print("[6/7] Attaching worker attributes and employer context")
    user_columns = [
        "user_id",
        "firstname",
        "lastname",
        "fullname",
        "sex_predicted",
        "ethnicity_predicted",
        "user_location",
        "country",
        "title",
        "summary",
        "profile_snapshot_date",
        "profile_snapshot_year",
        "f_prob",
        "m_prob",
        "white_prob",
        "black_prob",
        "api_prob",
        "hispanic_prob",
        "native_prob",
        "multiple_prob",
        "prestige",
        "numconnections",
        "connections_log1p",
        "highest_degree",
        "highest_degree_score",
        "has_bachelor_plus",
        "has_advanced_degree",
        "education_records",
        "has_top500_school",
        "has_stem_education",
        "has_business_education",
        "has_law_education",
        "has_health_education",
        "distinct_skills",
        "predicted_skill_share",
        "has_data_skill",
        "has_software_skill",
        "has_management_skill",
        "has_hr_skill",
        "has_sales_marketing_skill",
        "has_finance_skill",
        "has_operations_skill",
        "has_employee_feedback_tool_skill",
        "has_hr_technology_skill",
    ]
    user_context = prefix_columns(user_features.select(*[c for c in user_columns if c in user_features.columns]), "user_", ["user_id"])
    worker_year = worker_year.join(user_context, on="user_id", how="left")

    employer_keep = [
        "firm_key",
        "year",
        "firm_name",
        "parent_rcid",
        "workforce_weighted",
        "unique_sampled_users",
        "hires_weighted",
        "exits_weighted",
        "hire_rate",
        "exit_rate",
        "avg_salary",
        "female_share",
        "workers_with_data_skill_share",
        "workers_with_hr_skill_share",
        "hr_people_role_share",
        "data_analytics_role_share",
        "people_analytics_positions_any_study_weighted",
        "people_analytics_positions_any_enriched_weighted",
        "people_analytics_postings_any_study",
        "people_analytics_postings_any_enriched",
        "first_people_analytics_firm_year_any_study",
        "first_people_analytics_firm_year_any_enriched",
        "first_people_analytics_position_year_any_study",
        "first_people_analytics_position_year_any_enriched",
        "first_people_analytics_posting_year_any_study",
        "first_people_analytics_posting_year_any_enriched",
        "has_people_analytics_firm_any_enriched_by_year",
    ]

    firm_year_base = parquet_reader(spark, args.firm_year_dir)
    employer_context = prefix_columns(
        firm_year_base.select(*[c for c in employer_keep if c in firm_year_base.columns]),
        "employer_",
        ["firm_key", "year"],
    )
    worker_year = worker_year.join(
        employer_context,
        (worker_year["primary_firm_key"] == employer_context["firm_key"]) & (worker_year["year"] == employer_context["year"]),
        how="left",
    ).drop(employer_context["firm_key"]).drop(employer_context["year"])
    worker_year = worker_year.withColumn(
        "employer_sampled_coworkers",
        F.when(F.col("employer_unique_sampled_users").isNotNull(), F.greatest(F.col("employer_unique_sampled_users") - 1, F.lit(0))).otherwise(F.lit(None)),
    )
    worker_year = worker_year.withColumn(
        "employer_weighted_coworkers_approx",
        F.when(
            F.col("employer_workforce_weighted").isNotNull() & F.col("worker_weight_in_firm_year").isNotNull(),
            F.greatest(F.col("employer_workforce_weighted") - F.col("worker_weight_in_firm_year"), F.lit(0.0)),
        ).otherwise(F.lit(None)),
    )

    print("[7/7] Writing final worker-year panel only")
    write_parquet(worker_year, out_dir, args.coalesce)

    print("[INFO] Quick checks from written output")
    written = spark.read.parquet(out_dir)
    rows = written.count()
    workers = written.select("user_id").distinct().count()
    exposed = (
        written.where(F.col("employer_has_people_analytics_firm_any_enriched_by_year") == 1).count()
        if "employer_has_people_analytics_firm_any_enriched_by_year" in written.columns
        else 0
    )
    print(f"[INFO] Worker-year rows: {rows:,}")
    print(f"[INFO] Distinct workers: {workers:,}")
    print(f"[INFO] Worker-year observations exposed to adopted employer context: {exposed:,}")

    spark.stop()


if __name__ == "__main__":
    main()
