#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"

import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pyspark.sql import Window
from pyspark.sql import functions as F

from utils.revelio_analysis_utils import (
    build_analysis_paths,
    create_spark,
    ensure_analysis_directories,
    ensure_directory,
    load_json,
    setup_logging,
    write_pandas_csv,
)
from utils.revelio_event_study_design import outcome_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create summary tables for the Revelio event-study sample.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    parser.add_argument("--tmpdir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root)
    ensure_analysis_directories(paths)

    sample_path = Path(args.sample_path) if args.sample_path else paths.samples_root / "revelio_event_study_sample.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else paths.tables_root / "event_study_summary"
    ensure_directory(output_dir)
    logger = setup_logging("04_make_summary_tables", paths.logs_root)

    spark = create_spark(
        "revelio_event_study_summary_tables",
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    logger.info("Reading sample from %s", sample_path)
    df = spark.read.parquet(str(sample_path))
    outcomes = outcome_frame(config)
    baseline_outcomes = outcomes[outcomes["group"].isin(["primary", "composition", "wages"])].head(10)

    adoption_by_year = (
        df.groupBy("year")
        .agg(
            F.sum(F.when(F.col("is_first_people_analytics_firm_year_any_enriched") == 1, 1).otherwise(0)).alias("main_adoptions"),
            F.sum(F.when(F.col("is_first_people_analytics_position_year_any_enriched") == 1, 1).otherwise(0)).alias("position_adoptions"),
            F.sum(F.when(F.col("is_first_people_analytics_posting_year_any_enriched") == 1, 1).otherwise(0)).alias("posting_adoptions"),
            F.countDistinct("firm_key").alias("distinct_firms"),
        )
        .orderBy("year")
        .toPandas()
    )
    write_pandas_csv(adoption_by_year, output_dir / "adoption_by_year.csv")

    industry_adoption = (
        df.where(F.col("is_first_people_analytics_firm_year_any_enriched") == 1)
        .groupBy("naics2")
        .agg(
            F.countDistinct("firm_key").alias("adopting_firms"),
            F.min("year").alias("first_adoption_year"),
            F.max("year").alias("last_adoption_year"),
        )
        .orderBy(F.col("adopting_firms").desc(), F.col("naics2").asc_nulls_last())
        .toPandas()
    )
    write_pandas_csv(industry_adoption, output_dir / "adoption_by_industry.csv")

    support_tables = []
    for treatment in ["main", "position", "posting"]:
        event_col = f"{treatment}_event_time_binned"
        analysis_col = f"{treatment}_analysis_row"
        treated_col = f"{treatment}_balanced_treated"
        if event_col not in df.columns:
            continue
        support = (
            df.where((F.col(analysis_col) == 1) & (F.col(treated_col) == 1))
            .groupBy(F.col(event_col).alias("event_time"))
            .agg(
                F.count(F.lit(1)).alias("treated_rows"),
                F.countDistinct("firm_key").alias("treated_firms"),
            )
            .orderBy("event_time")
            .toPandas()
        )
        support["treatment_name"] = treatment
        support_tables.append(support)
    if support_tables:
        write_pandas_csv(pd.concat(support_tables, ignore_index=True), output_dir / "event_time_support.csv")

    around_adoption = (
        df.where((F.col("main_analysis_row") == 1) & (F.col("main_balanced_treated") == 1))
        .where(F.col("main_event_time_binned").between(-3, 3))
        .groupBy("main_event_time_binned")
        .agg(
            F.avg("exit_rate").alias("avg_exit_rate"),
            F.avg("hire_rate").alias("avg_hire_rate"),
            F.avg("log_workforce").alias("avg_log_workforce"),
            F.avg("avg_seniority").alias("avg_avg_seniority"),
            F.avg("data_analytics_role_share").alias("avg_data_analytics_role_share"),
            F.avg("hr_people_role_share").alias("avg_hr_people_role_share"),
            F.avg("avg_salary").alias("avg_avg_salary"),
        )
        .orderBy("main_event_time_binned")
        .toPandas()
        .rename(columns={"main_event_time_binned": "event_time"})
    )
    write_pandas_csv(around_adoption, output_dir / "average_outcomes_around_adoption.csv")

    treated_baseline = (
        df.where((F.col("main_analysis_row") == 1) & (F.col("main_balanced_treated") == 1))
        .where(F.col("main_event_time_raw").between(-3, -1))
        .groupBy("firm_key")
        .agg(*[F.avg(F.col(row["name"])).alias(row["name"]) for _, row in baseline_outcomes.iterrows()])
        .withColumn("group", F.lit("treated"))
    )

    first_year_window = Window.partitionBy("firm_key").orderBy("year")
    control_candidates = (
        df.where((F.col("main_analysis_row") == 1) & ((F.col("main_never_treated") == 1) | (F.col("main_late_treated_control") == 1)))
        .withColumn("baseline_rank", F.row_number().over(first_year_window))
        .where(F.col("baseline_rank") <= 3)
    )
    control_baseline = (
        control_candidates.groupBy("firm_key")
        .agg(*[F.avg(F.col(row["name"])).alias(row["name"]) for _, row in baseline_outcomes.iterrows()])
        .withColumn("group", F.lit("control"))
    )

    baseline_stack = treated_baseline.unionByName(control_baseline, allowMissingColumns=True)
    baseline_means = (
        baseline_stack.groupBy("group")
        .agg(*[F.avg(F.col(row["name"])).alias(row["name"]) for _, row in baseline_outcomes.iterrows()])
        .toPandas()
    )
    if not baseline_means.empty:
        treated_row = baseline_means.loc[baseline_means["group"] == "treated"]
        control_row = baseline_means.loc[baseline_means["group"] == "control"]
        comparison_rows = []
        for _, row in baseline_outcomes.iterrows():
            name = row["name"]
            treated_value = treated_row[name].iloc[0] if not treated_row.empty else None
            control_value = control_row[name].iloc[0] if not control_row.empty else None
            comparison_rows.append(
                {
                    "outcome": name,
                    "label": row["label"],
                    "treated_mean": treated_value,
                    "control_mean": control_value,
                    "difference": None if treated_value is None or control_value is None else treated_value - control_value,
                }
            )
        write_pandas_csv(pd.DataFrame(comparison_rows), output_dir / "baseline_treated_vs_control.csv")

    cohort_summary = (
        df.select(
            "firm_key",
            "first_people_analytics_firm_year_any_enriched",
            "main_balanced_treated",
            "hetero_public",
            "hetero_large",
            "hetero_data_intensive",
        )
        .dropDuplicates(["firm_key"])
        .where(F.col("main_balanced_treated") == 1)
        .groupBy("first_people_analytics_firm_year_any_enriched")
        .agg(
            F.countDistinct("firm_key").alias("balanced_treated_firms"),
            F.sum(F.when(F.col("hetero_public") == 1, 1).otherwise(0)).alias("public_firms"),
            F.sum(F.when(F.col("hetero_large") == 1, 1).otherwise(0)).alias("large_firms"),
            F.sum(F.when(F.col("hetero_data_intensive") == 1, 1).otherwise(0)).alias("data_intensive_firms"),
        )
        .orderBy("first_people_analytics_firm_year_any_enriched")
        .toPandas()
        .rename(columns={"first_people_analytics_firm_year_any_enriched": "cohort_year"})
    )
    write_pandas_csv(cohort_summary, output_dir / "treatment_cohort_counts.csv")

    logger.info("Summary tables written to %s", output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
