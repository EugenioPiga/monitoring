#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"

import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pyspark.sql import functions as F

from utils.revelio_analysis_utils import (
    build_analysis_paths,
    create_spark,
    default_dataset_path,
    ensure_analysis_directories,
    ensure_directory,
    load_json,
    setup_logging,
    write_json,
    write_pandas_csv,
    write_text,
)
from utils.revelio_event_study_design import recommend_windows


REQUIRED_COLUMNS = [
    "firm_key",
    "year",
    "has_position_data",
    "has_posting_data",
    "first_people_analytics_firm_year_any_enriched",
    "first_people_analytics_position_year_any_enriched",
    "first_people_analytics_posting_year_any_enriched",
    "is_first_people_analytics_firm_year_any_enriched",
    "is_first_people_analytics_position_year_any_enriched",
    "is_first_people_analytics_posting_year_any_enriched",
]

PROFILE_COLUMNS = [
    "firm_key",
    "year",
    "has_position_data",
    "has_posting_data",
    "parent_rcid_matched",
    "workforce_weighted",
    "hire_rate",
    "exit_rate",
    "avg_salary",
    "avg_start_salary",
    "avg_end_salary",
    "avg_seniority",
    "posting_count",
    "avg_posting_salary",
    "people_analytics_positions_any_enriched_share",
    "people_analytics_postings_any_enriched_share",
    "first_people_analytics_firm_year_any_enriched",
    "first_people_analytics_position_year_any_enriched",
    "first_people_analytics_posting_year_any_enriched",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Revelio firm-year inputs for event-study analysis.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for inspection outputs. Defaults to processed/analysis/diagnostics/input_inspection.",
    )
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    parser.add_argument("--tmpdir", default=None)
    return parser.parse_args()


def missing_profile(df, columns: list[str]) -> pd.DataFrame:
    dtype_map = dict(df.dtypes)
    exprs = []
    for column in columns:
        if column not in dtype_map:
            continue
        col_expr = F.col(column)
        if dtype_map[column] in {"double", "float"}:
            missing_expr = F.sum(F.when(col_expr.isNull() | F.isnan(col_expr), 1).otherwise(0)).alias(column)
        else:
            missing_expr = F.sum(F.when(col_expr.isNull(), 1).otherwise(0)).alias(column)
        exprs.append(missing_expr)

    row_count = df.limit(1000000).count()  # capped diagnostic count
    if not exprs:
        return pd.DataFrame(columns=["column_name", "dtype", "missing_count", "missing_share", "nonmissing_count"])

    row = df.agg(*exprs).collect()[0].asDict()
    records = []
    for column in columns:
        if column not in dtype_map:
            continue
        missing_count = int(row.get(column, 0) or 0)
        records.append(
            {
                "column_name": column,
                "dtype": dtype_map[column],
                "missing_count": missing_count,
                "missing_share": missing_count / row_count if row_count > 0 else None,
                "nonmissing_count": row_count - missing_count,
            }
        )
    return pd.DataFrame(records).sort_values(["missing_share", "column_name"], ascending=[False, True]).reset_index(drop=True)


def build_year_summary(df) -> pd.DataFrame:
    ordered = (
        df.groupBy("year")
        .agg(
            F.count(F.lit(1)).alias("row_count"),
            F.countDistinct("firm_key").alias("distinct_firms"),
            F.sum(F.when(F.col("has_position_data") == 1, 1).otherwise(0)).alias("position_rows"),
            F.sum(F.when(F.col("has_posting_data") == 1, 1).otherwise(0)).alias("posting_rows"),
            F.sum(F.when((F.col("has_position_data") == 1) & (F.col("has_posting_data") == 1), 1).otherwise(0)).alias("both_rows"),
            F.sum(F.when((F.col("has_position_data") == 1) & (F.col("has_posting_data") == 0), 1).otherwise(0)).alias("position_only_rows"),
            F.sum(F.when((F.col("has_position_data") == 0) & (F.col("has_posting_data") == 1), 1).otherwise(0)).alias("posting_only_rows"),
            F.sum(F.when(F.col("hire_rate").isNotNull(), 1).otherwise(0)).alias("nonmissing_hire_rate"),
            F.sum(F.when(F.col("exit_rate").isNotNull(), 1).otherwise(0)).alias("nonmissing_exit_rate"),
            F.sum(F.when(F.col("workforce_weighted").isNotNull(), 1).otherwise(0)).alias("nonmissing_workforce"),
            F.sum(F.when(F.col("posting_count").isNotNull(), 1).otherwise(0)).alias("nonmissing_posting_count"),
            F.sum(F.when(F.col("avg_salary").isNotNull(), 1).otherwise(0)).alias("nonmissing_avg_salary"),
            F.sum(F.when(F.col("is_first_people_analytics_firm_year_any_enriched") == 1, 1).otherwise(0)).alias("main_adoptions"),
            F.sum(F.when(F.col("is_first_people_analytics_position_year_any_enriched") == 1, 1).otherwise(0)).alias("position_adoptions"),
            F.sum(F.when(F.col("is_first_people_analytics_posting_year_any_enriched") == 1, 1).otherwise(0)).alias("posting_adoptions"),
        )
        .orderBy(F.col("year").asc_nulls_last())
    )
    return ordered.toPandas()


def build_first_treat_distribution(df, first_treat_col: str, label: str) -> pd.DataFrame:
    if first_treat_col not in df.columns:
        return pd.DataFrame(columns=["first_treat_year", "firm_count", "label"])
    frame = (
        df.select("firm_key", first_treat_col)
        .dropDuplicates(["firm_key"])
        .where(F.col(first_treat_col).isNotNull())
        .groupBy(first_treat_col)
        .agg(F.countDistinct("firm_key").alias("firm_count"))
        .orderBy(first_treat_col)
        .toPandas()
    )
    if frame.empty:
        frame = pd.DataFrame(columns=[first_treat_col, "firm_count"])
    frame = frame.rename(columns={first_treat_col: "first_treat_year"})
    frame["label"] = label
    return frame


def build_firm_coverage_regime(df) -> pd.DataFrame:
    firm_level = (
        df.groupBy("firm_key")
        .agg(
            F.max(F.when(F.col("has_position_data") == 1, 1).otherwise(0)).alias("ever_position"),
            F.max(F.when(F.col("has_posting_data") == 1, 1).otherwise(0)).alias("ever_posting"),
        )
        .withColumn(
            "coverage_regime",
            F.when((F.col("ever_position") == 1) & (F.col("ever_posting") == 1), F.lit("both"))
            .when((F.col("ever_position") == 1) & (F.col("ever_posting") == 0), F.lit("position_only"))
            .when((F.col("ever_position") == 0) & (F.col("ever_posting") == 1), F.lit("posting_only"))
            .otherwise(F.lit("neither")),
        )
    )
    return (
        firm_level.groupBy("coverage_regime")
        .agg(F.countDistinct("firm_key").alias("firm_count"))
        .orderBy("coverage_regime")
        .toPandas()
    )


def build_memo(
    summary: dict[str, object],
    recommended_windows: dict[str, object],
    year_flags: pd.DataFrame,
    firm_coverage: pd.DataFrame,
) -> str:
    lines = [
        "Revelio event-study input inspection memo",
        "",
        f"Dataset path: {summary['dataset_path']}",
        f"Total firm-year rows: {summary['row_count']:,}",
        f"Distinct firms: {summary['distinct_firms']:,}",
        f"Duplicate (firm_key, year) cells: {summary['duplicate_key_cells']:,}",
        "",
        "Recommended estimation windows:",
    ]
    for key, values in recommended_windows.items():
        lines.append(
            f"- {key}: {values['start_year']} to {values['end_year']} ({values['basis']})"
        )

    flagged = year_flags[
        year_flags["flag_invalid_calendar_year"]
        | year_flags["flag_tiny_tail_year"]
        | year_flags["flag_outside_main_window"]
        | year_flags["flag_outside_posting_window"]
    ].copy()
    lines.append("")
    lines.append("Flagged years:")
    if flagged.empty:
        lines.append("- No flagged years.")
    else:
        for _, row in flagged.sort_values("year").iterrows():
            reasons = []
            if row["flag_invalid_calendar_year"]:
                reasons.append("invalid_calendar_year")
            if row["flag_tiny_tail_year"]:
                reasons.append("tiny_tail_year")
            if row["flag_outside_main_window"]:
                reasons.append("outside_main_window")
            if row["flag_outside_posting_window"]:
                reasons.append("outside_posting_window")
            lines.append(f"- {int(row['year'])}: {', '.join(reasons)}")

    lines.append("")
    lines.append("Firm coverage regimes:")
    for _, row in firm_coverage.iterrows():
        lines.append(f"- {row['coverage_regime']}: {int(row['firm_count']):,} firms")
    lines.append("")
    lines.append("Suggested first-pass restrictions:")
    lines.append("- Exclude invalid year artifacts and tiny tail years automatically.")
    lines.append("- Use the firm-level adoption year as the main treatment and treat position-based and posting-based definitions as robustness checks.")
    lines.append("- Restrict posting-based event studies to the modern posting-support window rather than extrapolating into zero-coverage years.")
    lines.append("- Re-check 2023 in the live panel before using it for estimation; the diagnostics indicate an anomalous end year relative to the 2020-2022 pattern.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root)
    ensure_analysis_directories(paths)

    output_dir = Path(args.output_dir) if args.output_dir else paths.diagnostics_root / "input_inspection"
    ensure_directory(output_dir)

    logger = setup_logging("00_inspect_revelio_event_study_inputs", paths.logs_root)
    dataset_path = Path(args.dataset_path) if args.dataset_path else default_dataset_path(args.project_root, config)

    spark = create_spark(
        "revelio_event_study_input_inspection",
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    logger.info("Reading dataset from %s", dataset_path)
    df = spark.read.parquet(str(dataset_path))

    missing_required = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")

    row_count = df.limit(1000000).count()  # capped diagnostic count
    distinct_firms = df.select("firm_key").limit(1000000).distinct().limit(1000000).count()  # capped diagnostic count
    duplicate_key_cells = (
        df.groupBy("firm_key", "year")
        .limit(1000000).count()  # capped diagnostic count
        .where(F.col("count") > 1)
        .limit(1000000).count()  # capped diagnostic count
    )

    year_summary = build_year_summary(df)
    recommended = recommend_windows(year_summary, config, current_year=datetime.utcnow().year)
    classified_years = recommended["classified_years"].copy()
    windows = recommended["recommended_windows"]

    classified_years["flag_invalid_calendar_year"] = ~classified_years["valid_calendar_year"]
    classified_years["flag_tiny_tail_year"] = classified_years["tiny_tail_year"]
    classified_years["flag_outside_main_window"] = (
        (classified_years["year"] < windows["main"]["start_year"])
        | (classified_years["year"] > windows["main"]["end_year"])
    )
    classified_years["flag_outside_posting_window"] = (
        (classified_years["year"] < windows["posting"]["start_year"])
        | (classified_years["year"] > windows["posting"]["end_year"])
    )

    adoption_counts = classified_years[
        ["year", "main_adoptions", "position_adoptions", "posting_adoptions"]
    ].copy()
    first_treat_tables = []
    first_treat_tables.append(
        build_first_treat_distribution(df, "first_people_analytics_firm_year_any_enriched", "main")
    )
    first_treat_tables.append(
        build_first_treat_distribution(df, "first_people_analytics_position_year_any_enriched", "position")
    )
    first_treat_tables.append(
        build_first_treat_distribution(df, "first_people_analytics_posting_year_any_enriched", "posting")
    )
    first_treat_distribution = pd.concat(first_treat_tables, ignore_index=True)
    firm_coverage = build_firm_coverage_regime(df)
    profile = missing_profile(df, [column for column in PROFILE_COLUMNS if column in df.columns])

    summary = {
        "dataset_path": str(dataset_path),
        "row_count": int(row_count),
        "distinct_firms": int(distinct_firms),
        "duplicate_key_cells": int(duplicate_key_cells),
        "recommended_windows": windows,
    }

    memo = build_memo(summary, windows, classified_years, firm_coverage)

    write_json(summary, output_dir / "00_inspection_summary.json")
    write_pandas_csv(year_summary, output_dir / "01_year_summary.csv")
    write_pandas_csv(classified_years, output_dir / "02_year_classification.csv")
    write_pandas_csv(adoption_counts, output_dir / "03_adoption_counts_by_year.csv")
    write_pandas_csv(first_treat_distribution, output_dir / "04_first_treat_distributions.csv")
    write_pandas_csv(firm_coverage, output_dir / "05_firm_coverage_regimes.csv")
    write_pandas_csv(profile, output_dir / "06_missingness_profile.csv")
    write_json(windows, output_dir / "07_recommended_estimation_windows.json")
    write_text(memo, output_dir / "08_suggested_restrictions_memo.txt")

    logger.info("Inspection complete. Outputs written to %s", output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
