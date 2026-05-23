#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from pyspark.sql import functions as F

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.revelio_analysis_utils import (  # noqa: E402
    build_analysis_paths,
    create_spark,
    ensure_analysis_directories,
    ensure_directory,
    load_json,
    setup_logging,
    write_pandas_csv,
    write_pandas_latex,
)
from utils.revelio_event_study_design import safe_visibility_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact safe-v3 event-study tables.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visibility-sample-dir", default=None)
    parser.add_argument("--visibility-results-dir", default=None)
    parser.add_argument("--visibility-output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--mode", choices=["base", "visibility", "both"], default="both")
    return parser.parse_args()


def make_base_tables(sample_dir: Path, results_dir: Path, output_dir: Path, config: dict[str, object], spark) -> None:
    ensure_directory(output_dir)
    coefficients = pd.read_csv(results_dir / "02_event_study_coefficients.csv")
    pretrend = pd.read_csv(results_dir / "03_pretrend_summary.csv")
    status = pd.read_csv(results_dir / "05_model_status.csv")

    post_average = (
        coefficients.loc[coefficients["event_time"] >= 0]
        .groupby(["estimator", "estimator_label", "outcome", "outcome_label"], as_index=False)
        .agg(
            mean_post_estimate=("estimate", "mean"),
            mean_post_abs_tstat=("statistic", lambda x: x.abs().mean()),
            n_post_coefficients=("event_time", "count"),
        )
    )

    estimator_summary = post_average.merge(
        pretrend,
        on=["estimator", "estimator_label", "outcome", "outcome_label"],
        how="left",
    ).merge(
        status.loc[status["status"] == "ok", ["estimator", "outcome", "nobs", "n_clusters", "n_parent_occ", "n_occupations", "n_years"]],
        on=["estimator", "outcome"],
        how="left",
    )
    estimator_summary = estimator_summary.sort_values(["estimator", "outcome"])

    sample = spark.read.parquet(str(sample_dir / "parent_occ_event_study_sample.parquet"))
    active_outcomes = [row["name"] for row in config["outcomes"] if row["name"] in sample.columns]

    cohort_counts = (
        sample.select("parent_rcid", "first_people_analytics_posting_year_any_enriched")
        .dropDuplicates(["parent_rcid"])
        .where(F.col("first_people_analytics_posting_year_any_enriched").isNotNull())
        .groupBy("first_people_analytics_posting_year_any_enriched")
        .agg(F.count("*").alias("treated_parents"))
        .orderBy("first_people_analytics_posting_year_any_enriched")
        .toPandas()
        .rename(columns={"first_people_analytics_posting_year_any_enriched": "cohort_year"})
    )

    event_support = (
        sample.where(F.col("treated_event_row") == 1)
        .groupBy("event_time_binned")
        .agg(
            F.count(F.lit(1)).alias("treated_rows"),
            F.approx_count_distinct("parent_rcid").alias("treated_approx_parents"),
            F.approx_count_distinct("parent_occ_fe").alias("treated_approx_parent_occ_cells"),
        )
        .orderBy("event_time_binned")
        .toPandas()
        .rename(columns={"event_time_binned": "event_time"})
    )

    mean_exprs = [F.avg(F.col(col)).alias(col) for col in active_outcomes]
    raw_event_means = (
        sample.where(F.col("treated_event_row") == 1)
        .groupBy("event_time_binned")
        .agg(*mean_exprs)
        .orderBy("event_time_binned")
        .toPandas()
        .rename(columns={"event_time_binned": "event_time"})
    )

    write_pandas_csv(estimator_summary, output_dir / "01_estimator_summary.csv")
    write_pandas_latex(estimator_summary, output_dir / "01_estimator_summary.tex", index=False)
    write_pandas_csv(cohort_counts, output_dir / "02_cohort_counts.csv")
    write_pandas_csv(event_support, output_dir / "03_event_time_support.csv")
    write_pandas_csv(raw_event_means, output_dir / "04_raw_event_time_means.csv")


def make_visibility_tables(
    visibility_sample_dir: Path,
    visibility_results_dir: Path,
    visibility_output_dir: Path,
    config: dict[str, object],
    spark,
) -> None:
    ensure_directory(visibility_output_dir)
    coefficients = pd.read_csv(visibility_results_dir / "02_visibility_event_study_coefficients.csv")
    pretrend = pd.read_csv(visibility_results_dir / "03_visibility_pretrend_summary.csv")
    status = pd.read_csv(visibility_results_dir / "04_visibility_model_status.csv")
    variable_summary = pd.read_csv(visibility_sample_dir / "01_visibility_variable_summary.csv")
    missingness = pd.read_csv(visibility_sample_dir / "03_visibility_missingness.csv")

    post_average = (
        coefficients.loc[coefficients["event_time"] >= 0]
        .groupby(["estimator", "outcome", "outcome_label", "visibility_variable", "visibility_label"], as_index=False)
        .agg(
            mean_post_estimate=("estimate", "mean"),
            mean_post_abs_tstat=("statistic", lambda x: x.abs().mean()),
            n_post_coefficients=("event_time", "count"),
        )
    )
    estimator_summary = post_average.merge(
        pretrend,
        on=["estimator", "outcome", "outcome_label", "visibility_variable", "visibility_label"],
        how="left",
    ).merge(
        status.loc[status["status"] == "ok", ["estimator", "outcome", "visibility_variable", "nobs", "n_clusters", "n_parent_occ", "n_occupations", "n_years"]],
        on=["estimator", "outcome", "visibility_variable"],
        how="left",
    )
    estimator_summary = estimator_summary.sort_values(["estimator", "visibility_variable", "outcome"])

    support_by_variable = variable_summary.merge(
        missingness[["visibility_variable", "nonmissing_rows", "missing_rows", "n_parents", "n_occupations", "n_years"]],
        on="visibility_variable",
        how="left",
    )
    support_by_variable = support_by_variable.sort_values("visibility_variable")

    sample = spark.read.parquet(str(visibility_sample_dir / "parent_occ_visibility_event_study_sample.parquet"))
    active_outcomes = [row["name"] for row in config["outcomes"] if row["name"] in sample.columns]

    raw_mean_frames: list[pd.DataFrame] = []
    for _, row in variable_summary.iterrows():
        variable = row["visibility_variable"]
        safe_name = row.get("safe_name", safe_visibility_name(variable))
        high_col = f"{safe_name}_high"
        if high_col not in sample.columns:
            continue
        for outcome in active_outcomes:
            grouped = (
                sample.where(F.col("treated_event_row") == 1)
                .where(F.col(high_col).isNotNull())
                .groupBy("event_time_binned", high_col)
                .agg(
                    F.avg(F.col(outcome)).alias("mean_outcome"),
                    F.count(F.lit(1)).alias("nobs"),
                )
                .orderBy("event_time_binned", high_col)
                .toPandas()
            )
            if grouped.empty:
                continue
            grouped["outcome"] = outcome
            grouped["visibility_variable"] = variable
            grouped["visibility_label"] = row.get("visibility_label", variable)
            grouped["visibility_group"] = grouped[high_col].map({0: "low", 1: "high"})
            grouped = grouped.rename(columns={"event_time_binned": "event_time"})
            grouped = grouped[["outcome", "visibility_variable", "visibility_label", "visibility_group", "event_time", "mean_outcome", "nobs"]]
            raw_mean_frames.append(grouped)

    raw_means = pd.concat(raw_mean_frames, ignore_index=True) if raw_mean_frames else pd.DataFrame(
        columns=["outcome", "visibility_variable", "visibility_label", "visibility_group", "event_time", "mean_outcome", "nobs"]
    )

    write_pandas_csv(estimator_summary, visibility_output_dir / "01_visibility_estimator_summary.csv")
    write_pandas_csv(pretrend, visibility_output_dir / "02_visibility_pretrend_summary.csv")
    write_pandas_csv(support_by_variable, visibility_output_dir / "03_visibility_support_by_variable.csv")
    write_pandas_csv(raw_means, visibility_output_dir / "04_visibility_raw_means_by_event_time_and_high_low.csv")


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root, output_relative_root=config["output_relative_root"])
    ensure_analysis_directories(paths)

    sample_dir = Path(args.sample_dir) if args.sample_dir else paths.sample_root
    results_dir = Path(args.results_dir) if args.results_dir else paths.results_root
    output_dir = Path(args.output_dir) if args.output_dir else paths.tables_root
    visibility_sample_dir = Path(args.visibility_sample_dir) if args.visibility_sample_dir else paths.visibility_sample_root
    visibility_results_dir = Path(args.visibility_results_dir) if args.visibility_results_dir else paths.visibility_results_root
    visibility_output_dir = Path(args.visibility_output_dir) if args.visibility_output_dir else paths.visibility_tables_root

    logger = setup_logging("04_make_event_study_tables", paths.logs_root)
    spark = create_spark(
        "make_parent_occ_event_study_tables",
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    if args.mode in {"base", "both"}:
        make_base_tables(sample_dir, results_dir, output_dir, config, spark)
        logger.info("Base event-study tables written to %s", output_dir)

    if args.mode in {"visibility", "both"}:
        make_visibility_tables(visibility_sample_dir, visibility_results_dir, visibility_output_dir, config, spark)
        logger.info("Visibility event-study tables written to %s", visibility_output_dir)

    spark.stop()


if __name__ == "__main__":
    main()
