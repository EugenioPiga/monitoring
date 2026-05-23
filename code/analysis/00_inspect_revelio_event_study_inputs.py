#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

import pandas as pd
from pyspark.sql import DataFrame, functions as F

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.revelio_analysis_utils import (  # noqa: E402
    build_analysis_paths,
    create_spark,
    default_parent_occ_path,
    default_parent_year_path,
    default_visibility_panel_path,
    ensure_analysis_directories,
    ensure_directory,
    load_json,
    setup_logging,
    write_json,
    write_pandas_csv,
    write_text,
)
from utils.revelio_event_study_design import (  # noqa: E402
    build_joint_year_frame,
    configured_visibility_variables,
    optional_outcomes,
    outcome_frame,
    recommend_estimation_window,
    required_outcomes,
    visibility_candidate_patterns,
)


PARENT_YEAR_REQUIRED = [
    "parent_rcid",
    "year",
    "first_people_analytics_posting_year_any_enriched",
]

PARENT_OCC_REQUIRED = [
    "parent_rcid",
    "occupation",
    "year",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect safe-v3 parent-year and parent-occupation-year event-study inputs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--parent-year-dir", default=None)
    parser.add_argument("--parent-occ-dir", default=None)
    parser.add_argument("--visibility-panel-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--sample-cap", type=int, default=1_000_000)
    return parser.parse_args()


def approx_parent_year_duplicates(df: DataFrame) -> tuple[int, pd.DataFrame]:
    dup = (
        df.where(F.col("parent_rcid").isNotNull() & F.col("year").isNotNull())
        .groupBy("parent_rcid", "year")
        .count()
        .where(F.col("count") > 1)
    )
    count = dup.limit(1000).count()
    sample = dup.orderBy(F.col("count").desc()).limit(1000).toPandas()
    return int(count), sample


def approx_parent_occ_duplicates(df: DataFrame, sample_cap: int) -> tuple[int, pd.DataFrame]:
    probe = df.select("parent_rcid", "occupation", "year").where(
        F.col("parent_rcid").isNotNull() & F.col("occupation").isNotNull() & F.col("year").isNotNull()
    ).limit(sample_cap)
    dup = probe.groupBy("parent_rcid", "occupation", "year").count().where(F.col("count") > 1)
    count = dup.limit(1000).count()
    sample = dup.orderBy(F.col("count").desc()).limit(1000).toPandas()
    return int(count), sample


def build_parent_year_summary(df: DataFrame) -> pd.DataFrame:
    analysis_col = "analysis_sample" if "analysis_sample" in df.columns else None
    posting_col = "people_analytics_postings_any_enriched" if "people_analytics_postings_any_enriched" in df.columns else None
    grouped = (
        df.groupBy("year")
        .agg(
            F.count(F.lit(1)).alias("parent_year_rows"),
            F.approx_count_distinct("parent_rcid").alias("parent_year_approx_parents"),
            F.sum(F.when(F.col("is_first_people_analytics_posting_year_any_enriched") == 1, 1).otherwise(0)).alias("parent_year_adoptions"),
            F.sum(F.when(F.col("first_people_analytics_posting_year_any_enriched").isNotNull(), 1).otherwise(0)).alias("parent_year_rows_with_timing"),
            (
                F.sum(F.when(F.col(analysis_col) == 1, 1).otherwise(0)).alias("parent_year_analysis_rows")
                if analysis_col
                else F.lit(None).cast("double").alias("parent_year_analysis_rows")
            ),
            (
                F.sum(F.when(F.col(posting_col).isNotNull(), 1).otherwise(0)).alias("parent_year_nonmissing_posting_signal")
                if posting_col
                else F.lit(None).cast("double").alias("parent_year_nonmissing_posting_signal")
            ),
        )
        .orderBy("year")
    )
    return grouped.toPandas()


def build_parent_occ_summary(df: DataFrame, outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_col = "occupation_analysis_sample" if "occupation_analysis_sample" in df.columns else None
    agg_exprs = [
        F.count(F.lit(1)).alias("parent_occ_rows"),
        F.approx_count_distinct("parent_rcid").alias("parent_occ_approx_parents"),
        F.approx_count_distinct("occupation").alias("parent_occ_approx_occupations"),
        (
            F.sum(F.when(F.col(analysis_col) == 1, 1).otherwise(0)).alias("parent_occ_analysis_rows")
            if analysis_col
            else F.lit(None).cast("double").alias("parent_occ_analysis_rows")
        ),
    ]
    for outcome in outcomes:
        if outcome in df.columns:
            agg_exprs.append(F.sum(F.when(F.col(outcome).isNotNull(), 1).otherwise(0)).alias(f"nonmissing__{outcome}"))
    yearly = df.groupBy("year").agg(*agg_exprs).orderBy("year").toPandas()

    overall_exprs = []
    for outcome in outcomes:
        if outcome in df.columns:
            overall_exprs.append(F.sum(F.when(F.col(outcome).isNotNull(), 1).otherwise(0)).alias(outcome))
    overall = df.agg(*overall_exprs).toPandas().T.reset_index()
    overall.columns = ["outcome", "nonmissing_rows"]
    return yearly, overall


def build_outcome_presence(df: DataFrame, config: dict[str, object]) -> pd.DataFrame:
    frame = outcome_frame(config)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        outcome = str(row["name"])
        present = outcome in df.columns
        rows.append(
            {
                "outcome": outcome,
                "label": row["label"],
                "group": row["group"],
                "optional": bool(row.get("optional", False)),
                "present": int(present),
            }
        )
    return pd.DataFrame(rows)


def build_parent_timing_distribution(df: DataFrame) -> pd.DataFrame:
    timing = (
        df.select("parent_rcid", "first_people_analytics_posting_year_any_enriched")
        .where(F.col("parent_rcid").isNotNull())
        .groupBy("parent_rcid")
        .agg(F.min("first_people_analytics_posting_year_any_enriched").alias("first_treat_year"))
        .where(F.col("first_treat_year").isNotNull())
        .groupBy("first_treat_year")
        .agg(F.count("*").alias("treated_parents"))
        .orderBy("first_treat_year")
    )
    return timing.toPandas()


def build_parent_timing_consistency(df: DataFrame) -> pd.DataFrame:
    timing = (
        df.select("parent_rcid", "first_people_analytics_posting_year_any_enriched")
        .where(F.col("parent_rcid").isNotNull())
        .groupBy("parent_rcid")
        .agg(
            F.countDistinct("first_people_analytics_posting_year_any_enriched").alias("distinct_timing_values"),
            F.min("first_people_analytics_posting_year_any_enriched").alias("min_first_treat_year"),
            F.max("first_people_analytics_posting_year_any_enriched").alias("max_first_treat_year"),
        )
        .where(F.col("distinct_timing_values") > 1)
        .orderBy(F.col("distinct_timing_values").desc(), "parent_rcid")
        .limit(1000)
    )
    return timing.toPandas()


def is_numeric_dtype(dtype: str) -> bool:
    numeric_prefixes = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal", "long", "short")
    return any(dtype.startswith(prefix) for prefix in numeric_prefixes)


def candidate_visibility_columns(columns: list[str], patterns: list[str]) -> list[str]:
    regex = re.compile("|".join(re.escape(pattern) for pattern in patterns), re.IGNORECASE)
    return [column for column in columns if regex.search(column)]


def build_candidate_profile(df: DataFrame, columns: list[str], sample_cap: int, source_dataset: str) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "column_name",
                "dtype",
                "n_nonmissing",
                "n_distinct_approx",
                "mean",
                "sd",
                "min",
                "max",
            ]
        )
    probe = df.select(*columns).limit(sample_cap).cache()
    _ = probe.count()
    dtype_map = dict(probe.dtypes)
    rows: list[dict[str, object]] = []
    for column in columns:
        dtype = dtype_map.get(column, "")
        exprs = [
            F.sum(F.when(F.col(column).isNotNull(), 1).otherwise(0)).alias("n_nonmissing"),
            F.approx_count_distinct(F.col(column)).alias("n_distinct_approx"),
        ]
        if is_numeric_dtype(dtype):
            exprs.extend(
                [
                    F.avg(F.col(column).cast("double")).alias("mean"),
                    F.stddev(F.col(column).cast("double")).alias("sd"),
                    F.min(F.col(column).cast("double")).alias("min"),
                    F.max(F.col(column).cast("double")).alias("max"),
                ]
            )
        stats = probe.agg(*exprs).toPandas().iloc[0].to_dict()
        rows.append(
            {
                "source_dataset": source_dataset,
                "column_name": column,
                "dtype": dtype,
                "n_nonmissing": int(stats.get("n_nonmissing") or 0),
                "n_distinct_approx": int(stats.get("n_distinct_approx") or 0),
                "mean": stats.get("mean"),
                "sd": stats.get("sd"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }
        )
    probe.unpersist()
    return pd.DataFrame(rows)


def build_memo(
    *,
    parent_year_path: Path,
    parent_occ_path: Path,
    visibility_panel_path: Path | None,
    recommended_window: dict[str, object],
    classified_years: pd.DataFrame,
    optional_present: list[str],
    required_missing: list[str],
    selected_visibility: list[str],
    candidate_frame: pd.DataFrame,
) -> str:
    lines = [
        "Safe-v3 parent-occupation event-study inspection memo",
        "",
        f"Parent-year input: {parent_year_path}",
        f"Parent-occupation-year input: {parent_occ_path}",
        f"Visibility source inspected: {visibility_panel_path if visibility_panel_path is not None else 'not configured'}",
        "",
        "Recommended estimation window:",
        f"- Start year: {recommended_window['start_year']}",
        f"- End year: {recommended_window['end_year']}",
        f"- Basis: {recommended_window['basis']}",
        "",
        "Interpretation:",
        "- The treatment timing should come from the parent-year PA-posting adoption panel, not the old firm-year panel.",
        "- The outcome panel should remain at parent x occupation x year so the estimating variation is within parent-occupation cells over time.",
        "- The visibility mechanism branch can use static occupation visibility from the safe-v3 monitoring-exposure panel even when the base parent-occupation panel does not carry those columns directly.",
        "",
    ]
    if required_missing:
        lines.append("Missing required outcome columns:")
        for column in required_missing:
            lines.append(f"- {column}")
        lines.append("")
    if optional_present:
        lines.append("Optional five-year outcomes available:")
        for column in optional_present:
            lines.append(f"- {column}")
        lines.append("")

    lines.append("Configured visibility variables:")
    if selected_visibility:
        for column in selected_visibility:
            matches = candidate_frame.loc[candidate_frame["column_name"] == column, "source_dataset"].tolist()
            source = matches[0] if matches else "missing_from_live_schema"
            lines.append(f"- {column} ({source})")
    else:
        lines.append("- None selected.")
    lines.append("")

    flagged = classified_years[
        (~classified_years["valid_calendar_year"])
        | classified_years["tiny_tail_year"]
        | classified_years["outside_recommended_window"]
    ].copy()
    lines.append("Flagged years:")
    if flagged.empty:
        lines.append("- No flagged years.")
    else:
        for _, row in flagged.sort_values("year").iterrows():
            reasons: list[str] = []
            if not bool(row["valid_calendar_year"]):
                reasons.append("invalid_calendar_year")
            if bool(row["tiny_tail_year"]):
                reasons.append("tiny_tail_year")
            if bool(row["outside_recommended_window"]):
                reasons.append("outside_recommended_window")
            lines.append(f"- {int(row['year'])}: {', '.join(reasons)}")
    lines.append("")
    lines.append("Key caution:")
    lines.append("- hr_to_employee_ratio is excluded from event-study estimation because the current safe-v3 HR numerator is all zero.")
    return "\n".join(lines) + "\n"


def build_visibility_error_memo(
    *,
    parent_occ_path: Path,
    visibility_panel_path: Path | None,
    patterns: list[str],
) -> str:
    lines = [
        "Visibility-interacted event-study inspection failed.",
        "",
        "No plausible visibility / exposure columns were found using the configured search patterns.",
        f"Parent-occupation-year input inspected: {parent_occ_path}",
        f"Visibility panel inspected: {visibility_panel_path if visibility_panel_path is not None else 'not configured'}",
        f"Search patterns: {', '.join(patterns)}",
        "",
        "Action required:",
        "- Either populate the base parent-occupation-year panel with predetermined visibility variables,",
        "- or point the config to the correct companion visibility panel.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root, output_relative_root=config["output_relative_root"])
    ensure_analysis_directories(paths)
    output_dir = Path(args.output_dir) if args.output_dir else paths.inspection_root
    ensure_directory(output_dir)

    logger = setup_logging("00_inspect_revelio_event_study_inputs", paths.logs_root)
    parent_year_dir = Path(args.parent_year_dir) if args.parent_year_dir else default_parent_year_path(args.project_root, config)
    parent_occ_dir = Path(args.parent_occ_dir) if args.parent_occ_dir else default_parent_occ_path(args.project_root, config)
    visibility_panel_dir = Path(args.visibility_panel_dir) if args.visibility_panel_dir else default_visibility_panel_path(args.project_root, config)
    visibility_dir_exists = visibility_panel_dir.exists()

    spark = create_spark(
        "inspect_parent_occ_event_study_inputs",
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    logger.info("Reading parent-year panel from %s", parent_year_dir)
    parent_year = spark.read.parquet(str(parent_year_dir))
    logger.info("Reading parent-occupation-year panel from %s", parent_occ_dir)
    parent_occ = spark.read.parquet(str(parent_occ_dir))

    missing_parent_year = sorted(set(PARENT_YEAR_REQUIRED) - set(parent_year.columns))
    if missing_parent_year:
        raise ValueError(f"Parent-year panel is missing required columns: {missing_parent_year}")

    required_outcome_cols = required_outcomes(config)
    missing_parent_occ = sorted(set(PARENT_OCC_REQUIRED + required_outcome_cols) - set(parent_occ.columns))
    if missing_parent_occ:
        raise ValueError(f"Parent-occupation-year panel is missing required columns: {missing_parent_occ}")

    optional_outcome_cols = [column for column in optional_outcomes(config) if column in parent_occ.columns]

    parent_year_rows_capped = parent_year.limit(args.sample_cap).count()
    parent_occ_rows_capped = parent_occ.limit(args.sample_cap).count()
    parent_year_dup_count, parent_year_dup_sample = approx_parent_year_duplicates(parent_year)
    parent_occ_dup_count, parent_occ_dup_sample = approx_parent_occ_duplicates(parent_occ, args.sample_cap)

    parent_year_summary = build_parent_year_summary(parent_year)
    parent_occ_summary, parent_occ_nonmissing = build_parent_occ_summary(parent_occ, required_outcome_cols + optional_outcome_cols)
    year_frame = build_joint_year_frame(parent_year_summary, parent_occ_summary)
    recommended = recommend_estimation_window(year_frame, config, current_year=datetime.utcnow().year)
    classified_years = recommended["classified_years"].copy()
    window = recommended["recommended_window"]

    parent_timing_distribution = build_parent_timing_distribution(parent_year)
    timing_consistency = build_parent_timing_consistency(parent_year)
    outcome_presence = build_outcome_presence(parent_occ, config)

    optional_present = outcome_presence.loc[(outcome_presence["optional"] == True) & (outcome_presence["present"] == 1), "outcome"].tolist()
    required_missing = outcome_presence.loc[(outcome_presence["optional"] == False) & (outcome_presence["present"] == 0), "outcome"].tolist()

    patterns = visibility_candidate_patterns(config)
    parent_occ_candidates = candidate_visibility_columns(parent_occ.columns, patterns)
    candidate_frames = [build_candidate_profile(parent_occ, parent_occ_candidates, args.sample_cap, "parent_occ_panel")]

    visibility_panel = None
    visibility_panel_candidates: list[str] = []
    if visibility_dir_exists:
        logger.info("Reading visibility companion panel from %s", visibility_panel_dir)
        visibility_panel = spark.read.parquet(str(visibility_panel_dir))
        visibility_panel_candidates = candidate_visibility_columns(visibility_panel.columns, patterns)
        candidate_frames.append(build_candidate_profile(visibility_panel, visibility_panel_candidates, args.sample_cap, "visibility_panel"))

    candidate_frame = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(columns=["source_dataset", "column_name", "dtype", "n_nonmissing", "n_distinct_approx", "mean", "sd", "min", "max"])
    )
    candidate_frame = candidate_frame.sort_values(["source_dataset", "column_name"]).reset_index(drop=True)

    configured_visibility = configured_visibility_variables(config)
    configured_names = [item["name"] for item in configured_visibility]
    available_columns = set(parent_occ_candidates) | set(visibility_panel_candidates)
    selected_visibility = [name for name in configured_names if name in available_columns]

    metadata = {
        "parent_year_dir": str(parent_year_dir),
        "parent_occ_dir": str(parent_occ_dir),
        "visibility_panel_dir": str(visibility_panel_dir) if visibility_dir_exists else None,
        "output_dir": str(output_dir),
        "treatment_first_year_column": config["treatment"]["first_treat_col"],
        "required_outcomes": required_outcome_cols,
        "optional_outcomes_present": optional_present,
        "visibility_variables_configured": configured_names,
        "visibility_variables_found": selected_visibility,
        "event_time_window": {
            "bin_min": config["event_time"]["bin_min"],
            "bin_max": config["event_time"]["bin_max"],
            "omit_event_time": config["event_time"]["omit_event_time"],
        },
        "parent_year_rows_capped": int(parent_year_rows_capped),
        "parent_occ_rows_capped": int(parent_occ_rows_capped),
        "parent_year_duplicate_key_probe_count": int(parent_year_dup_count),
        "parent_occ_duplicate_key_probe_count": int(parent_occ_dup_count),
        "recommended_window": window,
        "cluster_var": config["metadata"]["cluster_var"],
        "baseline_fixed_effects": [
            config["metadata"]["baseline_unit_fe"],
            config["metadata"]["baseline_time_fe"],
        ],
    }

    write_json(metadata, output_dir / "00_metadata.json")
    write_pandas_csv(parent_year_summary, output_dir / "01_parent_year_summary_by_year.csv")
    write_pandas_csv(parent_occ_summary, output_dir / "02_parent_occ_summary_by_year.csv")
    write_pandas_csv(classified_years, output_dir / "03_joint_year_classification.csv")
    write_pandas_csv(parent_timing_distribution, output_dir / "04_parent_adoption_cohorts.csv")
    write_pandas_csv(parent_occ_nonmissing, output_dir / "05_outcome_nonmissing_counts.csv")
    write_pandas_csv(outcome_presence, output_dir / "06_outcome_presence.csv")
    write_pandas_csv(parent_year_dup_sample, output_dir / "07_parent_year_duplicate_probe.csv")
    write_pandas_csv(parent_occ_dup_sample, output_dir / "08_parent_occ_duplicate_probe.csv")
    write_pandas_csv(timing_consistency, output_dir / "09_parent_timing_inconsistencies.csv")
    write_json(window, output_dir / "10_recommended_window.json")
    write_pandas_csv(candidate_frame, output_dir / "12_visibility_candidate_columns.csv")

    if candidate_frame.empty:
        error_memo = build_visibility_error_memo(
            parent_occ_path=parent_occ_dir,
            visibility_panel_path=visibility_panel_dir if visibility_dir_exists else None,
            patterns=patterns,
        )
        write_text(error_memo, output_dir / "13_visibility_candidate_memo.txt")
        spark.stop()
        raise RuntimeError("No plausible visibility/exposure columns found. See inspection/13_visibility_candidate_memo.txt")

    memo = build_memo(
        parent_year_path=parent_year_dir,
        parent_occ_path=parent_occ_dir,
        visibility_panel_path=visibility_panel_dir if visibility_dir_exists else None,
        recommended_window=window,
        classified_years=classified_years,
        optional_present=optional_present,
        required_missing=required_missing,
        selected_visibility=selected_visibility,
        candidate_frame=candidate_frame,
    )
    write_text(memo, output_dir / "11_inspection_memo.txt")
    write_text(
        "Visibility candidates were searched first in the base parent-occupation-year panel and then in the configured visibility companion panel.\n",
        output_dir / "13_visibility_candidate_memo.txt",
    )

    if not selected_visibility:
        spark.stop()
        raise RuntimeError(
            "Configured visibility variables were not found in the live schemas. "
            "See inspection/12_visibility_candidate_columns.csv and inspection/13_visibility_candidate_memo.txt."
        )

    logger.info("Inspection complete. Outputs written to %s", output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
