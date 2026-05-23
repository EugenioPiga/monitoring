#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from pyspark.sql import DataFrame, functions as F

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.revelio_analysis_utils import (  # noqa: E402
    append_restriction,
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
    write_restriction_outputs,
)
from utils.revelio_event_study_design import (  # noqa: E402
    build_joint_year_frame,
    configured_visibility_variables,
    event_dummy_name,
    event_time_values,
    optional_outcomes,
    recommend_estimation_window,
    required_outcomes,
    safe_visibility_name,
    visibility_enabled,
    visibility_interaction_name,
)


JOIN_TREATMENT_COLUMNS = [
    "first_people_analytics_posting_year_any_enriched",
    "is_first_people_analytics_posting_year_any_enriched",
    "has_people_analytics_posting_any_enriched_by_year",
]

KEY_COLUMNS = ["parent_rcid", "occupation", "year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe-v3 parent-occupation event-study samples.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--parent-year-dir", default=None)
    parser.add_argument("--parent-occ-dir", default=None)
    parser.add_argument("--visibility-panel-dir", default=None)
    parser.add_argument("--inspection-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Directory where base sample parquet outputs will be written.")
    parser.add_argument("--visibility-output-dir", default=None, help="Directory where visibility sample parquet outputs will be written.")
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=1200)
    parser.add_argument("--coalesce", type=int, default=300)
    parser.add_argument("--tmpdir", default=None)
    return parser.parse_args()


def write_parquet(df: DataFrame, path: Path, coalesce: int) -> None:
    writer = df
    if coalesce > 0:
        writer = writer.coalesce(max(1, coalesce))
    writer.write.mode("overwrite").option("compression", "snappy").parquet(str(path))


def load_window(
    *,
    inspection_dir: Path,
    parent_year_summary: pd.DataFrame,
    parent_occ_summary: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    window_path = inspection_dir / "10_recommended_window.json"
    if window_path.exists():
        return load_json(window_path)
    joined = build_joint_year_frame(parent_year_summary, parent_occ_summary)
    recommended = recommend_estimation_window(joined, config, current_year=datetime.utcnow().year)
    return recommended["recommended_window"]


def build_parent_year_summary(df: DataFrame) -> pd.DataFrame:
    analysis_col = "analysis_sample" if "analysis_sample" in df.columns else None
    return (
        df.groupBy("year")
        .agg(
            F.count(F.lit(1)).alias("parent_year_rows"),
            F.approx_count_distinct("parent_rcid").alias("parent_year_approx_parents"),
            F.sum(F.when(F.col("is_first_people_analytics_posting_year_any_enriched") == 1, 1).otherwise(0)).alias("parent_year_adoptions"),
            (
                F.sum(F.when(F.col(analysis_col) == 1, 1).otherwise(0)).alias("parent_year_analysis_rows")
                if analysis_col
                else F.lit(None).cast("double").alias("parent_year_analysis_rows")
            ),
        )
        .orderBy("year")
        .toPandas()
    )


def build_parent_occ_summary(df: DataFrame) -> pd.DataFrame:
    analysis_col = "occupation_analysis_sample" if "occupation_analysis_sample" in df.columns else None
    return (
        df.groupBy("year")
        .agg(
            F.count(F.lit(1)).alias("parent_occ_rows"),
            F.approx_count_distinct("parent_rcid").alias("parent_occ_approx_parents"),
            F.approx_count_distinct("occupation").alias("parent_occ_approx_occupations"),
            (
                F.sum(F.when(F.col(analysis_col) == 1, 1).otherwise(0)).alias("parent_occ_analysis_rows")
                if analysis_col
                else F.lit(None).cast("double").alias("parent_occ_analysis_rows")
            ),
        )
        .orderBy("year")
        .toPandas()
    )


def build_parent_treatment(parent_year: DataFrame) -> tuple[DataFrame, DataFrame]:
    timing = (
        parent_year.where(F.col("parent_rcid").isNotNull())
        .groupBy("parent_rcid")
        .agg(
            F.min("first_people_analytics_posting_year_any_enriched").alias("first_people_analytics_posting_year_any_enriched"),
            F.max("is_first_people_analytics_posting_year_any_enriched").alias("any_first_indicator"),
            F.max("has_people_analytics_posting_any_enriched_by_year").alias("ever_adopted_by_year_flag"),
            F.countDistinct("first_people_analytics_posting_year_any_enriched").alias("distinct_timing_values"),
        )
        .withColumn(
            "ever_treated",
            F.when(F.col("first_people_analytics_posting_year_any_enriched").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
        )
    )
    inconsistent = timing.where(F.col("distinct_timing_values") > 1)
    return timing.drop("any_first_indicator", "ever_adopted_by_year_flag"), inconsistent


def add_event_columns(df: DataFrame, config: dict[str, object]) -> DataFrame:
    bin_min = int(config["event_time"]["bin_min"])
    bin_max = int(config["event_time"]["bin_max"])
    omit_event_time = int(config["event_time"]["omit_event_time"])
    event_values = event_time_values(config, include_omitted=True)

    df = df.withColumn(
        "event_time_raw",
        F.when(
            F.col("first_people_analytics_posting_year_any_enriched").isNotNull(),
            F.col("year") - F.col("first_people_analytics_posting_year_any_enriched"),
        ).otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "event_time_binned",
        F.when(F.col("event_time_raw").isNull(), F.lit(None))
        .when(F.col("event_time_raw") < F.lit(bin_min), F.lit(bin_min))
        .when(F.col("event_time_raw") > F.lit(bin_max), F.lit(bin_max))
        .otherwise(F.col("event_time_raw")),
    )
    df = df.withColumn("never_treated", F.when(F.col("ever_treated") == 1, F.lit(0)).otherwise(F.lit(1)))
    df = df.withColumn(
        "not_yet_treated",
        F.when((F.col("ever_treated") == 1) & (F.col("year") < F.col("first_people_analytics_posting_year_any_enriched")), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "post",
        F.when((F.col("ever_treated") == 1) & (F.col("year") >= F.col("first_people_analytics_posting_year_any_enriched")), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn("treated_event_row", F.when(F.col("ever_treated") == 1, F.lit(1)).otherwise(F.lit(0)))
    for event_time in event_values:
        dummy_name = event_dummy_name(event_time)
        df = df.withColumn(
            dummy_name,
            F.when(
                (F.col("treated_event_row") == 1) & (F.col("event_time_binned") == F.lit(event_time)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    df = df.withColumn("omit_event_time", F.lit(omit_event_time))
    return df


def build_support_table(df: DataFrame) -> pd.DataFrame:
    return (
        df.where(F.col("treated_event_row") == 1)
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


def build_cohort_table(df: DataFrame) -> pd.DataFrame:
    return (
        df.select("parent_rcid", "first_people_analytics_posting_year_any_enriched")
        .dropDuplicates(["parent_rcid"])
        .where(F.col("first_people_analytics_posting_year_any_enriched").isNotNull())
        .groupBy("first_people_analytics_posting_year_any_enriched")
        .agg(F.count("*").alias("treated_parents"))
        .orderBy("first_people_analytics_posting_year_any_enriched")
        .toPandas()
        .rename(columns={"first_people_analytics_posting_year_any_enriched": "cohort_year"})
    )


def build_outcome_nonmissing(df: DataFrame, outcomes: list[str]) -> pd.DataFrame:
    agg_exprs = []
    for outcome in outcomes:
        if outcome in df.columns:
            agg_exprs.append(F.sum(F.when(F.col(outcome).isNotNull(), 1).otherwise(0)).alias(outcome))
    overall = df.agg(*agg_exprs).toPandas().T.reset_index()
    overall.columns = ["outcome", "nonmissing_rows"]
    return overall


def build_stacked_sample(base_df: DataFrame, config: dict[str, object], logger) -> tuple[DataFrame | None, pd.DataFrame]:
    pre_window = int(config["event_time"]["pre_window"])
    post_window = int(config["event_time"]["post_window"])
    min_treated = int(config["event_time"]["stack_min_treated_parents"])
    event_values = event_time_values(config, include_omitted=True)

    cohort_frame = (
        base_df.select("parent_rcid", "first_people_analytics_posting_year_any_enriched")
        .dropDuplicates(["parent_rcid"])
        .where(F.col("first_people_analytics_posting_year_any_enriched").isNotNull())
        .groupBy("first_people_analytics_posting_year_any_enriched")
        .agg(F.count("*").alias("treated_parents"))
        .where(F.col("treated_parents") >= F.lit(min_treated))
        .orderBy("first_people_analytics_posting_year_any_enriched")
        .collect()
    )
    eligible_cohorts = [int(row["first_people_analytics_posting_year_any_enriched"]) for row in cohort_frame]
    logger.info("Eligible stacked cohorts: %s", eligible_cohorts)
    if not eligible_cohorts:
        return None, pd.DataFrame(columns=["stack_cohort", "treated_parents", "control_parents"])

    stacked_parts = []
    support_rows: list[dict[str, object]] = []
    for cohort in eligible_cohorts:
        subset = (
            base_df.where(F.col("year").between(cohort - pre_window, cohort + post_window))
            .where(
                (F.col("first_people_analytics_posting_year_any_enriched") == F.lit(cohort))
                | F.col("first_people_analytics_posting_year_any_enriched").isNull()
                | (F.col("first_people_analytics_posting_year_any_enriched") > F.lit(cohort + post_window))
            )
            .withColumn("stack_cohort", F.lit(cohort))
            .withColumn(
                "stack_treated",
                F.when(F.col("first_people_analytics_posting_year_any_enriched") == F.lit(cohort), F.lit(1)).otherwise(F.lit(0)),
            )
            .withColumn("stack_event_time_raw", F.col("year") - F.lit(cohort))
            .withColumn("stack_event_time_binned", F.col("year") - F.lit(cohort))
            .withColumn("stack_parent_occ_fe", F.concat_ws("::", F.lit(str(cohort)), F.col("parent_occ_fe")))
            .withColumn("stack_parent_year_fe", F.concat_ws("::", F.lit(str(cohort)), F.col("parent_year_fe")))
            .withColumn("stack_occupation_year_fe", F.concat_ws("::", F.lit(str(cohort)), F.col("occupation_year_fe")))
        )
        for event_time in event_values:
            subset = subset.withColumn(
                event_dummy_name(event_time, prefix="stack_event"),
                F.when(
                    (F.col("stack_treated") == 1) & (F.col("stack_event_time_binned") == F.lit(event_time)),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
        stacked_parts.append(subset)
        counts = (
            subset.groupBy("stack_treated")
            .agg(F.approx_count_distinct("parent_rcid").alias("approx_parents"))
            .toPandas()
        )
        treated_count = int(counts.loc[counts["stack_treated"] == 1, "approx_parents"].sum()) if not counts.empty else 0
        control_count = int(counts.loc[counts["stack_treated"] == 0, "approx_parents"].sum()) if not counts.empty else 0
        support_rows.append({"stack_cohort": cohort, "treated_parents": treated_count, "control_parents": control_count})

    unioned = stacked_parts[0]
    for part in stacked_parts[1:]:
        unioned = unioned.unionByName(part, allowMissingColumns=True)
    return unioned, pd.DataFrame(support_rows)


def distinct_unit_columns(source_level: str) -> list[str]:
    if source_level == "occupation":
        return ["occupation"]
    if source_level == "parent_occ":
        return ["parent_rcid", "occupation"]
    return ["parent_rcid", "occupation", "year"]


def find_available_visibility_specs(
    config: dict[str, object],
    base_columns: list[str],
    visibility_columns: list[str],
) -> list[dict[str, object]]:
    available = set(base_columns) | set(visibility_columns)
    specs: list[dict[str, object]] = []
    for spec in configured_visibility_variables(config):
        name = spec["name"]
        if name in available:
            enriched = dict(spec)
            enriched["safe_name"] = safe_visibility_name(name)
            enriched["present_in_base_sample"] = name in base_columns
            enriched["present_in_visibility_panel"] = name in visibility_columns
            specs.append(enriched)
    return specs


def add_visibility_columns(
    sample: DataFrame,
    *,
    visibility_specs: list[dict[str, object]],
    visibility_panel: DataFrame | None,
    logger,
    config: dict[str, object],
) -> tuple[DataFrame, pd.DataFrame]:
    if not visibility_specs:
        return sample, pd.DataFrame(
            columns=[
                "visibility_variable",
                "visibility_label",
                "safe_name",
                "source_level",
                "mean",
                "std_dev",
                "median",
                "min",
                "max",
                "n_nonmissing_rows",
                "n_nonmissing_units",
                "skip_regression",
                "skip_reason",
            ]
        )

    source_columns = [spec["name"] for spec in visibility_specs if spec["name"] not in sample.columns]
    if source_columns:
        if visibility_panel is None:
            missing = ", ".join(source_columns)
            raise ValueError(f"Visibility variables require a companion panel, but none was available: {missing}")
        visibility_source = visibility_panel.select(*KEY_COLUMNS, *source_columns)
        dup = visibility_source.groupBy(*KEY_COLUMNS).count().where(F.col("count") > 1)
        if dup.limit(1).count() > 0:
            raise ValueError("Configured visibility companion panel is not unique on (parent_rcid, occupation, year).")
        visibility_source = visibility_source.dropDuplicates(KEY_COLUMNS)
        sample = sample.join(visibility_source, on=KEY_COLUMNS, how="left")

    minimum_std = float(config["visibility_event_studies"].get("minimum_std_dev", 1e-8))
    summaries: list[dict[str, object]] = []
    for spec in visibility_specs:
        name = spec["name"]
        safe_name = spec["safe_name"]
        raw_col = f"{safe_name}_raw"
        std_col = f"{safe_name}_std"
        high_col = f"{safe_name}_high"
        unit_cols = distinct_unit_columns(str(spec.get("source_level", "occupation")))

        sample = sample.withColumn(raw_col, F.col(name).cast("double"))
        unit_frame = sample.select(*(unit_cols + [raw_col])).where(F.col(raw_col).isNotNull()).dropDuplicates(unit_cols)
        n_units = unit_frame.count()
        n_nonmissing_rows = sample.where(F.col(raw_col).isNotNull()).count()
        stats_row = (
            unit_frame.agg(
                F.avg(F.col(raw_col)).alias("mean"),
                F.stddev(F.col(raw_col)).alias("std_dev"),
                F.min(F.col(raw_col)).alias("min"),
                F.max(F.col(raw_col)).alias("max"),
            ).toPandas().iloc[0].to_dict()
            if n_units > 0
            else {"mean": None, "std_dev": None, "min": None, "max": None}
        )
        median = None
        if n_units > 0:
            quantiles = unit_frame.approxQuantile(raw_col, [0.5], 0.001)
            median = quantiles[0] if quantiles else None

        mean_value = stats_row.get("mean")
        std_value = stats_row.get("std_dev")
        skip_reason = ""
        skip_regression = False
        if n_units == 0 or mean_value is None or std_value is None:
            skip_regression = True
            skip_reason = "no_nonmissing_visibility"
            sample = sample.withColumn(std_col, F.lit(None).cast("double"))
            sample = sample.withColumn(high_col, F.lit(None).cast("int"))
        elif abs(float(std_value)) <= minimum_std:
            skip_regression = True
            skip_reason = "near_zero_variance"
            sample = sample.withColumn(std_col, F.lit(None).cast("double"))
            sample = sample.withColumn(
                high_col,
                F.when(F.col(raw_col).isNotNull(), F.lit(1)).otherwise(F.lit(None).cast("int")),
            )
        else:
            sample = sample.withColumn(std_col, (F.col(raw_col) - F.lit(float(mean_value))) / F.lit(float(std_value)))
            sample = sample.withColumn(
                high_col,
                F.when(F.col(raw_col).isNull(), F.lit(None).cast("int"))
                .when(F.col(raw_col) >= F.lit(float(median)), F.lit(1))
                .otherwise(F.lit(0)),
            )

        logger.info(
            "Visibility variable %s: n_units=%s n_nonmissing_rows=%s mean=%s std=%s skip=%s reason=%s",
            name,
            n_units,
            n_nonmissing_rows,
            mean_value,
            std_value,
            skip_regression,
            skip_reason,
        )
        summaries.append(
            {
                "visibility_variable": name,
                "visibility_label": spec.get("label", name),
                "safe_name": safe_name,
                "source_level": spec.get("source_level", "occupation"),
                "mean": mean_value,
                "std_dev": std_value,
                "median": median,
                "min": stats_row.get("min"),
                "max": stats_row.get("max"),
                "n_nonmissing_rows": n_nonmissing_rows,
                "n_nonmissing_units": n_units,
                "skip_regression": int(skip_regression),
                "skip_reason": skip_reason,
            }
        )

    return sample, pd.DataFrame(summaries)


def add_visibility_interactions(sample: DataFrame, config: dict[str, object], visibility_specs: list[dict[str, object]], *, prefix: str = "event") -> DataFrame:
    for spec in visibility_specs:
        std_col = f"{spec['safe_name']}_std"
        for event_time in event_time_values(config, include_omitted=False):
            interaction_name = visibility_interaction_name(event_time, spec["name"], prefix=prefix)
            sample = sample.withColumn(interaction_name, F.col(event_dummy_name(event_time, prefix=prefix)).cast("double") * F.col(std_col))
    return sample


def build_visibility_support(sample: DataFrame, visibility_specs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in visibility_specs:
        raw_col = f"{spec['safe_name']}_raw"
        support = (
            sample.where((F.col("treated_event_row") == 1) & F.col(raw_col).isNotNull())
            .groupBy("event_time_binned")
            .agg(
                F.count(F.lit(1)).alias("treated_rows"),
                F.approx_count_distinct("parent_rcid").alias("treated_approx_parents"),
                F.approx_count_distinct("parent_occ_fe").alias("treated_approx_parent_occ_cells"),
            )
            .orderBy("event_time_binned")
            .toPandas()
        )
        if support.empty:
            rows.append(
                {
                    "visibility_variable": spec["name"],
                    "visibility_label": spec.get("label", spec["name"]),
                    "event_time": None,
                    "treated_rows": 0,
                    "treated_approx_parents": 0,
                    "treated_approx_parent_occ_cells": 0,
                }
            )
            continue
        support["visibility_variable"] = spec["name"]
        support["visibility_label"] = spec.get("label", spec["name"])
        support = support.rename(columns={"event_time_binned": "event_time"})
        rows.extend(support.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_visibility_missingness(sample: DataFrame, visibility_specs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in visibility_specs:
        raw_col = f"{spec['safe_name']}_raw"
        row = (
            sample.agg(
                F.count(F.lit(1)).alias("n_rows"),
                F.sum(F.when(F.col(raw_col).isNull(), 1).otherwise(0)).alias("missing_rows"),
                F.approx_count_distinct("parent_rcid").alias("n_parents"),
                F.approx_count_distinct("occupation").alias("n_occupations"),
                F.approx_count_distinct("year").alias("n_years"),
            ).toPandas().iloc[0].to_dict()
        )
        row["visibility_variable"] = spec["name"]
        row["visibility_label"] = spec.get("label", spec["name"])
        row["nonmissing_rows"] = int(row["n_rows"] - row["missing_rows"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_stacked_visibility_support(stacked_sample: DataFrame | None, visibility_specs: list[dict[str, object]]) -> pd.DataFrame:
    if stacked_sample is None:
        return pd.DataFrame(columns=["visibility_variable", "stack_cohort", "event_time", "treated_rows", "treated_approx_parents", "treated_approx_parent_occ_cells"])
    rows: list[dict[str, object]] = []
    for spec in visibility_specs:
        raw_col = f"{spec['safe_name']}_raw"
        support = (
            stacked_sample.where((F.col("stack_treated") == 1) & F.col(raw_col).isNotNull())
            .groupBy("stack_cohort", "stack_event_time_binned")
            .agg(
                F.count(F.lit(1)).alias("treated_rows"),
                F.approx_count_distinct("parent_rcid").alias("treated_approx_parents"),
                F.approx_count_distinct("parent_occ_fe").alias("treated_approx_parent_occ_cells"),
            )
            .orderBy("stack_cohort", "stack_event_time_binned")
            .toPandas()
        )
        if support.empty:
            continue
        support["visibility_variable"] = spec["name"]
        support = support.rename(columns={"stack_event_time_binned": "event_time"})
        rows.extend(support.to_dict(orient="records"))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root, output_relative_root=config["output_relative_root"])
    ensure_analysis_directories(paths)

    inspection_dir = Path(args.inspection_dir) if args.inspection_dir else paths.inspection_root
    output_dir = Path(args.output_dir) if args.output_dir else paths.sample_root
    visibility_output_dir = Path(args.visibility_output_dir) if args.visibility_output_dir else paths.visibility_sample_root
    ensure_directory(inspection_dir)
    ensure_directory(output_dir)
    ensure_directory(visibility_output_dir)

    logger = setup_logging("01_build_event_study_sample", paths.logs_root)
    parent_year_dir = Path(args.parent_year_dir) if args.parent_year_dir else default_parent_year_path(args.project_root, config)
    parent_occ_dir = Path(args.parent_occ_dir) if args.parent_occ_dir else default_parent_occ_path(args.project_root, config)
    visibility_panel_dir = Path(args.visibility_panel_dir) if args.visibility_panel_dir else default_visibility_panel_path(args.project_root, config)

    spark = create_spark(
        "build_parent_occ_event_study_sample",
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    parent_year = spark.read.parquet(str(parent_year_dir))
    parent_occ = spark.read.parquet(str(parent_occ_dir))
    visibility_panel = spark.read.parquet(str(visibility_panel_dir)) if visibility_enabled(config) and visibility_panel_dir.exists() else None

    required_outcome_cols = required_outcomes(config)
    optional_outcome_cols = [column for column in optional_outcomes(config) if column in parent_occ.columns]
    active_outcomes = [column for column in required_outcome_cols + optional_outcome_cols if column in parent_occ.columns]

    restriction_records: list[dict[str, object]] = []
    parent_year_summary = build_parent_year_summary(parent_year)
    parent_occ_summary = build_parent_occ_summary(parent_occ)
    window = load_window(
        inspection_dir=inspection_dir,
        parent_year_summary=parent_year_summary,
        parent_occ_summary=parent_occ_summary,
        config=config,
    )

    treatment_map, inconsistent_timing = build_parent_treatment(parent_year)
    inconsistent_count = inconsistent_timing.count()
    if inconsistent_count > 0:
        raise ValueError(f"Found {inconsistent_count} parents with inconsistent treatment timing in parent-year input.")

    base_before = parent_occ.count()
    parent_occ = parent_occ.where(F.col("parent_rcid").isNotNull() & F.col("occupation").isNotNull() & F.col("year").isNotNull())
    base_after = parent_occ.count()
    append_restriction(
        restriction_records,
        step="drop_missing_parent_occ_keys",
        before_rows=base_before,
        after_rows=base_after,
        reason="parent_rcid, occupation, and year must all be present",
    )

    if "occupation_analysis_sample" in parent_occ.columns:
        before = base_after
        parent_occ = parent_occ.where(F.col("occupation_analysis_sample") == 1)
        after = parent_occ.count()
        append_restriction(
            restriction_records,
            step="occupation_analysis_sample",
            before_rows=before,
            after_rows=after,
            reason="keep safe-v3 analysis cells with enough within-cell support",
        )
    else:
        after = base_after

    before = after
    parent_occ = parent_occ.where(F.col("year").between(int(window["start_year"]), int(window["end_year"])))
    after = parent_occ.count()
    append_restriction(
        restriction_records,
        step="recommended_window",
        before_rows=before,
        after_rows=after,
        reason="restrict to recommended estimation years from joint inspection",
        detail=f"{window['start_year']} to {window['end_year']}",
    )

    dup = parent_occ.groupBy(*KEY_COLUMNS).count().where(F.col("count") > 1)
    if dup.limit(1).count() > 0:
        dup.orderBy(F.col("count").desc()).limit(1000).coalesce(1).write.mode("overwrite").option("header", True).csv(str(output_dir / "duplicate_parent_occ_keys_probe"))
        raise ValueError("parent_occupation_year_panel_paonly_safe_v3 is not unique on (parent_rcid, occupation, year).")

    drop_existing = [column for column in JOIN_TREATMENT_COLUMNS if column in parent_occ.columns]
    if drop_existing:
        parent_occ = parent_occ.drop(*drop_existing)

    sample = parent_occ.join(
        treatment_map.select("parent_rcid", "first_people_analytics_posting_year_any_enriched", "ever_treated"),
        on="parent_rcid",
        how="left",
    )
    sample = sample.withColumn("parent_occ_fe", F.concat_ws("::", F.col("parent_rcid").cast("string"), F.col("occupation").cast("string")))
    sample = sample.withColumn("parent_year_fe", F.concat_ws("::", F.col("parent_rcid").cast("string"), F.col("year").cast("string")))
    sample = sample.withColumn("occupation_year_fe", F.concat_ws("::", F.col("occupation").cast("string"), F.col("year").cast("string")))
    sample = add_event_columns(sample, config)

    base_sample_path = output_dir / "parent_occ_event_study_sample.parquet"
    stacked_sample_path = output_dir / "parent_occ_event_study_stacked_sample.parquet"

    support_table = build_support_table(sample)
    cohort_table = build_cohort_table(sample)
    outcome_nonmissing = build_outcome_nonmissing(sample, active_outcomes)

    stacked_sample, stacked_support = build_stacked_sample(sample, config, logger)
    write_parquet(sample, base_sample_path, args.coalesce)
    if stacked_sample is not None:
        write_parquet(stacked_sample, stacked_sample_path, args.coalesce)

    stacked_event_support = pd.DataFrame()
    if stacked_sample is not None:
        stacked_event_support = (
            stacked_sample.where(F.col("stack_treated") == 1)
            .groupBy("stack_cohort", "stack_event_time_binned")
            .agg(
                F.count(F.lit(1)).alias("treated_rows"),
                F.approx_count_distinct("parent_rcid").alias("treated_approx_parents"),
                F.approx_count_distinct("parent_occ_fe").alias("treated_approx_parent_occ_cells"),
            )
            .orderBy("stack_cohort", "stack_event_time_binned")
            .toPandas()
            .rename(columns={"stack_event_time_binned": "event_time"})
        )

    metadata = {
        "parent_year_dir": str(parent_year_dir),
        "parent_occ_dir": str(parent_occ_dir),
        "sample_path": str(base_sample_path),
        "stacked_sample_path": str(stacked_sample_path) if stacked_sample is not None else None,
        "output_dir": str(output_dir),
        "recommended_window": window,
        "treatment_first_year_column": config["treatment"]["first_treat_col"],
        "event_time": config["event_time"],
        "active_outcomes": active_outcomes,
        "cluster_var": config["metadata"]["cluster_var"],
        "baseline_fixed_effects": [config["metadata"]["baseline_unit_fe"], config["metadata"]["baseline_time_fe"]],
        "stacked_fixed_effects": [config["metadata"]["stacked_unit_fe"], config["metadata"]["stacked_time_fe"]],
        "n_rows_sample": sample.count(),
        "n_parent_occ_cells": sample.select("parent_occ_fe").distinct().count(),
        "n_parents": sample.select("parent_rcid").distinct().count(),
        "n_treated_parents": sample.where(F.col("ever_treated") == 1).select("parent_rcid").distinct().count(),
        "stacked_available": stacked_sample is not None,
    }

    write_json(metadata, output_dir / "00_sample_metadata.json")
    write_pandas_csv(cohort_table, output_dir / "01_adoption_cohort_counts.csv")
    write_pandas_csv(support_table, output_dir / "02_event_time_support.csv")
    write_pandas_csv(outcome_nonmissing, output_dir / "03_outcome_nonmissing.csv")
    write_pandas_csv(stacked_support, output_dir / "04_stacked_cohort_support.csv")
    write_pandas_csv(stacked_event_support, output_dir / "05_stacked_event_time_support.csv")
    write_restriction_outputs(
        restriction_records,
        output_dir / "06_sample_restrictions.csv",
        output_dir / "06_sample_restrictions.md",
        "Parent-Occupation Event-Study Sample Restrictions",
    )

    logger.info("Base event-study sample written to %s", base_sample_path)

    if visibility_enabled(config):
        visibility_specs = find_available_visibility_specs(
            config,
            base_columns=sample.columns,
            visibility_columns=visibility_panel.columns if visibility_panel is not None else [],
        )
        if not visibility_specs:
            raise ValueError("Visibility event studies are enabled, but no configured visibility variables were found in the live sample or companion panel.")

        visibility_sample = sample
        visibility_sample, visibility_summary = add_visibility_columns(
            visibility_sample,
            visibility_specs=visibility_specs,
            visibility_panel=visibility_panel,
            logger=logger,
            config=config,
        )
        visibility_sample = add_visibility_interactions(visibility_sample, config, visibility_specs, prefix="event")
        visibility_stacked_sample, visibility_stacked_support = build_stacked_sample(visibility_sample, config, logger)
        if visibility_stacked_sample is not None:
            visibility_stacked_sample = add_visibility_interactions(visibility_stacked_sample, config, visibility_specs, prefix="stack_event")

        visibility_support = build_visibility_support(visibility_sample, visibility_specs)
        visibility_missingness = build_visibility_missingness(visibility_sample, visibility_specs)
        visibility_stacked_event_support = build_stacked_visibility_support(visibility_stacked_sample, visibility_specs)

        visibility_sample_path = visibility_output_dir / "parent_occ_visibility_event_study_sample.parquet"
        visibility_stacked_sample_path = visibility_output_dir / "parent_occ_visibility_event_study_stacked_sample.parquet"
        write_parquet(visibility_sample, visibility_sample_path, args.coalesce)
        if visibility_stacked_sample is not None:
            write_parquet(visibility_stacked_sample, visibility_stacked_sample_path, args.coalesce)

        visibility_metadata = {
            "visibility_panel_dir": str(visibility_panel_dir) if visibility_panel_dir.exists() else None,
            "visibility_sample_path": str(visibility_sample_path),
            "visibility_stacked_sample_path": str(visibility_stacked_sample_path) if visibility_stacked_sample is not None else None,
            "visibility_variables_used": visibility_summary["visibility_variable"].tolist(),
            "visibility_output_dir": str(visibility_output_dir),
            "n_rows_visibility_sample": visibility_sample.count(),
            "n_parent_occ_cells": visibility_sample.select("parent_occ_fe").distinct().count(),
            "n_parents": visibility_sample.select("parent_rcid").distinct().count(),
            "n_occupations": visibility_sample.select("occupation").distinct().count(),
            "n_years": visibility_sample.select("year").distinct().count(),
            "cluster_var": config["metadata"]["cluster_var"],
            "fixed_effects": [
                config["metadata"]["visibility_unit_fe"],
                config["metadata"]["visibility_parent_time_fe"],
                config["metadata"]["visibility_occ_time_fe"],
            ],
            "stacked_fixed_effects": [
                config["metadata"]["stacked_visibility_unit_fe"],
                config["metadata"]["stacked_visibility_parent_time_fe"],
                config["metadata"]["stacked_visibility_occ_time_fe"],
            ],
        }

        write_json(visibility_metadata, visibility_output_dir / "00_visibility_sample_metadata.json")
        write_pandas_csv(visibility_summary, visibility_output_dir / "01_visibility_variable_summary.csv")
        write_pandas_csv(visibility_support, visibility_output_dir / "02_visibility_event_time_support.csv")
        write_pandas_csv(visibility_missingness, visibility_output_dir / "03_visibility_missingness.csv")
        write_pandas_csv(visibility_stacked_support, visibility_output_dir / "04_visibility_stacked_cohort_support.csv")
        write_pandas_csv(visibility_stacked_event_support, visibility_output_dir / "05_visibility_stacked_event_time_support.csv")
        logger.info("Visibility event-study sample written to %s", visibility_sample_path)

    spark.stop()


if __name__ == "__main__":
    main()
