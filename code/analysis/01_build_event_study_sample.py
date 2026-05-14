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

from pyspark.sql import Window
from pyspark.sql import functions as F

from utils.revelio_analysis_utils import (
    append_restriction,
    build_analysis_paths,
    create_spark,
    default_dataset_path,
    ensure_analysis_directories,
    ensure_directory,
    extract_naics_digits,
    load_json,
    setup_logging,
    write_json,
    write_pandas_csv,
    write_restriction_outputs,
)
from utils.revelio_event_study_design import choose_supported_event_window, recommend_windows, treatment_frame


BASELINE_DATA_INTENSITY_COMPONENTS = [
    "workers_with_data_skill_share",
    "avg_predicted_skill_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Revelio event-study estimation sample.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--inspection-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for the sample parquet.")
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=600)
    parser.add_argument("--coalesce", type=int, default=120)
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--min-pre-periods", type=int, default=None)
    parser.add_argument("--min-post-periods", type=int, default=None)
    return parser.parse_args()


def build_year_summary(df) -> pd.DataFrame:
    ordered = (
        df.groupBy("year")
        .agg(
            F.count(F.lit(1)).alias("row_count"),
            F.countDistinct("firm_key").alias("distinct_firms"),
            F.sum(F.when(F.col("has_position_data") == 1, 1).otherwise(0)).alias("position_rows"),
            F.sum(F.when(F.col("has_posting_data") == 1, 1).otherwise(0)).alias("posting_rows"),
            F.sum(F.when(F.col("is_first_people_analytics_firm_year_any_enriched") == 1, 1).otherwise(0)).alias("main_adoptions"),
            F.sum(F.when(F.col("is_first_people_analytics_position_year_any_enriched") == 1, 1).otherwise(0)).alias("position_adoptions"),
            F.sum(F.when(F.col("is_first_people_analytics_posting_year_any_enriched") == 1, 1).otherwise(0)).alias("posting_adoptions"),
        )
        .orderBy(F.col("year").asc_nulls_last())
    )
    return ordered.toPandas()


def load_windows(df, inspection_dir: Path, config: dict[str, object]) -> dict[str, object]:
    window_path = inspection_dir / "07_recommended_estimation_windows.json"
    if window_path.exists():
        return load_json(window_path)
    year_summary = build_year_summary(df)
    recommended = recommend_windows(year_summary, config, current_year=datetime.utcnow().year)
    return recommended["recommended_windows"]


def write_parquet(df, path: Path, coalesce: int) -> None:
    writer = df
    if coalesce > 0:
        writer = writer.coalesce(max(1, coalesce))
    writer.write.mode("overwrite").option("compression", "snappy").parquet(str(path))


def add_winsorized_columns(df, winsorization: list[dict[str, object]], logger) -> tuple[object, dict[str, dict[str, float]]]:
    quantile_map: dict[str, dict[str, float]] = {}
    for spec in winsorization:
        column = str(spec["column"])
        if column not in df.columns:
            continue
        lower_prob = float(spec["lower"])
        upper_prob = float(spec["upper"])
        quantiles = df.approxQuantile(column, [lower_prob, upper_prob], 0.001)
        if len(quantiles) != 2:
            logger.warning("Skipping winsorization for %s because approximate quantiles were unavailable.", column)
            continue
        lower_value, upper_value = quantiles
        quantile_map[column] = {
            "lower_probability": lower_prob,
            "upper_probability": upper_prob,
            "lower_value": float(lower_value),
            "upper_value": float(upper_value),
        }
        winsorized_name = f"{column}_winsor_p{int(round(lower_prob * 100)):02d}_p{int(round(upper_prob * 100)):02d}"
        logger.info(
            "Winsorizing %s into %s with bounds [%s, %s]",
            column,
            winsorized_name,
            lower_value,
            upper_value,
        )
        df = df.withColumn(
            winsorized_name,
            F.when(
                F.col(column).isNull(),
                F.lit(None),
            ).otherwise(
                F.least(F.greatest(F.col(column).cast("double"), F.lit(lower_value)), F.lit(upper_value))
            ),
        )
    return df, quantile_map


def add_main_heterogeneity_flags(df, global_start_year: int):
    main_treat_col = "first_people_analytics_firm_year_any_enriched"
    df = df.withColumn(
        "main_event_time_raw",
        F.when(F.col(main_treat_col).isNotNull(), F.col("year") - F.col(main_treat_col)).otherwise(F.lit(None)),
    )
    data_intensity_components = [F.coalesce(F.col(column), F.lit(0.0)) for column in BASELINE_DATA_INTENSITY_COMPONENTS if column in df.columns]
    if data_intensity_components:
        data_intensity_expr = data_intensity_components[0]
        for component in data_intensity_components[1:]:
            data_intensity_expr = data_intensity_expr + component
        data_intensity_expr = data_intensity_expr / F.lit(float(len(data_intensity_components)))
    else:
        data_intensity_expr = F.lit(None)
    df = df.withColumn("baseline_data_intensity_source", data_intensity_expr)

    baseline_selector = (
        (F.col(main_treat_col).isNotNull() & F.col("main_event_time_raw").between(-3, -1))
        | (F.col(main_treat_col).isNull() & F.col("year").between(global_start_year, global_start_year + 2))
        | (F.col(main_treat_col).isNotNull() & (F.col(main_treat_col) > global_start_year + 2) & F.col("year").between(global_start_year, global_start_year + 2))
    )
    baseline_frame = (
        df.where(baseline_selector)
        .groupBy("firm_key")
        .agg(
            F.avg("workforce_weighted").alias("baseline_workforce_main"),
            F.avg("baseline_data_intensity_source").alias("baseline_data_intensity_main"),
            F.max(F.when((F.col("has_position_data") == 1) & (F.col("has_posting_data") == 1), 1).otherwise(0)).alias("firm_has_both_data_any_year"),
            F.max("is_public_company").alias("baseline_is_public_company"),
        )
    )

    medians = baseline_frame.select("baseline_workforce_main", "baseline_data_intensity_main").toPandas()
    size_cutoff = float(medians["baseline_workforce_main"].median()) if medians["baseline_workforce_main"].notna().any() else None
    intensity_cutoff = float(medians["baseline_data_intensity_main"].median()) if medians["baseline_data_intensity_main"].notna().any() else None

    if size_cutoff is None:
        baseline_frame = baseline_frame.withColumn("hetero_large", F.lit(None)).withColumn("hetero_small", F.lit(None))
    else:
        baseline_frame = baseline_frame.withColumn(
            "hetero_large",
            F.when(F.col("baseline_workforce_main") >= F.lit(size_cutoff), F.lit(1)).otherwise(F.lit(0)),
        )
        baseline_frame = baseline_frame.withColumn(
            "hetero_small",
            F.when(F.col("baseline_workforce_main").isNull(), F.lit(None)).when(F.col("hetero_large") == 1, F.lit(0)).otherwise(F.lit(1)),
        )

    if intensity_cutoff is None:
        baseline_frame = baseline_frame.withColumn("hetero_data_intensive", F.lit(None)).withColumn("hetero_less_data_intensive", F.lit(None))
    else:
        baseline_frame = baseline_frame.withColumn(
            "hetero_data_intensive",
            F.when(F.col("baseline_data_intensity_main") >= F.lit(intensity_cutoff), F.lit(1)).otherwise(F.lit(0)),
        )
        baseline_frame = baseline_frame.withColumn(
            "hetero_less_data_intensive",
            F.when(F.col("baseline_data_intensity_main").isNull(), F.lit(None)).when(F.col("hetero_data_intensive") == 1, F.lit(0)).otherwise(F.lit(1)),
        )
    baseline_frame = baseline_frame.withColumn(
        "hetero_public",
        F.when(F.col("baseline_is_public_company") == 1, F.lit(1))
        .when(F.col("baseline_is_public_company").isNull(), F.lit(None))
        .otherwise(F.lit(0)),
    )
    baseline_frame = baseline_frame.withColumn(
        "hetero_private",
        F.when(F.col("baseline_is_public_company") == 0, F.lit(1))
        .when(F.col("baseline_is_public_company").isNull(), F.lit(None))
        .otherwise(F.lit(0)),
    )

    return df.join(baseline_frame, on="firm_key", how="left"), {
        "size_cutoff": size_cutoff,
        "data_intensity_cutoff": intensity_cutoff,
    }


def add_treatment_design_columns(
    df,
    *,
    treatment_name: str,
    first_treat_col: str,
    start_year: int,
    end_year: int,
    min_pre_periods: int,
    min_post_periods: int,
):
    firm_window = Window.partitionBy("firm_key")
    prefix = treatment_name

    df = df.withColumn(f"{prefix}_analysis_year", F.col("year").between(start_year, end_year).cast("int"))
    df = df.withColumn(f"{prefix}_event_time_raw", F.when(F.col(first_treat_col).isNotNull(), F.col("year") - F.col(first_treat_col)).otherwise(F.lit(None)))
    df = df.withColumn(f"{prefix}_ever_treated", F.when(F.col(first_treat_col).isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    df = df.withColumn(f"{prefix}_never_treated", F.when(F.col(first_treat_col).isNull(), F.lit(1)).otherwise(F.lit(0)))
    df = df.withColumn(
        f"{prefix}_not_yet_treated",
        F.when(F.col(first_treat_col).isNotNull() & (F.col("year") < F.col(first_treat_col)), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        f"{prefix}_post",
        F.when(F.col(first_treat_col).isNotNull() & (F.col("year") >= F.col(first_treat_col)), F.lit(1)).otherwise(F.lit(0)),
    )

    earliest_eligible_cohort = start_year + min_pre_periods
    latest_eligible_cohort = end_year - min_post_periods
    df = df.withColumn(
        f"{prefix}_treated_cohort_in_window",
        F.when(F.col(first_treat_col).between(earliest_eligible_cohort, latest_eligible_cohort), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        f"{prefix}_early_treated_excluded",
        F.when(F.col(first_treat_col).isNotNull() & (F.col(first_treat_col) < earliest_eligible_cohort), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        f"{prefix}_late_treated_control",
        F.when(F.col(first_treat_col).isNotNull() & (F.col(first_treat_col) > latest_eligible_cohort), F.lit(1)).otherwise(F.lit(0)),
    )

    df = df.withColumn(
        f"{prefix}_pre_obs_count",
        F.sum(
            F.when(
                F.col("year").between(start_year, end_year) & F.col(first_treat_col).isNotNull() & (F.col("year") < F.col(first_treat_col)),
                1,
            ).otherwise(0)
        ).over(firm_window),
    )
    df = df.withColumn(
        f"{prefix}_post_obs_count",
        F.sum(
            F.when(
                F.col("year").between(start_year, end_year) & F.col(first_treat_col).isNotNull() & (F.col("year") >= F.col(first_treat_col)),
                1,
            ).otherwise(0)
        ).over(firm_window),
    )
    df = df.withColumn(
        f"{prefix}_balanced_treated",
        F.when(
            (F.col(f"{prefix}_treated_cohort_in_window") == 1)
            & (F.col(f"{prefix}_pre_obs_count") >= min_pre_periods)
            & (F.col(f"{prefix}_post_obs_count") >= min_post_periods),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    df = df.withColumn(
        f"{prefix}_analysis_row",
        F.when(
            (F.col("year").between(start_year, end_year))
            & (
                (F.col(f"{prefix}_never_treated") == 1)
                | (F.col(f"{prefix}_balanced_treated") == 1)
                | ((F.col(f"{prefix}_late_treated_control") == 1) & (F.col("year") < F.col(first_treat_col)))
            ),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    return df


def build_event_support(df, treatment_name: str) -> pd.DataFrame:
    prefix = treatment_name
    support = (
        df.where((F.col(f"{prefix}_analysis_row") == 1) & (F.col(f"{prefix}_balanced_treated") == 1))
        .where(F.col(f"{prefix}_event_time_raw").isNotNull())
        .where(F.col(f"{prefix}_event_time_raw").between(-8, 8))
        .groupBy(F.col(f"{prefix}_event_time_raw").alias("event_time"))
        .agg(
            F.count(F.lit(1)).alias("treated_rows"),
            F.countDistinct("firm_key").alias("treated_firms"),
        )
        .orderBy("event_time")
        .toPandas()
    )
    if support.empty:
        support = pd.DataFrame(columns=["event_time", "treated_rows", "treated_firms"])
    support["treatment_name"] = treatment_name
    return support


def add_binned_event_time(df, treatment_name: str, supported_window: int):
    prefix = treatment_name
    lower = -supported_window
    upper = supported_window
    df = df.withColumn(
        f"{prefix}_event_time_binned",
        F.when(
            F.col(f"{prefix}_event_time_raw").isNull(),
            F.lit(None),
        ).otherwise(
            F.when(F.col(f"{prefix}_event_time_raw") < lower, F.lit(lower))
            .when(F.col(f"{prefix}_event_time_raw") > upper, F.lit(upper))
            .otherwise(F.col(f"{prefix}_event_time_raw"))
        ),
    )
    df = df.withColumn(
        f"{prefix}_supported_window",
        F.when(F.col(f"{prefix}_analysis_row") == 1, F.lit(supported_window)).otherwise(F.lit(None)),
    )
    return df


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root)
    ensure_analysis_directories(paths)

    sample_path = Path(args.output_dir) if args.output_dir else paths.samples_root / "revelio_event_study_sample.parquet"
    diagnostics_dir = paths.diagnostics_root / "event_study_sample"
    tables_dir = paths.tables_root / "event_study_sample"
    inspection_dir = Path(args.inspection_dir) if args.inspection_dir else paths.diagnostics_root / "input_inspection"
    ensure_directory(sample_path.parent)
    ensure_directory(diagnostics_dir)
    ensure_directory(tables_dir)

    logger = setup_logging("01_build_event_study_sample", paths.logs_root)
    dataset_path = Path(args.dataset_path) if args.dataset_path else default_dataset_path(args.project_root, config)
    min_pre_periods = args.min_pre_periods or int(config["event_time_defaults"]["min_pre_periods"])
    min_post_periods = args.min_post_periods or int(config["event_time_defaults"]["min_post_periods"])

    spark = create_spark(
        "revelio_event_study_sample_builder",
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    logger.info("Reading dataset from %s", dataset_path)
    df = spark.read.parquet(str(dataset_path))
    initial_rows = df.count()
    restriction_records: list[dict[str, object]] = []

    before = initial_rows
    df = df.where(F.col("firm_key").isNotNull() & F.col("year").isNotNull())
    after = df.count()
    append_restriction(
        restriction_records,
        step="drop_missing_keys",
        before_rows=before,
        after_rows=after,
        reason="firm_key and year must both be present",
    )

    windows = load_windows(df, inspection_dir, config)
    global_start_year = min(window["start_year"] for window in windows.values())
    global_end_year = max(window["end_year"] for window in windows.values())

    before = after
    df = df.where(F.col("year").between(global_start_year, global_end_year))
    after = df.count()
    append_restriction(
        restriction_records,
        step="global_year_window",
        before_rows=before,
        after_rows=after,
        reason="restrict to common estimation years implied by inspection windows",
        detail=f"{global_start_year} to {global_end_year}",
    )

    df = df.withColumn("naics2", extract_naics_digits(F.col("naics_code"), 2))
    df = df.withColumn("has_both_data_by_year", F.when((F.col("has_position_data") == 1) & (F.col("has_posting_data") == 1), F.lit(1)).otherwise(F.lit(0)))
    df = df.withColumn("log_workforce", F.when(F.col("workforce_weighted") > 0, F.log(F.col("workforce_weighted"))).otherwise(F.lit(None)))
    df = df.withColumn("log_posting_count", F.when(F.col("posting_count").isNotNull(), F.log1p(F.col("posting_count"))).otherwise(F.lit(None)))

    lag_window = Window.partitionBy("firm_key").orderBy("year")
    df = df.withColumn("lag_log_workforce", F.lag("log_workforce").over(lag_window))
    df = df.withColumn(
        "workforce_growth",
        F.when(F.col("log_workforce").isNotNull() & F.col("lag_log_workforce").isNotNull(), F.col("log_workforce") - F.col("lag_log_workforce")).otherwise(F.lit(None)),
    )

    df, heterogeneity_meta = add_main_heterogeneity_flags(df, global_start_year)
    df, winsor_quantiles = add_winsorized_columns(df, config.get("winsorization", []), logger)

    treatment_specs = treatment_frame(config).to_dict(orient="records")
    support_tables: list[pd.DataFrame] = []
    cohort_records: list[dict[str, object]] = []
    supported_windows: dict[str, int] = {}

    for treatment in treatment_specs:
        treatment_name = str(treatment["name"])
        first_treat_col = str(treatment["first_treat_col"])
        window = windows[treatment_name]
        logger.info(
            "Adding treatment design columns for %s using %s over %s-%s",
            treatment_name,
            first_treat_col,
            window["start_year"],
            window["end_year"],
        )
        df = add_treatment_design_columns(
            df,
            treatment_name=treatment_name,
            first_treat_col=first_treat_col,
            start_year=int(window["start_year"]),
            end_year=int(window["end_year"]),
            min_pre_periods=min_pre_periods,
            min_post_periods=min_post_periods,
        )

        support = build_event_support(df, treatment_name)
        supported_window = choose_supported_event_window(
            support,
            config["event_time_defaults"]["candidate_windows"],
            min_event_bin_treated_firms=int(config["event_time_defaults"]["min_event_bin_treated_firms"]),
            min_event_bin_rows=int(config["event_time_defaults"]["min_event_bin_rows"]),
        )
        support["selected_window"] = supported_window
        support_tables.append(support)
        supported_windows[treatment_name] = supported_window
        df = add_binned_event_time(df, treatment_name, supported_window)

        cohort_counts = (
            df.select(
                "firm_key",
                F.col(first_treat_col).alias("first_treat_year"),
                F.col(f"{treatment_name}_ever_treated").alias("ever_treated"),
                F.col(f"{treatment_name}_treated_cohort_in_window").alias("treated_cohort_in_window"),
                F.col(f"{treatment_name}_balanced_treated").alias("balanced_treated"),
                F.col(f"{treatment_name}_early_treated_excluded").alias("early_treated_excluded"),
                F.col(f"{treatment_name}_late_treated_control").alias("late_treated_control"),
            )
            .dropDuplicates(["firm_key"])
            .toPandas()
        )
        cohort_records.append(
            {
                "treatment_name": treatment_name,
                "eligible_window_start_year": int(window["start_year"]),
                "eligible_window_end_year": int(window["end_year"]),
                "selected_event_window": int(supported_window),
                "never_treated_firms": int((cohort_counts["ever_treated"] == 0).sum()),
                "treated_firms_any": int((cohort_counts["ever_treated"] == 1).sum()),
                "treated_cohort_in_window_firms": int((cohort_counts["treated_cohort_in_window"] == 1).sum()),
                "balanced_treated_firms": int((cohort_counts["balanced_treated"] == 1).sum()),
                "early_treated_excluded_firms": int((cohort_counts["early_treated_excluded"] == 1).sum()),
                "late_treated_control_firms": int((cohort_counts["late_treated_control"] == 1).sum()),
            }
        )

        cohort_by_year = (
            cohort_counts.loc[cohort_counts["balanced_treated"] == 1, ["first_treat_year"]]
            .value_counts()
            .reset_index(name="balanced_treated_firms")
            .sort_values("first_treat_year")
        )
        cohort_by_year["treatment_name"] = treatment_name
        write_pandas_csv(cohort_by_year, tables_dir / f"{treatment_name}_cohort_counts_by_year.csv")

    support_frame = pd.concat(support_tables, ignore_index=True) if support_tables else pd.DataFrame()
    cohort_summary = pd.DataFrame(cohort_records)

    metadata = {
        "dataset_path": str(dataset_path),
        "sample_path": str(sample_path),
        "global_start_year": global_start_year,
        "global_end_year": global_end_year,
        "recommended_windows": windows,
        "supported_event_windows": supported_windows,
        "min_pre_periods": min_pre_periods,
        "min_post_periods": min_post_periods,
        "winsor_quantiles": winsor_quantiles,
        "heterogeneity_cutoffs": heterogeneity_meta,
    }

    logger.info("Writing cleaned event-study sample to %s", sample_path)
    write_parquet(df, sample_path, args.coalesce)
    write_pandas_csv(support_frame, diagnostics_dir / "event_time_support_by_treatment.csv")
    write_pandas_csv(cohort_summary, tables_dir / "cohort_summary.csv")
    write_json(metadata, diagnostics_dir / "sample_metadata.json")
    write_restriction_outputs(
        restriction_records,
        diagnostics_dir / "sample_restrictions.csv",
        diagnostics_dir / "sample_restrictions.md",
        "Revelio Event-Study Sample Restrictions",
    )

    logger.info("Sample build complete. Outputs written to %s", sample_path)
    spark.stop()


if __name__ == "__main__":
    main()
