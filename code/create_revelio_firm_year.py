#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

from pyspark.sql import functions as F

from revelio_people_analytics_utils import (
    DEFAULT_RAW_PATHS,
    add_first_event_years,
    attach_parent_static,
    build_company_lookup,
    build_default_paths,
    build_position_firm_year,
    build_posting_firm_year,
    build_user_features,
    choose_primary_position,
    create_spark,
    detect_analysis_end_year,
    ensure_directory,
    extract_postings_if_needed,
    outer_join_all,
    prepare_positions,
    prepare_postings,
)


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Build Revelio people-analytics firm-year panel.")
    parser.add_argument("--project-root", default=defaults["project_root"])
    parser.add_argument("--company-ref-dir", default=DEFAULT_RAW_PATHS["company_ref"])
    parser.add_argument("--education-dir", default=DEFAULT_RAW_PATHS["education"])
    parser.add_argument("--positions-dir", default=DEFAULT_RAW_PATHS["position"])
    parser.add_argument("--skills-dir", default=DEFAULT_RAW_PATHS["skill"])
    parser.add_argument("--users-dir", default=DEFAULT_RAW_PATHS["user"])
    parser.add_argument("--postings-path", default=DEFAULT_RAW_PATHS["postings"])
    parser.add_argument("--postings-extract-dir", default=None)
    parser.add_argument("--intermediate-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    parser.add_argument("--shuffle-partitions", type=int, default=1200)
    parser.add_argument("--coalesce", type=int, default=200)
    parser.add_argument("--tmpdir", default=None)
    return parser.parse_args()


def min_two_dates(left: str, right: str) -> F.Column:
    return F.coalesce(F.least(F.col(left), F.col(right)), F.col(left), F.col(right))


def write_parquet(df, path: str, coalesce: int | None = None) -> None:
    writer = df
    if coalesce is not None and coalesce > 0:
        writer = writer.coalesce(max(1, coalesce))
    (
        writer.write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(path)
    )


def ensure_columns(df, columns: list[str]) -> object:
    for name in columns:
        if name not in df.columns:
            df = df.withColumn(name, F.lit(None))
    return df


def main() -> None:
    args = parse_args()
    paths = build_default_paths(args.project_root)
    intermediate_dir = args.intermediate_dir or paths["firm_year_intermediate"]
    out_dir = args.out_dir or paths["firm_year_output"]
    postings_extract_dir = args.postings_extract_dir or os.path.join(paths["intermediate_root"], "postings_extracted")

    for path in [args.project_root, paths["processed_root"], paths["intermediate_root"], paths["final_root"], intermediate_dir, out_dir]:
        ensure_directory(path)

    spark = create_spark(
        app_name="revelio_people_analytics_firm_year",
        threads=args.threads,
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    print("[1/8] Resolving postings input")
    resolved_postings = extract_postings_if_needed(args.postings_path, postings_extract_dir)

    print("[2/8] Detecting analysis horizon")
    analysis_end_year = detect_analysis_end_year(args.positions_dir, resolved_postings, args.users_dir, spark)
    print(f"[INFO] Analysis end year: {analysis_end_year}")

    print("[3/8] Building company and user features")
    entity_lookup, parent_static = build_company_lookup(spark, args.company_ref_dir)
    user_features = build_user_features(spark, args.users_dir, args.education_dir, args.skills_dir)

    print("[4/8] Preparing positions and aggregating to firm-year")
    positions = prepare_positions(spark, args.positions_dir, user_features, entity_lookup, analysis_end_year)
    position_firm_year, position_signal_dates, flagged_positions = build_position_firm_year(positions)

    print("[5/8] Preparing postings and aggregating to firm-year")
    postings = prepare_postings(spark, resolved_postings, entity_lookup)
    posting_firm_year, posting_signal_dates, flagged_postings = build_posting_firm_year(postings)

    print("[6/8] Building final firm-year panel")
    position_panel = position_firm_year.withColumnRenamed("parent_rcid", "parent_rcid_pos").withColumnRenamed("firm_name", "firm_name_pos")
    posting_panel = posting_firm_year.withColumnRenamed("parent_rcid", "parent_rcid_post").withColumnRenamed("firm_name", "firm_name_post")
    panel = position_panel.join(posting_panel, on=["firm_key", "year"], how="outer")
    panel = panel.withColumn("parent_rcid", F.coalesce(F.col("parent_rcid_pos"), F.col("parent_rcid_post")))
    panel = panel.withColumn("firm_name", F.coalesce(F.col("firm_name_pos"), F.col("firm_name_post")))
    panel = panel.drop("parent_rcid_pos", "parent_rcid_post", "firm_name_pos", "firm_name_post")
    panel = attach_parent_static(panel, parent_static)

    signal_dates = outer_join_all([position_signal_dates, posting_signal_dates], on=["firm_key"])
    if signal_dates is not None:
        date_columns = [c for c in signal_dates.columns if c != "firm_key" and "_date" in c]
        signal_dates = add_first_event_years(signal_dates, date_columns)
        signal_dates = signal_dates.withColumn(
            "first_people_analytics_firm_date_any_study",
            min_two_dates("first_people_analytics_position_date_any_study", "first_people_analytics_posting_date_any_study"),
        )
        signal_dates = signal_dates.withColumn(
            "first_people_analytics_firm_date_any_enriched",
            min_two_dates("first_people_analytics_position_date_any_enriched", "first_people_analytics_posting_date_any_enriched"),
        )
        signal_dates = add_first_event_years(
            signal_dates,
            ["first_people_analytics_firm_date_any_study", "first_people_analytics_firm_date_any_enriched"],
        )
        panel = panel.join(signal_dates, on="firm_key", how="left")

    panel = ensure_columns(
        panel,
        [
            "first_people_analytics_position_year_any_enriched",
            "first_people_analytics_posting_year_any_enriched",
            "first_people_analytics_firm_year_any_enriched",
        ],
    )

    panel = panel.withColumn(
        "is_first_people_analytics_position_year_any_enriched",
        F.when(F.col("year") == F.col("first_people_analytics_position_year_any_enriched"), F.lit(1)).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "is_first_people_analytics_posting_year_any_enriched",
        F.when(F.col("year") == F.col("first_people_analytics_posting_year_any_enriched"), F.lit(1)).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "is_first_people_analytics_firm_year_any_enriched",
        F.when(F.col("year") == F.col("first_people_analytics_firm_year_any_enriched"), F.lit(1)).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "has_people_analytics_position_any_enriched_by_year",
        F.when(
            F.col("first_people_analytics_position_year_any_enriched").isNotNull()
            & (F.col("year") >= F.col("first_people_analytics_position_year_any_enriched")),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "has_people_analytics_posting_any_enriched_by_year",
        F.when(
            F.col("first_people_analytics_posting_year_any_enriched").isNotNull()
            & (F.col("year") >= F.col("first_people_analytics_posting_year_any_enriched")),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "has_people_analytics_firm_any_enriched_by_year",
        F.when(
            F.col("first_people_analytics_firm_year_any_enriched").isNotNull()
            & (F.col("year") >= F.col("first_people_analytics_firm_year_any_enriched")),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    panel = panel.withColumn("has_position_data", F.when(F.col("workforce_weighted").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    panel = panel.withColumn("has_posting_data", F.when(F.col("posting_count").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    panel = panel.withColumn("parent_rcid_matched", F.when(F.col("parent_rcid").isNotNull(), F.lit(1)).otherwise(F.lit(0)))

    print("[7/8] Writing final firm-year panel only")
    write_parquet(panel, out_dir, args.coalesce)

    print("[8/8] Quick checks from written output")
    written = spark.read.parquet(out_dir)
    firm_rows = written.count()
    firms = written.select("firm_key").distinct().count()
    adopting_firms = (
        written.where(F.col("first_people_analytics_firm_year_any_enriched").isNotNull())
        .select("firm_key")
        .distinct()
        .count()
    )
    print(f"[INFO] Firm-year rows: {firm_rows:,}")
    print(f"[INFO] Firms: {firms:,}")
    print(f"[INFO] Firms with any enriched people-analytics adoption: {adopting_firms:,}")

    spark.stop()


if __name__ == "__main__":
    main()
