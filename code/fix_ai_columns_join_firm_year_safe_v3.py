#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

from revelio_people_analytics_utils import (
    build_company_lookup,
    parquet_reader,
    safe_divide,
    slug_expr,
    ensure_directory,
)

OBS_START_YEAR = 1950

AI_STRICT_REGEX = (
    r"\b(?:artificial intelligence|generative artificial intelligence|generative ai|genai|"
    r"responsible ai|trustworthy ai|ai engineer|ai scientist|ai researcher|ai research scientist|"
    r"ai specialist|ai architect|ai developer|ai product manager|ai platform|ai systems|"
    r"ai infrastructure|ai strategy|ai solutions|ai applications|ai governance|ai ethics|"
    r"machine learning|ml engineer|machine learning engineer|machine learning scientist|"
    r"machine learning researcher|applied machine learning|ml scientist|ml researcher|ml ops|mlops|"
    r"machine learning operations|ml platform|ml infrastructure|ml systems|ml model|"
    r"model training|model deployment|model evaluation|model monitoring|model validation|"
    r"deep learning|neural network|neural networks|deep neural network|dnn|"
    r"convolutional neural network|cnn|recurrent neural network|rnn|transformer model|foundation model|"
    r"large language model|llm|llm engineer|llm researcher|language model|"
    r"natural language processing|nlp engineer|nlp scientist|computational linguist|"
    r"text mining|text analytics|speech recognition|speech ai|conversational ai|chatbot|chat bot|"
    r"computer vision|vision engineer|vision scientist|image recognition|object detection|visual recognition|video analytics|"
    r"reinforcement learning|rl engineer|rl researcher|bandit algorithms|contextual bandit|"
    r"recommendation algorithms|recommender systems|personalization algorithms|"
    r"predictive modeling|predictive analytics|predictive algorithms|prediction model|"
    r"algorithmic decision|automated decision|decision intelligence|decision science|"
    r"data science machine learning|statistical learning|robotics ai|autonomous systems|"
    r"autonomous driving|perception engineer|autonomy engineer|planning and control engineer)\b"
)

AI_TOKEN_ROLE_REGEX = (
    r"\b(?:engineer|scientist|researcher|specialist|architect|developer|product manager|"
    r"platform|systems|infrastructure|strategy|solutions|applications|governance|ethics)\b"
)

AI_BROAD_ANCHOR_REGEX = (
    r"\b(?:data scientist|data science|advanced analytics|analytics scientist|algorithmic|"
    r"predictive|prediction|modeling|modelling)\b"
)

AI_BROAD_CONTEXT_REGEX = (
    r"\b(?:model|models|predict|prediction|predictive|algorithm|algorithms|machine learning|ml|ai|"
    r"artificial intelligence)\b"
)


def parse_args():
    p = argparse.ArgumentParser(description="Fix AI columns in safe_v3 firm-year panel without rebuilding the full firm-year pipeline.")
    p.add_argument("--project-root", default="/labs/khanna/predictive_capital/revelio_people_analytics")
    p.add_argument("--company-ref-dir", default="/labs/bharadwajlab/linkedin/company_ref")
    p.add_argument("--positions-dir", default="/labs/bharadwajlab/linkedin/individual_position")
    p.add_argument("--postings-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/intermediate/postings_extracted")
    p.add_argument("--base-firm-year-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--diagnostics-dir", default=None)
    p.add_argument("--analysis-end-year", type=int, default=2023)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "24")))
    p.add_argument("--shuffle-partitions", type=int, default=3200)
    p.add_argument("--coalesce", type=int, default=800)
    p.add_argument("--tmpdir", default=None)
    return p.parse_args()


def create_spark(args) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("fix_ai_columns_join_firm_year_safe_v3")
        .master(f"local[{args.threads}]")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.default.parallelism", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "LEGACY")
        .config("spark.sql.parquet.int96RebaseModeInRead", "LEGACY")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "LEGACY")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "LEGACY")
    )
    if args.tmpdir:
        ensure_directory(args.tmpdir)
        ensure_directory(os.path.join(args.tmpdir, "warehouse"))
        builder = (
            builder
            .config("spark.local.dir", args.tmpdir)
            .config("spark.sql.warehouse.dir", os.path.join(args.tmpdir, "warehouse"))
            .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={args.tmpdir}")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def norm_expr(cols: list[str]) -> F.Column:
    pieces = [F.coalesce(F.col(c).cast("string"), F.lit("")) for c in cols]
    if not pieces:
        return F.lit("")
    return F.trim(
        F.regexp_replace(
            F.lower(F.regexp_replace(F.concat_ws(" ", *pieces), r"[^a-z0-9]+", " ")),
            r"\s+",
            " ",
        )
    )


def add_ai_flags(df: DataFrame, title_cols: list[str], desc_cols: list[str]) -> DataFrame:
    title_cols = [c for c in title_cols if c in df.columns]
    desc_cols = [c for c in desc_cols if c in df.columns]

    title_text = norm_expr(title_cols)
    desc_text = norm_expr(desc_cols)
    all_text = F.concat_ws(" ", title_text, desc_text)

    title_ai_token_role = title_text.rlike(r"\bai\b") & title_text.rlike(AI_TOKEN_ROLE_REGEX)
    desc_ai_token_role = desc_text.rlike(r"\bai\b") & desc_text.rlike(AI_TOKEN_ROLE_REGEX)

    strict_title = title_text.rlike(AI_STRICT_REGEX) | title_ai_token_role
    strict_desc = desc_text.rlike(AI_STRICT_REGEX) | desc_ai_token_role

    broad_title = strict_title | (
        title_text.rlike(AI_BROAD_ANCHOR_REGEX) & all_text.rlike(AI_BROAD_CONTEXT_REGEX)
    )
    broad_desc = strict_desc | (
        desc_text.rlike(AI_BROAD_ANCHOR_REGEX) & all_text.rlike(AI_BROAD_CONTEXT_REGEX)
    )

    return (
        df.withColumn("ai_title_strict", F.when(strict_title, F.lit(1)).otherwise(F.lit(0)))
          .withColumn("ai_description_strict", F.when(strict_desc, F.lit(1)).otherwise(F.lit(0)))
          .withColumn("ai_any_strict", F.greatest(F.col("ai_title_strict"), F.col("ai_description_strict")))
          .withColumn("ai_title_broad", F.when(broad_title, F.lit(1)).otherwise(F.lit(0)))
          .withColumn("ai_description_broad", F.when(broad_desc, F.lit(1)).otherwise(F.lit(0)))
          .withColumn("ai_any_broad", F.greatest(F.col("ai_title_broad"), F.col("ai_description_broad")))
    )


def add_firm_key_from_rcid(df: DataFrame, entity_lookup: DataFrame, is_posting: bool) -> DataFrame:
    for c in ["rcid", "ultimate_parent_rcid"]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("long"))

    lookup = entity_lookup.select(
        F.col("entity_rcid").alias("lookup_entity_rcid"),
        F.col("ultimate_parent_rcid").alias("lookup_parent_rcid"),
        F.col("parent_company_name").alias("lookup_parent_company_name"),
        F.col("entity_company_name").alias("lookup_entity_company_name"),
    )

    df = df.join(lookup, df["rcid"] == lookup["lookup_entity_rcid"], how="left")
    df = df.withColumn("parent_rcid", F.coalesce(F.col("ultimate_parent_rcid"), F.col("lookup_parent_rcid")))

    if is_posting:
        df = df.withColumn(
            "firm_name",
            F.coalesce(
                F.col("ultimate_parent_company_name") if "ultimate_parent_company_name" in df.columns else F.lit(None),
                F.col("lookup_parent_company_name"),
                F.col("company") if "company" in df.columns else F.lit(None),
            ),
        )
        id_col = "job_id"
    else:
        df = df.withColumn(
            "firm_name",
            F.coalesce(
                F.col("ultimate_parent_company_name") if "ultimate_parent_company_name" in df.columns else F.lit(None),
                F.col("lookup_parent_company_name"),
                F.col("company_name") if "company_name" in df.columns else F.lit(None),
                F.col("lookup_entity_company_name"),
                F.col("company_cleaned") if "company_cleaned" in df.columns else F.lit(None),
                F.col("company_raw") if "company_raw" in df.columns else F.lit(None),
            ),
        )
        id_col = "position_id"

    df = df.withColumn(
        "firm_key",
        F.when(F.col("parent_rcid").isNotNull(), F.concat(F.lit("parent_"), F.col("parent_rcid").cast("string")))
        .when(F.col("firm_name").isNotNull(), F.concat(F.lit("name_"), slug_expr("firm_name")))
        .otherwise(F.concat(F.lit("unknown_ai_"), F.col(id_col).cast("string"))),
    )

    return df.drop("lookup_entity_rcid")


def min_two_dates(left: str, right: str) -> F.Column:
    return F.coalesce(F.least(F.col(left), F.col(right)), F.col(left), F.col(right))


def add_first_year_cols(df: DataFrame) -> DataFrame:
    for c in df.columns:
        if c.startswith("first_ai_") and "_date" in c:
            y = c.replace("_date_", "_year_").replace("_date", "_year")
            df = df.withColumn(y, F.year(F.col(c)))
    return df


def build_position_ai(spark: SparkSession, args, entity_lookup: DataFrame) -> tuple[DataFrame, DataFrame]:
    pos = parquet_reader(spark, args.positions_dir).dropDuplicates(["position_id"])
    keep = [
        "position_id", "user_id", "rcid", "ultimate_parent_rcid", "startdate", "enddate", "weight",
        "jobtitle_raw", "description", "company_raw", "company_cleaned", "company_name",
        "ultimate_parent_company_name",
    ]
    pos = pos.select(*[c for c in keep if c in pos.columns])
    pos = add_firm_key_from_rcid(pos, entity_lookup, is_posting=False)
    pos = add_ai_flags(pos, ["jobtitle_raw"], ["description"])

    pos = (
        pos.withColumn("start_date", F.to_date("startdate"))
           .withColumn("end_date", F.to_date("enddate"))
           .withColumn("start_year_obs", F.year("start_date"))
           .withColumn("end_year_obs", F.year("end_date"))
           .withColumn("panel_start_year", F.greatest(F.coalesce(F.col("start_year_obs"), F.col("end_year_obs"), F.lit(OBS_START_YEAR)), F.lit(OBS_START_YEAR)))
           .withColumn("panel_end_year", F.least(F.coalesce(F.col("end_year_obs"), F.lit(args.analysis_end_year)), F.lit(args.analysis_end_year)))
           .withColumn("panel_start_year", F.when(F.col("panel_start_year") > F.col("panel_end_year"), F.col("panel_end_year")).otherwise(F.col("panel_start_year")))
           .withColumn("weight", F.greatest(F.coalesce(F.col("weight").cast("double"), F.lit(1.0)), F.lit(0.0)))
    )

    expanded = pos.withColumn("year", F.explode(F.sequence(F.col("panel_start_year"), F.col("panel_end_year")))).withColumn("year", F.col("year").cast("int"))

    agg = (
        expanded.groupBy("firm_key", "year")
        .agg(
            F.first("parent_rcid", ignorenulls=True).alias("parent_rcid_ai_pos"),
            F.first("firm_name", ignorenulls=True).alias("firm_name_ai_pos"),
            F.sum(F.col("weight") * F.col("ai_title_strict")).alias("ai_positions_title_strict_weighted"),
            F.sum(F.col("weight") * F.col("ai_description_strict")).alias("ai_positions_description_strict_weighted"),
            F.sum(F.col("weight") * F.col("ai_any_strict")).alias("ai_positions_any_strict_weighted"),
            F.sum(F.col("weight") * F.col("ai_title_broad")).alias("ai_positions_title_broad_weighted"),
            F.sum(F.col("weight") * F.col("ai_description_broad")).alias("ai_positions_description_broad_weighted"),
            F.sum(F.col("weight") * F.col("ai_any_broad")).alias("ai_positions_any_broad_weighted"),
        )
    )

    dates = (
        pos.groupBy("firm_key")
        .agg(
            F.min(F.when(F.col("ai_title_strict") == 1, F.col("start_date"))).alias("first_ai_position_date_title_strict"),
            F.min(F.when(F.col("ai_description_strict") == 1, F.col("start_date"))).alias("first_ai_position_date_description_strict"),
            F.min(F.when(F.col("ai_any_strict") == 1, F.col("start_date"))).alias("first_ai_position_date_any_strict"),
            F.min(F.when(F.col("ai_title_broad") == 1, F.col("start_date"))).alias("first_ai_position_date_title_broad"),
            F.min(F.when(F.col("ai_description_broad") == 1, F.col("start_date"))).alias("first_ai_position_date_description_broad"),
            F.min(F.when(F.col("ai_any_broad") == 1, F.col("start_date"))).alias("first_ai_position_date_any_broad"),
        )
    )

    detail = (
        pos.where(F.col("ai_any_broad") == 1)
        .select(*[c for c in ["position_id", "firm_key", "parent_rcid", "firm_name", "start_date", "jobtitle_raw", "description", "ai_any_strict", "ai_any_broad"] if c in pos.columns])
    )

    return agg, dates, detail


def build_posting_ai(spark: SparkSession, args, entity_lookup: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    post = parquet_reader(spark, args.postings_dir).dropDuplicates(["job_id"])
    keep = [
        "job_id", "rcid", "ultimate_parent_rcid", "post_date",
        "jobtitle_raw", "jobtitle", "jobtitle_translated", "description",
        "company", "ultimate_parent_company_name",
    ]
    post = post.select(*[c for c in keep if c in post.columns])
    post = add_firm_key_from_rcid(post, entity_lookup, is_posting=True)
    post = add_ai_flags(post, ["jobtitle_raw", "jobtitle", "jobtitle_translated"], ["description"])
    post = post.withColumn("post_date", F.to_date("post_date")).withColumn("year", F.year("post_date").cast("int"))

    agg = (
        post.where((F.col("year") >= OBS_START_YEAR) & (F.col("year") <= args.analysis_end_year))
        .groupBy("firm_key", "year")
        .agg(
            F.first("parent_rcid", ignorenulls=True).alias("parent_rcid_ai_post"),
            F.first("firm_name", ignorenulls=True).alias("firm_name_ai_post"),
            F.count(F.lit(1)).alias("posting_count_for_ai_flags"),
            F.sum("ai_title_strict").alias("ai_postings_title_strict"),
            F.sum("ai_description_strict").alias("ai_postings_description_strict"),
            F.sum("ai_any_strict").alias("ai_postings_any_strict"),
            F.sum("ai_title_broad").alias("ai_postings_title_broad"),
            F.sum("ai_description_broad").alias("ai_postings_description_broad"),
            F.sum("ai_any_broad").alias("ai_postings_any_broad"),
        )
    )

    dates = (
        post.groupBy("firm_key")
        .agg(
            F.min(F.when(F.col("ai_title_strict") == 1, F.col("post_date"))).alias("first_ai_posting_date_title_strict"),
            F.min(F.when(F.col("ai_description_strict") == 1, F.col("post_date"))).alias("first_ai_posting_date_description_strict"),
            F.min(F.when(F.col("ai_any_strict") == 1, F.col("post_date"))).alias("first_ai_posting_date_any_strict"),
            F.min(F.when(F.col("ai_title_broad") == 1, F.col("post_date"))).alias("first_ai_posting_date_title_broad"),
            F.min(F.when(F.col("ai_description_broad") == 1, F.col("post_date"))).alias("first_ai_posting_date_description_broad"),
            F.min(F.when(F.col("ai_any_broad") == 1, F.col("post_date"))).alias("first_ai_posting_date_any_broad"),
        )
    )

    detail = (
        post.where(F.col("ai_any_broad") == 1)
        .select(*[c for c in ["job_id", "firm_key", "parent_rcid", "firm_name", "post_date", "jobtitle_raw", "jobtitle", "jobtitle_translated", "description", "ai_any_strict", "ai_any_broad"] if c in post.columns])
    )

    return agg, dates, detail


def main():
    args = parse_args()
    project = args.project_root
    base_dir = args.base_firm_year_dir or f"{project}/processed/final/firm_year_panel_ai_hr_manager_safe_v3"
    out_dir = args.out_dir or f"{project}/processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed"
    diag_dir = args.diagnostics_dir or f"{project}/processed/diagnostics/firm_year_ai_fix_safe_v3"

    ensure_directory(out_dir)
    ensure_directory(diag_dir)

    spark = create_spark(args)

    print("[1/7] Building company lookup")
    entity_lookup, _ = build_company_lookup(spark, args.company_ref_dir)

    print("[2/7] Building corrected AI position aggregates")
    pos_agg, pos_dates, pos_detail = build_position_ai(spark, args, entity_lookup)

    print("[3/7] Building corrected AI posting aggregates")
    post_agg, post_dates, post_detail = build_posting_ai(spark, args, entity_lookup)

    print("[4/7] Combining AI add-on")
    ai_year = pos_agg.join(post_agg, on=["firm_key", "year"], how="outer")
    ai_dates = pos_dates.join(post_dates, on="firm_key", how="outer")
    ai_dates = (
        ai_dates
        .withColumn("first_ai_firm_date_any_strict", min_two_dates("first_ai_position_date_any_strict", "first_ai_posting_date_any_strict"))
        .withColumn("first_ai_firm_date_any_broad", min_two_dates("first_ai_position_date_any_broad", "first_ai_posting_date_any_broad"))
    )
    ai_dates = add_first_year_cols(ai_dates)

    ai = ai_year.join(ai_dates, on="firm_key", how="left")

    print("[5/7] Reading base firm-year and replacing AI columns")
    base = spark.read.parquet(base_dir)

    ai_cols_to_drop = [
        c for c in base.columns
        if c.startswith("ai_")
        or c.startswith("first_ai_")
        or c.startswith("is_first_ai_")
        or c.startswith("has_ai_")
        or c == "posting_count_for_ai_flags"
    ]
    base_clean = base.drop(*ai_cols_to_drop)

    joined = base_clean.join(ai, on=["firm_key", "year"], how="left")

    zero_cols = [
        "ai_positions_title_strict_weighted", "ai_positions_description_strict_weighted", "ai_positions_any_strict_weighted",
        "ai_positions_title_broad_weighted", "ai_positions_description_broad_weighted", "ai_positions_any_broad_weighted",
        "posting_count_for_ai_flags",
        "ai_postings_title_strict", "ai_postings_description_strict", "ai_postings_any_strict",
        "ai_postings_title_broad", "ai_postings_description_broad", "ai_postings_any_broad",
    ]
    for c in zero_cols:
        if c in joined.columns:
            joined = joined.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

    joined = (
        joined
        .withColumn("ai_positions_any_strict_share", safe_divide(F.col("ai_positions_any_strict_weighted"), F.col("n_employees")))
        .withColumn("ai_positions_any_broad_share", safe_divide(F.col("ai_positions_any_broad_weighted"), F.col("n_employees")))
        .withColumn("ai_postings_any_strict_share", safe_divide(F.col("ai_postings_any_strict"), F.col("posting_count_for_ai_flags")))
        .withColumn("ai_postings_any_broad_share", safe_divide(F.col("ai_postings_any_broad"), F.col("posting_count_for_ai_flags")))
        .withColumn("ai_position_log1p", F.log1p(F.coalesce(F.col("ai_positions_any_strict_weighted"), F.lit(0.0))))
        .withColumn("ai_posting_log1p", F.log1p(F.coalesce(F.col("ai_postings_any_strict"), F.lit(0.0))))
    )

    for variant in ["strict", "broad"]:
        for source in ["position", "posting", "firm"]:
            first_year = f"first_ai_{source}_year_any_{variant}"
            joined = joined.withColumn(
                f"is_first_ai_{source}_year_any_{variant}",
                F.when(F.col("year") == F.col(first_year), F.lit(1)).otherwise(F.lit(0)),
            )
            joined = joined.withColumn(
                f"has_ai_{source}_any_{variant}_by_year",
                F.when(F.col(first_year).isNotNull() & (F.col("year") >= F.col(first_year)), F.lit(1)).otherwise(F.lit(0)),
            )

    print("[6/7] Writing corrected firm-year panel")
    joined.coalesce(args.coalesce).write.mode("overwrite").option("compression", "snappy").parquet(out_dir)

    print("[7/7] Writing diagnostics")
    pos_detail.coalesce(1).write.mode("overwrite").option("compression", "snappy").parquet(os.path.join(diag_dir, "ai_matched_positions_fixed"))
    post_detail.coalesce(1).write.mode("overwrite").option("compression", "snappy").parquet(os.path.join(diag_dir, "ai_matched_postings_fixed"))

    written = spark.read.parquet(out_dir)
    print("[INFO] corrected rows:", written.count())
    print("[INFO] strict AI adopter firms:",
          written.where(F.col("first_ai_firm_year_any_strict").isNotNull()).select("firm_key").distinct().count())
    print("[INFO] corrected output:", out_dir)

    spark.stop()


if __name__ == "__main__":
    main()
