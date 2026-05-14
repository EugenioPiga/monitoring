#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from revelio_people_analytics_utils import (
    DEFAULT_RAW_PATHS,
    add_first_event_years,
    attach_parent_static,
    build_company_lookup,
    build_default_paths,
    build_first_event_dates,
    build_position_firm_year,
    build_posting_firm_year,
    build_user_features,
    create_spark,
    detect_analysis_end_year,
    ensure_directory,
    expand_positions_to_years,
    extract_postings_if_needed,
    normalize_text_expr,
    outer_join_all,
    prepare_positions,
    prepare_postings,
    safe_divide,
)

# -----------------------------------------------------------------------------
# Role dictionaries. These are intentionally defined here rather than replacing
# existing people-analytics code. The script keeps the existing PA pipeline intact
# and appends AI / broad HR / manager measures to the same firm-year output.
# -----------------------------------------------------------------------------
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
    r"large language model|large language models|llm|llm engineer|llm researcher|language model|"
    r"natural language processing|nlp engineer|nlp scientist|computational linguist|text mining|"
    r"text analytics|speech recognition|speech ai|conversational ai|chatbot|chat bot|"
    r"computer vision|vision engineer|vision scientist|image recognition|object detection|"
    r"visual recognition|video analytics|reinforcement learning|rl engineer|rl researcher|"
    r"bandit algorithms|contextual bandit|recommendation algorithms|recommender systems|"
    r"personalization algorithms|predictive modeling|predictive analytics|predictive algorithms|"
    r"prediction model|algorithmic decision|automated decision|decision intelligence|"
    r"decision science|data science machine learning|statistical learning|robotics ai|"
    r"autonomous systems|autonomous driving|perception engineer|autonomy engineer|"
    r"planning and control engineer)\b"
)

AI_TOKEN_ROLE_REGEX = (
    r"\b(?:engineer|scientist|researcher|architect|developer|product manager|platform|systems|"
    r"infrastructure|strategy|solutions|applications|governance|ethics|responsible|trustworthy|"
    r"generative|research|model|models|automation|algorithm|algorithms)\b"
)

AI_BROAD_ANCHOR_REGEX = r"\b(?:data scientist|data science|advanced analytics|analytics scientist|algorithmic|predictive|prediction|modeling|modelling)\b"
AI_BROAD_CONTEXT_REGEX = r"\b(?:model|models|predict|prediction|predictive|algorithm|algorithms|machine learning|ml|ai|artificial intelligence)\b"

HR_GENERAL_REGEX = (
    r"\b(?:human resources|hr|people operations|people ops|talent acquisition|recruiter|recruiting|"
    r"compensation and benefits|employee relations|workforce planning|organizational development|"
    r"organisation development|learning and development|hr business partner|hrbp|talent management|"
    r"personnel|labor relations|labour relations|benefits|payroll|hris|human capital management|hcm)\b"
)

MANAGER_TITLE_REGEX = (
    r"\b(?:senior manager|manager|director|senior director|head of|lead|supervisor|vp|vice president|"
    r"chief|officer|executive|principal|managing director|general manager|team lead|department head)\b"
)

NON_MANAGER_FALSE_POSITIVE_REGEX = r"\b(?:account manager|customer success manager|sales manager|product manager|project manager)\b"


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Build firm-year panel with PA + AI + HR/manager measures.")
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
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    parser.add_argument("--shuffle-partitions", type=int, default=1200)
    parser.add_argument("--coalesce", type=int, default=200)
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--manager-seniority-threshold", type=float, default=4.0)
    parser.add_argument(
        "--analysis-end-year",
        type=int,
        default=2023,
        help="Hard cap for the position-year expansion. Prevents future-dated records from expanding the panel to years like 2029.",
    )
    return parser.parse_args()


def write_parquet(df: DataFrame, path: str, coalesce: int | None = None) -> None:
    writer = df
    if coalesce is not None and coalesce > 0:
        writer = writer.coalesce(max(1, coalesce))
    writer.write.mode("overwrite").option("compression", "snappy").parquet(path)


def ensure_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    for name in columns:
        if name not in df.columns:
            df = df.withColumn(name, F.lit(None))
    return df


def min_two_dates(left: str, right: str) -> F.Column:
    return F.coalesce(F.least(F.col(left), F.col(right)), F.col(left), F.col(right))


def add_ai_hr_manager_flags(
    frame: DataFrame,
    title_columns: list[str],
    description_columns: list[str],
    role_columns: list[str] | None = None,
    seniority_column: str | None = "seniority",
    manager_seniority_threshold: float = 4.0,
) -> DataFrame:
    role_columns = role_columns or []
    title_text = normalize_text_expr(*title_columns)
    description_text = normalize_text_expr(*description_columns)
    role_text = normalize_text_expr(*role_columns)
    all_text = F.concat_ws(" ", title_text, description_text, role_text)
    title_role_text = F.concat_ws(" ", title_text, role_text)

    frame = frame.withColumn("ai_title_text", title_text)
    frame = frame.withColumn("ai_description_text", description_text)
    frame = frame.withColumn("ai_role_text", role_text)

    title_ai_token_role = title_text.rlike(r"\bai\b") & title_text.rlike(AI_TOKEN_ROLE_REGEX)
    desc_ai_token_role = description_text.rlike(r"\bai\b") & description_text.rlike(AI_TOKEN_ROLE_REGEX)

    strict_title = title_text.rlike(AI_STRICT_REGEX) | title_ai_token_role
    strict_desc = description_text.rlike(AI_STRICT_REGEX) | desc_ai_token_role
    broad_title = strict_title | (title_text.rlike(AI_BROAD_ANCHOR_REGEX) & all_text.rlike(AI_BROAD_CONTEXT_REGEX))
    broad_desc = strict_desc | (description_text.rlike(AI_BROAD_ANCHOR_REGEX) & all_text.rlike(AI_BROAD_CONTEXT_REGEX))

    frame = frame.withColumn("ai_title_strict", F.when(strict_title, F.lit(1)).otherwise(F.lit(0)))
    frame = frame.withColumn("ai_description_strict", F.when(strict_desc, F.lit(1)).otherwise(F.lit(0)))
    frame = frame.withColumn("ai_any_strict", F.greatest(F.col("ai_title_strict"), F.col("ai_description_strict")))
    frame = frame.withColumn("ai_title_broad", F.when(broad_title, F.lit(1)).otherwise(F.lit(0)))
    frame = frame.withColumn("ai_description_broad", F.when(broad_desc, F.lit(1)).otherwise(F.lit(0)))
    frame = frame.withColumn("ai_any_broad", F.greatest(F.col("ai_title_broad"), F.col("ai_description_broad")))

    frame = frame.withColumn("is_hr_position", F.when(title_role_text.rlike(HR_GENERAL_REGEX), F.lit(1)).otherwise(F.lit(0)))

    title_manager = title_text.rlike(MANAGER_TITLE_REGEX) & (~title_text.rlike(NON_MANAGER_FALSE_POSITIVE_REGEX))
    if seniority_column and seniority_column in frame.columns:
        structured_manager = F.col(seniority_column).cast("double") >= F.lit(float(manager_seniority_threshold))
        manager_cond = structured_manager | title_manager
    else:
        manager_cond = title_manager
    frame = frame.withColumn("is_manager_position", F.when(manager_cond, F.lit(1)).otherwise(F.lit(0)))
    return frame


def weighted_flag_sum(flag: str, weight_col: str = "weight") -> F.Column:
    return F.sum(F.coalesce(F.col(flag).cast("double"), F.lit(0.0)) * F.coalesce(F.col(weight_col).cast("double"), F.lit(1.0)))


def build_position_extra_firm_year(positions: DataFrame, manager_seniority_threshold: float) -> tuple[DataFrame, DataFrame | None, DataFrame]:
    positions = add_ai_hr_manager_flags(
        positions,
        title_columns=["jobtitle_raw"],
        description_columns=["description"],
        role_columns=["mapped_role", "role_k50", "role_k150", "onet_title", "job_category"],
        seniority_column="seniority",
        manager_seniority_threshold=manager_seniority_threshold,
    )
    expanded = expand_positions_to_years(positions)
    extra = (
        expanded.groupBy("firm_key", "year")
        .agg(
            F.first("parent_rcid", ignorenulls=True).alias("parent_rcid"),
            F.first("firm_name", ignorenulls=True).alias("firm_name"),
            F.sum("weight").alias("n_employees"),
            weighted_flag_sum("ai_title_strict").alias("ai_positions_title_strict_weighted"),
            weighted_flag_sum("ai_description_strict").alias("ai_positions_description_strict_weighted"),
            weighted_flag_sum("ai_any_strict").alias("ai_positions_any_strict_weighted"),
            weighted_flag_sum("ai_title_broad").alias("ai_positions_title_broad_weighted"),
            weighted_flag_sum("ai_description_broad").alias("ai_positions_description_broad_weighted"),
            weighted_flag_sum("ai_any_broad").alias("ai_positions_any_broad_weighted"),
            weighted_flag_sum("is_hr_position").alias("n_hr_positions"),
            weighted_flag_sum("is_manager_position").alias("n_managers"),
        )
        .withColumn("ai_positions_any_strict_share", safe_divide(F.col("ai_positions_any_strict_weighted"), F.col("n_employees")))
        .withColumn("ai_positions_any_broad_share", safe_divide(F.col("ai_positions_any_broad_weighted"), F.col("n_employees")))
        .withColumn("hr_to_employee_ratio", safe_divide(F.col("n_hr_positions"), F.col("n_employees")))
        .withColumn("managers_to_employee_ratio", safe_divide(F.col("n_managers"), F.col("n_employees")))
        .withColumn("ai_position_log1p", F.log1p(F.coalesce(F.col("ai_positions_any_strict_weighted"), F.lit(0.0))))
    )

    first_dates = build_first_event_dates(
        positions,
        "firm_key",
        "start_date",
        [
            ("ai_title_strict", "first_ai_position_date_title_strict"),
            ("ai_description_strict", "first_ai_position_date_description_strict"),
            ("ai_any_strict", "first_ai_position_date_any_strict"),
            ("ai_title_broad", "first_ai_position_date_title_broad"),
            ("ai_description_broad", "first_ai_position_date_description_broad"),
            ("ai_any_broad", "first_ai_position_date_any_broad"),
        ],
    )
    detail_cols = [
        "user_id", "position_id", "firm_key", "firm_name", "parent_rcid", "start_date", "panel_start_year",
        "jobtitle_raw", "description", "mapped_role", "role_k50", "role_k150", "job_category",
        "ai_title_strict", "ai_description_strict", "ai_any_strict", "ai_title_broad", "ai_description_broad", "ai_any_broad",
        "is_hr_position", "is_manager_position", "seniority",
    ]
    detail = positions.where(F.col("ai_any_broad") == 1).select(*[c for c in detail_cols if c in positions.columns])
    return extra, first_dates, detail


def build_posting_extra_firm_year(postings: DataFrame, manager_seniority_threshold: float) -> tuple[DataFrame, DataFrame | None, DataFrame]:
    postings = add_ai_hr_manager_flags(
        postings,
        title_columns=["jobtitle_raw", "jobtitle", "jobtitle_translated"],
        description_columns=["description"],
        role_columns=["mapped_role", "role_k50", "role_k150", "job_category"],
        seniority_column=None,
        manager_seniority_threshold=manager_seniority_threshold,
    )
    extra = (
        postings.groupBy("firm_key", "year")
        .agg(
            F.first("parent_rcid", ignorenulls=True).alias("parent_rcid"),
            F.first("firm_name", ignorenulls=True).alias("firm_name"),
            F.count(F.lit(1)).alias("posting_count_for_ai_flags"),
            F.sum("ai_title_strict").alias("ai_postings_title_strict"),
            F.sum("ai_description_strict").alias("ai_postings_description_strict"),
            F.sum("ai_any_strict").alias("ai_postings_any_strict"),
            F.sum("ai_title_broad").alias("ai_postings_title_broad"),
            F.sum("ai_description_broad").alias("ai_postings_description_broad"),
            F.sum("ai_any_broad").alias("ai_postings_any_broad"),
        )
        .withColumn("ai_postings_any_strict_share", safe_divide(F.col("ai_postings_any_strict"), F.col("posting_count_for_ai_flags")))
        .withColumn("ai_postings_any_broad_share", safe_divide(F.col("ai_postings_any_broad"), F.col("posting_count_for_ai_flags")))
        .withColumn("ai_posting_log1p", F.log1p(F.coalesce(F.col("ai_postings_any_strict"), F.lit(0.0))))
    )
    first_dates = build_first_event_dates(
        postings,
        "firm_key",
        "post_date",
        [
            ("ai_title_strict", "first_ai_posting_date_title_strict"),
            ("ai_description_strict", "first_ai_posting_date_description_strict"),
            ("ai_any_strict", "first_ai_posting_date_any_strict"),
            ("ai_title_broad", "first_ai_posting_date_title_broad"),
            ("ai_description_broad", "first_ai_posting_date_description_broad"),
            ("ai_any_broad", "first_ai_posting_date_any_broad"),
        ],
    )
    detail_cols = [
        "job_id", "firm_key", "firm_name", "parent_rcid", "post_date", "jobtitle_raw", "jobtitle", "jobtitle_translated",
        "description", "mapped_role", "role_k50", "role_k150", "job_category",
        "ai_title_strict", "ai_description_strict", "ai_any_strict", "ai_title_broad", "ai_description_broad", "ai_any_broad",
    ]
    detail = postings.where(F.col("ai_any_broad") == 1).select(*[c for c in detail_cols if c in postings.columns])
    return extra, first_dates, detail


def add_adoption_timing(panel: DataFrame, prefix: str, variants: list[str]) -> DataFrame:
    for variant in variants:
        pos_year = f"first_{prefix}_position_year_any_{variant}"
        post_year = f"first_{prefix}_posting_year_any_{variant}"
        firm_date = f"first_{prefix}_firm_date_any_{variant}"
        firm_year = f"first_{prefix}_firm_year_any_{variant}"
        panel = ensure_columns(panel, [pos_year, post_year, firm_year])
        panel = panel.withColumn(
            f"is_first_{prefix}_position_year_any_{variant}",
            F.when(F.col("year") == F.col(pos_year), F.lit(1)).otherwise(F.lit(0)),
        )
        panel = panel.withColumn(
            f"is_first_{prefix}_posting_year_any_{variant}",
            F.when(F.col("year") == F.col(post_year), F.lit(1)).otherwise(F.lit(0)),
        )
        panel = panel.withColumn(
            f"is_first_{prefix}_firm_year_any_{variant}",
            F.when(F.col("year") == F.col(firm_year), F.lit(1)).otherwise(F.lit(0)),
        )
        panel = panel.withColumn(
            f"has_{prefix}_position_any_{variant}_by_year",
            F.when(F.col(pos_year).isNotNull() & (F.col("year") >= F.col(pos_year)), F.lit(1)).otherwise(F.lit(0)),
        )
        panel = panel.withColumn(
            f"has_{prefix}_posting_any_{variant}_by_year",
            F.when(F.col(post_year).isNotNull() & (F.col("year") >= F.col(post_year)), F.lit(1)).otherwise(F.lit(0)),
        )
        panel = panel.withColumn(
            f"has_{prefix}_firm_any_{variant}_by_year",
            F.when(F.col(firm_year).isNotNull() & (F.col("year") >= F.col(firm_year)), F.lit(1)).otherwise(F.lit(0)),
        )
    return panel


def main() -> None:
    args = parse_args()
    paths = build_default_paths(args.project_root)
    intermediate_dir = args.intermediate_dir or paths["firm_year_intermediate"]
    out_dir = args.out_dir or paths["firm_year_output"]
    diag_dir = args.diagnostics_dir or os.path.join(paths["processed_root"], "diagnostics", "firm_year_ai_hr_manager")
    postings_extract_dir = args.postings_extract_dir or os.path.join(paths["intermediate_root"], "postings_extracted")

    for path in [args.project_root, paths["processed_root"], paths["intermediate_root"], paths["final_root"], intermediate_dir, out_dir, diag_dir]:
        ensure_directory(path)

    spark = create_spark(
        app_name="revelio_people_analytics_firm_year_ai_hr_manager",
        threads=args.threads,
        shuffle_partitions=args.shuffle_partitions,
        tmpdir=args.tmpdir,
    )

    print("[1/9] Resolving postings input")
    resolved_postings = extract_postings_if_needed(args.postings_path, postings_extract_dir)

    print("[2/9] Detecting analysis horizon")
    detected_analysis_end_year = detect_analysis_end_year(args.positions_dir, resolved_postings, args.users_dir, spark)
    analysis_end_year = min(int(detected_analysis_end_year), int(args.analysis_end_year))
    print(f"[INFO] Detected analysis end year: {detected_analysis_end_year}")
    print(f"[INFO] Capped analysis end year used for expansion: {analysis_end_year}")

    print("[3/9] Building company and user features")
    entity_lookup, parent_static = build_company_lookup(spark, args.company_ref_dir)
    user_features = build_user_features(spark, args.users_dir, args.education_dir, args.skills_dir)

    print("[4/9] Preparing positions, existing PA aggregates, and AI/HR/manager aggregates")
    positions = prepare_positions(spark, args.positions_dir, user_features, entity_lookup, analysis_end_year)
    position_firm_year, position_signal_dates_pa, flagged_positions_pa = build_position_firm_year(positions)
    position_extra, position_signal_dates_ai, flagged_positions_ai = build_position_extra_firm_year(positions, args.manager_seniority_threshold)

    print("[5/9] Preparing postings, existing PA aggregates, and AI aggregates")
    postings = prepare_postings(spark, resolved_postings, entity_lookup)
    posting_firm_year, posting_signal_dates_pa, flagged_postings_pa = build_posting_firm_year(postings)
    posting_extra, posting_signal_dates_ai, flagged_postings_ai = build_posting_extra_firm_year(postings, args.manager_seniority_threshold)

    print("[6/9] Building final firm-year panel")
    position_panel = position_firm_year.withColumnRenamed("parent_rcid", "parent_rcid_pos").withColumnRenamed("firm_name", "firm_name_pos")
    posting_panel = posting_firm_year.withColumnRenamed("parent_rcid", "parent_rcid_post").withColumnRenamed("firm_name", "firm_name_post")
    panel = position_panel.join(posting_panel, on=["firm_key", "year"], how="outer")
    panel = panel.withColumn("parent_rcid", F.coalesce(F.col("parent_rcid_pos"), F.col("parent_rcid_post")))
    panel = panel.withColumn("firm_name", F.coalesce(F.col("firm_name_pos"), F.col("firm_name_post")))
    panel = panel.drop("parent_rcid_pos", "parent_rcid_post", "firm_name_pos", "firm_name_post")

    pos_extra = position_extra.withColumnRenamed("parent_rcid", "parent_rcid_pos_extra").withColumnRenamed("firm_name", "firm_name_pos_extra")
    post_extra = posting_extra.withColumnRenamed("parent_rcid", "parent_rcid_post_extra").withColumnRenamed("firm_name", "firm_name_post_extra")
    panel = panel.join(pos_extra, on=["firm_key", "year"], how="left").join(post_extra, on=["firm_key", "year"], how="left")
    panel = panel.withColumn("parent_rcid", F.coalesce(F.col("parent_rcid"), F.col("parent_rcid_pos_extra"), F.col("parent_rcid_post_extra")))
    panel = panel.withColumn("firm_name", F.coalesce(F.col("firm_name"), F.col("firm_name_pos_extra"), F.col("firm_name_post_extra")))
    panel = panel.drop("parent_rcid_pos_extra", "parent_rcid_post_extra", "firm_name_pos_extra", "firm_name_post_extra")

    # Preserve existing workforce denominator but expose a clearer name.
    panel = panel.withColumn("n_employees", F.coalesce(F.col("n_employees"), F.col("workforce_weighted")))
    panel = panel.withColumn("hr_to_employee_ratio", safe_divide(F.col("n_hr_positions"), F.col("n_employees")))
    panel = panel.withColumn("managers_to_employee_ratio", safe_divide(F.col("n_managers"), F.col("n_employees")))
    panel = attach_parent_static(panel, parent_static)

    print("[7/9] Adding first-event timing")
    signal_dates = outer_join_all([position_signal_dates_pa, posting_signal_dates_pa, position_signal_dates_ai, posting_signal_dates_ai], on=["firm_key"])
    if signal_dates is not None:
        date_columns = [c for c in signal_dates.columns if c != "firm_key" and "_date" in c]
        signal_dates = add_first_event_years(signal_dates, date_columns)
        # Existing people-analytics firm-level first dates.
        signal_dates = signal_dates.withColumn(
            "first_people_analytics_firm_date_any_study",
            min_two_dates("first_people_analytics_position_date_any_study", "first_people_analytics_posting_date_any_study"),
        )
        signal_dates = signal_dates.withColumn(
            "first_people_analytics_firm_date_any_enriched",
            min_two_dates("first_people_analytics_position_date_any_enriched", "first_people_analytics_posting_date_any_enriched"),
        )
        # New AI firm-level first dates.
        signal_dates = signal_dates.withColumn(
            "first_ai_firm_date_any_strict",
            min_two_dates("first_ai_position_date_any_strict", "first_ai_posting_date_any_strict"),
        )
        signal_dates = signal_dates.withColumn(
            "first_ai_firm_date_any_broad",
            min_two_dates("first_ai_position_date_any_broad", "first_ai_posting_date_any_broad"),
        )
        signal_dates = add_first_event_years(
            signal_dates,
            [
                "first_people_analytics_firm_date_any_study",
                "first_people_analytics_firm_date_any_enriched",
                "first_ai_firm_date_any_strict",
                "first_ai_firm_date_any_broad",
            ],
        )
        panel = panel.join(signal_dates, on="firm_key", how="left")

    # Existing PA timing flags retained.
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
        F.when(F.col("first_people_analytics_position_year_any_enriched").isNotNull() & (F.col("year") >= F.col("first_people_analytics_position_year_any_enriched")), F.lit(1)).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "has_people_analytics_posting_any_enriched_by_year",
        F.when(F.col("first_people_analytics_posting_year_any_enriched").isNotNull() & (F.col("year") >= F.col("first_people_analytics_posting_year_any_enriched")), F.lit(1)).otherwise(F.lit(0)),
    )
    panel = panel.withColumn(
        "has_people_analytics_firm_any_enriched_by_year",
        F.when(F.col("first_people_analytics_firm_year_any_enriched").isNotNull() & (F.col("year") >= F.col("first_people_analytics_firm_year_any_enriched")), F.lit(1)).otherwise(F.lit(0)),
    )
    panel = add_adoption_timing(panel, "ai", ["strict", "broad"])

    panel = panel.withColumn("has_position_data", F.when(F.col("workforce_weighted").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    panel = panel.withColumn("has_posting_data", F.when(F.col("posting_count").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    panel = panel.withColumn("parent_rcid_matched", F.when(F.col("parent_rcid").isNotNull(), F.lit(1)).otherwise(F.lit(0)))

    print("[8/9] Writing final firm-year panel and diagnostics")
    write_parquet(panel, out_dir, args.coalesce)
    write_parquet(flagged_positions_ai, os.path.join(diag_dir, "ai_matched_positions"), 20)
    write_parquet(flagged_postings_ai, os.path.join(diag_dir, "ai_matched_postings"), 20)

    written = spark.read.parquet(out_dir)
    (
        written.groupBy("year")
        .agg(
            F.count("*").alias("firm_year_rows"),
            F.countDistinct("firm_key").alias("firms"),
            F.sum(F.coalesce(F.col("ai_positions_any_strict_weighted"), F.lit(0.0))).alias("ai_positions_any_strict_weighted"),
            F.sum(F.coalesce(F.col("ai_postings_any_strict"), F.lit(0.0))).alias("ai_postings_any_strict"),
            F.avg("hr_to_employee_ratio").alias("mean_hr_to_employee_ratio"),
            F.avg("managers_to_employee_ratio").alias("mean_managers_to_employee_ratio"),
        )
        .orderBy("year")
        .coalesce(1)
        .write.mode("overwrite").option("header", True).csv(os.path.join(diag_dir, "yearly_ai_hr_manager_summary_csv"))
    )

    overlap = written.select(
        "firm_key",
        "first_people_analytics_firm_year_any_enriched",
        "first_ai_firm_year_any_strict",
        "first_ai_firm_year_any_broad",
    ).dropDuplicates(["firm_key"])
    (
        overlap.agg(
            F.count("*").alias("firms"),
            F.sum(F.when(F.col("first_people_analytics_firm_year_any_enriched").isNotNull(), 1).otherwise(0)).alias("pa_adopters"),
            F.sum(F.when(F.col("first_ai_firm_year_any_strict").isNotNull(), 1).otherwise(0)).alias("strict_ai_adopters"),
            F.sum(F.when(F.col("first_people_analytics_firm_year_any_enriched").isNotNull() & F.col("first_ai_firm_year_any_strict").isNotNull(), 1).otherwise(0)).alias("pa_and_strict_ai_adopters"),
        )
        .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(diag_dir, "pa_ai_adoption_overlap_csv"))
    )

    print("[9/9] Quick checks")
    print(f"[INFO] Firm-year rows: {written.count():,}")
    print(f"[INFO] Firms: {written.select('firm_key').distinct().count():,}")
    print(f"[INFO] Strict AI adopting firms: {written.where(F.col('first_ai_firm_year_any_strict').isNotNull()).select('firm_key').distinct().count():,}")
    print(f"[INFO] Diagnostics: {diag_dir}")
    spark.stop()


if __name__ == "__main__":
    main()
