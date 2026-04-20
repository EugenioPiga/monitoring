#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study people analytics / monitoring variables in final firm-year panel.")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--year-col", default="year")
    p.add_argument("--master", default=None)
    p.add_argument("--shuffle-partitions", type=int, default=None)
    p.add_argument("--corr-sample-frac", type=float, default=0.01)
    p.add_argument("--hist-bin-width", type=float, default=0.02)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_spark(app_name: str, master: str | None = None, shuffle_partitions: int | None = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    if shuffle_partitions is not None:
        spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
    spark.sparkContext.setLogLevel("WARN")
    return spark


def save_plot(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def collect_rows(df) -> List[dict]:
    return [r.asDict() for r in df.collect()]


def stats_for_var(df, v: str) -> dict:
    row = (
        df.agg(
            F.count(F.lit(1)).alias("n_rows"),
            F.count(F.when(F.col(v).isNotNull(), 1)).alias("n_nonmissing"),
            F.sum(F.when(F.col(v).isNull(), 1).otherwise(0)).alias("n_missing"),
            F.sum(F.when(F.col(v) == 0, 1).otherwise(0)).alias("n_zero"),
            F.sum(F.when(F.col(v) > 0, 1).otherwise(0)).alias("n_positive"),
            F.mean(v).alias("mean"),
            F.stddev(v).alias("sd"),
            F.min(v).alias("min"),
            F.expr(f"percentile_approx({v}, 0.01)").alias("p01"),
            F.expr(f"percentile_approx({v}, 0.05)").alias("p05"),
            F.expr(f"percentile_approx({v}, 0.25)").alias("p25"),
            F.expr(f"percentile_approx({v}, 0.50)").alias("p50"),
            F.expr(f"percentile_approx({v}, 0.75)").alias("p75"),
            F.expr(f"percentile_approx({v}, 0.95)").alias("p95"),
            F.expr(f"percentile_approx({v}, 0.99)").alias("p99"),
            F.max(v).alias("max"),
        )
        .collect()[0]
        .asDict()
    )
    n_rows = row["n_rows"] or 0
    row["variable"] = v
    row["missing_share"] = (row["n_missing"] / n_rows) if n_rows else None
    row["zero_share"] = (row["n_zero"] / n_rows) if n_rows else None
    row["positive_share"] = (row["n_positive"] / n_rows) if n_rows else None
    return row


def positive_stats_for_var(df, v: str) -> dict:
    pos = df.where(F.col(v) > 0)
    row = (
        pos.agg(
            F.count(F.lit(1)).alias("n_positive_rows"),
            F.mean(v).alias("mean_positive"),
            F.stddev(v).alias("sd_positive"),
            F.min(v).alias("min_positive"),
            F.expr(f"percentile_approx({v}, 0.25)").alias("p25_positive"),
            F.expr(f"percentile_approx({v}, 0.50)").alias("p50_positive"),
            F.expr(f"percentile_approx({v}, 0.75)").alias("p75_positive"),
            F.expr(f"percentile_approx({v}, 0.90)").alias("p90_positive"),
            F.expr(f"percentile_approx({v}, 0.95)").alias("p95_positive"),
            F.expr(f"percentile_approx({v}, 0.99)").alias("p99_positive"),
            F.max(v).alias("max_positive"),
        )
        .collect()[0]
        .asDict()
    )
    row["variable"] = v
    return row


def yearly_summary(df, v: str, year_col: str) -> pd.DataFrame:
    out = (
        df.groupBy(year_col)
          .agg(
              F.count(F.when(F.col(v).isNotNull(), 1)).alias("n_nonmissing"),
              F.mean(v).alias("mean"),
              F.avg(F.when(F.col(v) > 0, 1).otherwise(0)).alias("positive_share"),
              F.expr(f"percentile_approx({v}, 0.50)").alias("p50"),
              F.expr(f"percentile_approx({v}, 0.90)").alias("p90"),
          )
          .orderBy(year_col)
          .toPandas()
    )
    out["variable"] = v
    return out


def yearly_summary_conditional(df, v: str, year_col: str, cond_col: str) -> pd.DataFrame:
    if cond_col not in df.columns:
        return pd.DataFrame()
    out = (
        df.where(F.col(cond_col) == 1)
          .groupBy(year_col)
          .agg(
              F.count(F.when(F.col(v).isNotNull(), 1)).alias("n_nonmissing"),
              F.mean(v).alias("mean"),
              F.avg(F.when(F.col(v) > 0, 1).otherwise(0)).alias("positive_share"),
              F.expr(f"percentile_approx({v}, 0.50)").alias("p50"),
          )
          .orderBy(year_col)
          .toPandas()
    )
    out["variable"] = v
    out["condition"] = cond_col
    return out


def histogram_bins(df, v: str, bin_width: float) -> pd.DataFrame:
    max_bin = int(round(1.0 / bin_width))
    out = (
        df.where(F.col(v).isNotNull())
          .withColumn(
              "bin_left",
              F.when(F.col(v) >= 1, F.lit(1.0 - bin_width))
               .otherwise(F.floor(F.col(v) / F.lit(bin_width)) * F.lit(bin_width))
          )
          .groupBy("bin_left")
          .count()
          .orderBy("bin_left")
          .toPandas()
    )
    if not out.empty:
        out["bin_right"] = out["bin_left"] + bin_width
        out["variable"] = v
    return out


def plot_yearly_lines(df_year: pd.DataFrame, vars_to_plot: List[str], value_col: str, title: str, path: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for v in vars_to_plot:
        tmp = df_year[df_year["variable"] == v].sort_values("year")
        if tmp.empty:
            continue
        ax.plot(tmp["year"], tmp[value_col], marker="o", linewidth=1.8, label=v)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    save_plot(fig, path)


def plot_histograms(hist_map: Dict[str, pd.DataFrame], title: str, path: str) -> None:
    n = len(hist_map)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (v, h) in zip(axes, hist_map.items()):
        ax.bar(h["bin_left"], h["count"], width=(h["bin_right"] - h["bin_left"]).iloc[0], align="edge")
        ax.set_title(v)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

    for ax in axes[len(hist_map):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    save_plot(fig, path)


def plot_first_adoption_counts(df_adopt: pd.DataFrame, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in [col for col in df_adopt.columns if col != "year"]:
        ax.plot(df_adopt["year"], df_adopt[c], marker="o", linewidth=1.8, label=c)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    save_plot(fig, path)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    spark = build_spark(
        app_name="study_pa_variables",
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
    )

    position_vars = [
        "people_analytics_positions_title_study_share",
        "people_analytics_positions_description_study_share",
        "people_analytics_positions_any_study_share",
        "people_analytics_positions_title_enriched_share",
        "people_analytics_positions_description_enriched_share",
        "people_analytics_positions_any_enriched_share",
    ]

    posting_vars = [
        "people_analytics_postings_title_study_share",
        "people_analytics_postings_description_study_share",
        "people_analytics_postings_any_study_share",
        "people_analytics_postings_title_enriched_share",
        "people_analytics_postings_description_enriched_share",
        "people_analytics_postings_any_enriched_share",
    ]

    monitoring_vars = [
        "workers_with_employee_feedback_tool_skill_share",
        "workers_with_hr_technology_skill_share",
        "workers_with_hr_skill_share",
        "hr_people_role_share",
        "data_analytics_role_share",
    ]

    adoption_vars = [
        "is_first_people_analytics_firm_year_any_enriched",
        "has_people_analytics_firm_any_enriched_by_year",
        "is_first_people_analytics_position_year_any_enriched",
        "has_people_analytics_position_any_enriched_by_year",
        "is_first_people_analytics_posting_year_any_enriched",
        "has_people_analytics_posting_any_enriched_by_year",
    ]

    support_vars = [
        "has_position_data",
        "has_posting_data",
        "workforce_weighted",
        "avg_salary",
        "hire_rate",
        "exit_rate",
    ]

    requested = position_vars + posting_vars + monitoring_vars + adoption_vars + support_vars + [args.year_col]
    df0 = spark.read.parquet(args.dataset_path)
    keep = [c for c in requested if c in df0.columns]
    df = df0.select(*keep)

    manifest = {
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "n_columns_loaded": len(keep),
        "variables_present": keep,
        "position_vars_present": [v for v in position_vars if v in keep],
        "posting_vars_present": [v for v in posting_vars if v in keep],
        "monitoring_vars_present": [v for v in monitoring_vars if v in keep],
        "adoption_vars_present": [v for v in adoption_vars if v in keep],
    }
    write_json(manifest, os.path.join(args.output_dir, "00_manifest.json"))

    study_vars = [v for v in (position_vars + posting_vars + monitoring_vars + adoption_vars) if v in keep]

    # -----------------------------
    # 1. Overall distribution tables
    # -----------------------------
    overall_rows = []
    positive_rows = []
    for v in study_vars:
        overall_rows.append(stats_for_var(df, v))
        positive_rows.append(positive_stats_for_var(df, v))

    overall_df = pd.DataFrame(overall_rows).sort_values("variable")
    overall_df.to_csv(os.path.join(args.output_dir, "01_overall_summary.csv"), index=False)

    positive_df = pd.DataFrame(positive_rows).sort_values("variable")
    positive_df.to_csv(os.path.join(args.output_dir, "02_positive_only_summary.csv"), index=False)

    # -----------------------------------
    # 2. Yearly evolution and yearly stats
    # -----------------------------------
    yearly_frames = []
    yearly_cond_frames = []

    for v in study_vars:
        yearly_frames.append(yearly_summary(df, v, args.year_col))

        if v in posting_vars and "has_posting_data" in df.columns:
            yc = yearly_summary_conditional(df, v, args.year_col, "has_posting_data")
            if not yc.empty:
                yearly_cond_frames.append(yc)

        if v in position_vars and "has_position_data" in df.columns:
            yc = yearly_summary_conditional(df, v, args.year_col, "has_position_data")
            if not yc.empty:
                yearly_cond_frames.append(yc)

    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(os.path.join(args.output_dir, "03_yearly_summary.csv"), index=False)

    yearly_cond_df = pd.concat(yearly_cond_frames, ignore_index=True) if yearly_cond_frames else pd.DataFrame()
    yearly_cond_df.to_csv(os.path.join(args.output_dir, "04_yearly_summary_conditional_on_data.csv"), index=False)

    # -----------------------------------
    # 3. Adoption timing counts by year
    # -----------------------------------
    first_adopt_cols = [c for c in adoption_vars if c.startswith("is_first_") and c in keep]
    if first_adopt_cols:
        aggs = [F.sum(F.col(c)).alias(c) for c in first_adopt_cols]
        by_year_adopt = df.groupBy(args.year_col).agg(*aggs).orderBy(args.year_col).toPandas()
        by_year_adopt.to_csv(os.path.join(args.output_dir, "05_first_adoption_counts_by_year.csv"), index=False)
        plot_first_adoption_counts(
            by_year_adopt.rename(columns={args.year_col: "year"}),
            "First-adoption counts by year",
            os.path.join(args.output_dir, "05_first_adoption_counts_by_year.png"),
        )

    # -----------------------------------
    # 4. Correlation table on a small sample
    # -----------------------------------
    corr_vars = [v for v in [
        "people_analytics_positions_any_enriched_share",
        "people_analytics_postings_any_enriched_share",
        "workers_with_employee_feedback_tool_skill_share",
        "workers_with_hr_technology_skill_share",
        "workers_with_hr_skill_share",
        "hr_people_role_share",
        "data_analytics_role_share",
    ] if v in keep]

    corr_rows = []
    if len(corr_vars) >= 2:
        sdf = df.select(*corr_vars).sample(False, args.corr_sample_frac, seed=0)
        for i, a in enumerate(corr_vars):
            for b in corr_vars[i:]:
                try:
                    corr = sdf.stat.corr(a, b)
                except Exception:
                    corr = None
                corr_rows.append({"var1": a, "var2": b, "corr_sample": corr, "sample_frac": args.corr_sample_frac})
                if a != b:
                    corr_rows.append({"var1": b, "var2": a, "corr_sample": corr, "sample_frac": args.corr_sample_frac})

    pd.DataFrame(corr_rows).to_csv(os.path.join(args.output_dir, "06_correlation_table_sample.csv"), index=False)

    # -----------------------------------
    # 5. Histograms / binned distributions
    # -----------------------------------
    hist_vars = [v for v in [
        "people_analytics_positions_any_study_share",
        "people_analytics_positions_any_enriched_share",
        "people_analytics_postings_any_study_share",
        "people_analytics_postings_any_enriched_share",
        "workers_with_employee_feedback_tool_skill_share",
        "workers_with_hr_technology_skill_share",
        "workers_with_hr_skill_share",
        "hr_people_role_share",
    ] if v in keep]

    hist_map = {}
    hist_rows = []
    for v in hist_vars:
        h = histogram_bins(df, v, args.hist_bin_width)
        if not h.empty:
            hist_map[v] = h
            hist_rows.append(h)

    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "07_histogram_bins.csv"),
            index=False
        )
        plot_histograms(
            hist_map,
            "Binned distributions of key people-analytics / monitoring variables",
            os.path.join(args.output_dir, "07_histogram_bins.png"),
        )

    # -----------------------------------
    # 6. Graphs: yearly means and positivity
    # -----------------------------------
    if not yearly_df.empty:
        ydf = yearly_df.rename(columns={args.year_col: "year"})

        pos_present = [v for v in position_vars if v in keep]
        post_present = [v for v in posting_vars if v in keep]
        mon_present = [v for v in monitoring_vars if v in keep]
        adopt_present = [v for v in adoption_vars if v in keep]

        if pos_present:
            plot_yearly_lines(
                ydf, pos_present, "mean",
                "People-analytics position-share variables: yearly means",
                os.path.join(args.output_dir, "08_positions_yearly_means.png"),
                "Mean"
            )
            plot_yearly_lines(
                ydf, pos_present, "positive_share",
                "People-analytics position-share variables: yearly positive share",
                os.path.join(args.output_dir, "09_positions_yearly_positive_share.png"),
                "Positive share"
            )

        if post_present:
            plot_yearly_lines(
                ydf, post_present, "mean",
                "People-analytics posting-share variables: yearly means",
                os.path.join(args.output_dir, "10_postings_yearly_means.png"),
                "Mean"
            )
            plot_yearly_lines(
                ydf, post_present, "positive_share",
                "People-analytics posting-share variables: yearly positive share",
                os.path.join(args.output_dir, "11_postings_yearly_positive_share.png"),
                "Positive share"
            )

        if mon_present:
            plot_yearly_lines(
                ydf, mon_present, "mean",
                "Monitoring / related proxy variables: yearly means",
                os.path.join(args.output_dir, "12_monitoring_yearly_means.png"),
                "Mean"
            )
            plot_yearly_lines(
                ydf, mon_present, "positive_share",
                "Monitoring / related proxy variables: yearly positive share",
                os.path.join(args.output_dir, "13_monitoring_yearly_positive_share.png"),
                "Positive share"
            )

        if adopt_present:
            plot_yearly_lines(
                ydf, adopt_present, "mean",
                "Adoption indicator variables: yearly means",
                os.path.join(args.output_dir, "14_adoption_indicators_yearly_means.png"),
                "Mean"
            )

    # -----------------------------------
    # 7. Posting-data conditional comparisons
    # -----------------------------------
    compare_rows = []
    for v in [vv for vv in posting_vars if vv in keep]:
        base = stats_for_var(df, v)
        if "has_posting_data" in df.columns:
            cond = stats_for_var(df.where(F.col("has_posting_data") == 1), v)
            cond = {f"posting_data_{k}": val for k, val in cond.items() if k != "variable"}
        else:
            cond = {}
        row = {"variable": v}
        row.update({f"all_rows_{k}": val for k, val in base.items() if k != "variable"})
        row.update(cond)
        compare_rows.append(row)

    for v in [vv for vv in position_vars if vv in keep]:
        base = stats_for_var(df, v)
        if "has_position_data" in df.columns:
            cond = stats_for_var(df.where(F.col("has_position_data") == 1), v)
            cond = {f"position_data_{k}": val for k, val in cond.items() if k != "variable"}
        else:
            cond = {}
        row = {"variable": v}
        row.update({f"all_rows_{k}": val for k, val in base.items() if k != "variable"})
        row.update(cond)
        compare_rows.append(row)

    pd.DataFrame(compare_rows).to_csv(
        os.path.join(args.output_dir, "15_distribution_conditional_on_position_or_posting_data.csv"),
        index=False
    )

    # -----------------------------------
    # 8. Metadata
    # -----------------------------------
    meta = {
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "year_col": args.year_col,
        "n_rows_panel": df.count(),
        "n_variables_studied": len(study_vars),
        "variables_studied": study_vars,
        "corr_sample_frac": args.corr_sample_frac,
        "hist_bin_width": args.hist_bin_width,
    }
    write_json(meta, os.path.join(args.output_dir, "99_metadata.json"))

    print("[INFO] Done. Outputs written to:", args.output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
