#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F


OUTCOMES = [
    "d5_log_workforce",
    "d5_log_avg_salary",
    "d5_hire_rate",
    "d5_exit_rate",
    "d5_skill_count_sd",
    "d5_skill_bundle_dispersion",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run lightweight regressions on collapsed parent-year panel.")
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-years", type=int, default=0)
    p.add_argument("--min-avg-workforce", type=float, default=0.0)
    p.add_argument("--with-fe", action="store_true")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def load_filtered_base(spark: SparkSession, panel_dir: str, min_years: int, min_avg_workforce: float):
    df = spark.read.parquet(panel_dir)

    # first restrict to intended analysis sample
    df = df.where(F.col("analysis_sample") == 1)

    # optional parent-level restrictions
    parent_stats = (
        df.groupBy("parent_rcid")
        .agg(
            F.countDistinct("year").alias("n_years"),
            F.avg("workforce_weighted").alias("avg_workforce"),
        )
    )

    if min_years > 0:
        parent_stats = parent_stats.where(F.col("n_years") >= min_years)
    if min_avg_workforce > 0:
        parent_stats = parent_stats.where(F.col("avg_workforce") >= F.lit(float(min_avg_workforce)))

    keep_parents = parent_stats.select("parent_rcid")
    df = df.join(keep_parents, on="parent_rcid", how="inner")

    return df


def prepare_outcome_pdf(df_spark, outcome: str) -> pd.DataFrame:
    cols = [
        "parent_rcid",
        "year",
        "naics3",
        "pa_posting_log1p",
        "log_workforce",
        "avg_salary",
        "has_position_data",
        "has_posting_data",
        "L1_d_log_workforce",
        "L1_d_log_avg_salary",
        outcome,
    ]
    cols = [c for c in cols if c in df_spark.columns]

    sdf = (
        df_spark
        .where(F.col(outcome).isNotNull())
        .where(F.col("pa_posting_log1p").isNotNull())
        .where(F.col("naics3").isNotNull())
        .select(*cols)
    )

    pdf = sdf.toPandas()
    if pdf.empty:
        return pdf

    pdf = pdf.replace([np.inf, -np.inf], np.nan)

    if outcome == "d5_log_workforce":
        pdf["lag_growth"] = pdf["L1_d_log_workforce"] if "L1_d_log_workforce" in pdf.columns else np.nan
        pdf["level_control"] = pdf["log_workforce"] if "log_workforce" in pdf.columns else np.nan

    elif outcome == "d5_log_avg_salary":
        pdf["lag_growth"] = pdf["L1_d_log_avg_salary"] if "L1_d_log_avg_salary" in pdf.columns else np.nan
        if "avg_salary" in pdf.columns:
            pdf["level_control"] = np.where(pdf["avg_salary"] > 0, np.log(pdf["avg_salary"]), np.nan)
        else:
            pdf["level_control"] = np.nan

    else:
        pdf["lag_growth"] = np.nan
        pdf["level_control"] = pdf["log_workforce"] if "log_workforce" in pdf.columns else np.nan

    pdf["fe_ind_year"] = pdf["naics3"].astype(str) + "_y" + pdf["year"].astype(int).astype(str)
    return pdf

def fit_ols_clustered(pdf: pd.DataFrame, outcome: str, with_fe: bool) -> pd.DataFrame:
    rhs = ["pa_posting_log1p", "level_control", "has_position_data", "has_posting_data"]
    if pdf["lag_growth"].notna().sum() > 0:
        rhs.append("lag_growth")

    use = pdf.dropna(subset=[outcome, "pa_posting_log1p", "level_control"]).copy()

    X = use[rhs].copy()

    if with_fe:
        fe = pd.get_dummies(use["fe_ind_year"], prefix="fe", drop_first=True, dtype=float)
        X = pd.concat([X, fe], axis=1)

    X.insert(0, "const", 1.0)

    y = use[outcome].astype(float).to_numpy()
    Xmat = X.to_numpy(dtype=float)
    groups = use["parent_rcid"].to_numpy()

    n, k = Xmat.shape
    if n <= k:
        return pd.DataFrame()

    XtX = Xmat.T @ Xmat
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xmat.T @ y)
    resid = y - Xmat @ beta

    # cluster-robust sandwich
    unique_groups = pd.unique(groups)
    meat = np.zeros((k, k))
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        Xg = Xmat[idx, :]
        ug = resid[idx]
        Xgu = Xg.T @ ug
        meat += np.outer(Xgu, Xgu)

    G = len(unique_groups)
    if G > 1 and n > k:
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
    else:
        correction = 1.0

    V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    tvals = beta / se
    pvals = np.array([two_sided_p_from_z(t) if np.isfinite(t) else np.nan for t in tvals])

    # simple R2
    ybar = np.mean(y)
    tss = np.sum((y - ybar) ** 2)
    rss = np.sum(resid ** 2)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    out_rows = []
    for i, name in enumerate(X.columns):
        if name in ["const"] or name.startswith("fe_"):
            continue
        out_rows.append(
            {
                "model": "OLS_FE" if with_fe else "OLS",
                "outcome": outcome,
                "term": name,
                "coef": float(beta[i]),
                "std_err": float(se[i]),
                "t_stat": float(tvals[i]),
                "p_value": float(pvals[i]),
                "nobs": int(n),
                "n_clusters": int(G),
                "r2": float(r2),
            }
        )

    return pd.DataFrame(out_rows)


def make_summary_table(df_spark) -> pd.DataFrame:
    rows = []

    for outcome in OUTCOMES + ["pa_posting_log1p"]:
        if outcome not in df_spark.columns:
            continue

        row = (
            df_spark
            .agg(
                F.count(F.col(outcome)).alias("n_nonmissing"),
                F.mean(F.col(outcome)).alias("mean"),
                F.stddev(F.col(outcome)).alias("sd"),
                F.expr(f"percentile_approx({outcome}, 0.5)").alias("p50"),
            )
            .collect()[0]
        )

        rows.append(
            {
                "outcome": outcome,
                "n_nonmissing": int(row["n_nonmissing"]) if row["n_nonmissing"] is not None else 0,
                "mean": float(row["mean"]) if row["mean"] is not None else np.nan,
                "sd": float(row["sd"]) if row["sd"] is not None else np.nan,
                "p50": float(row["p50"]) if row["p50"] is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    spark = SparkSession.builder.appName("parent_first_pass_regressions").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = load_filtered_base(
        spark,
        args.panel_dir,
        min_years=args.min_years,
        min_avg_workforce=args.min_avg_workforce,
    )

    meta = {
        "panel_dir": args.panel_dir,
        "filtered_rows": df.count(),
        "n_parents": df.select("parent_rcid").distinct().count(),
        "years": [r[0] for r in df.select("year").distinct().orderBy("year").collect()],
        "min_years": args.min_years,
        "min_avg_workforce": args.min_avg_workforce,
        "with_fe": args.with_fe,
    }
    save_json(meta, os.path.join(args.out_dir, "00_regression_metadata.json"))

    summary = make_summary_table(df)
    summary.to_csv(os.path.join(args.out_dir, "01_outcome_summary.csv"), index=False)

    all_results = []
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        pdf = prepare_outcome_pdf(df, outcome)
        if pdf.empty:
            continue
        res = fit_ols_clustered(pdf, outcome, with_fe=args.with_fe)
        if not res.empty:
            all_results.append(res)

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "10_ols_results.csv"),
            index=False,
        )

    spark.stop()


if __name__ == "__main__":
    main()
