#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F


OUTCOMES = [
    "d5_log_workers",
    "d5_exit_rate",
    "d5_hire_rate",
    "d5_skill_count_sd",
    "d5_skill_bundle_dispersion",
    "d5_skill_hhi_mean",
    "d5_specialist_share",
]

VISIBILITY_CANDIDATES = [
    "mean_user_has_data_skill",
    "mean_user_has_software_skill",
    "mean_user_has_hr_skill",
    "mean_user_has_hr_technology_skill",
    "mean_user_has_employee_feedback_tool_skill",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pre-start-year", type=int, default=2016)
    p.add_argument("--pre-end-year", type=int, default=2018)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pvalue_from_t(t: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(t)))


def residualize_twfe(df: pd.DataFrame, cols, fe1, fe2, max_iter=30, tol=1e-8):
    """
    Alternating demeaning for two high-dimensional fixed effects.
    Residualizes all columns in `cols` on fe1 and fe2.
    """
    out = df[cols].astype(float).copy()
    last_norm = None

    for _ in range(max_iter):
        for c in cols:
            out[c] = out[c] - out.groupby(df[fe1])[c].transform("mean")
            out[c] = out[c] - out.groupby(df[fe2])[c].transform("mean")

        norm = float(np.sqrt(np.nanmean(out[cols].to_numpy() ** 2)))
        if last_norm is not None and abs(last_norm - norm) < tol:
            break
        last_norm = norm

    return out


def fit_cluster_ols(y, X, clusters):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    clusters = np.asarray(clusters)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[ok]
    X = X[ok, :]
    clusters = clusters[ok]

    n, k = X.shape
    if n <= k:
        return None

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    unique_clusters = pd.unique(clusters)
    meat = np.zeros((k, k))

    for g in unique_clusters:
        idx = np.where(clusters == g)[0]
        Xg = X[idx, :]
        ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)

    G = len(unique_clusters)
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t = beta / se
    p = np.array([pvalue_from_t(x) if np.isfinite(x) else np.nan for x in t])

    ybar = np.mean(y)
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((y - ybar) ** 2)

    return beta, se, t, p, n, G, r2


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    spark = SparkSession.builder.appName("parent_occ_visibility_analysis").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df0 = spark.read.parquet(args.panel_dir)
    cols_present = df0.columns

    visibility_cols = [c for c in VISIBILITY_CANDIDATES if c in cols_present]
    if not visibility_cols:
        raise ValueError(
            "No visibility proxy columns found. Expected columns like mean_user_has_data_skill or mean_user_has_hr_technology_skill."
        )

    keep = [
        "parent_rcid",
        "occupation",
        "year",
        "occupation_analysis_sample",
        "pa_posting_log1p",
        "n_workers",
        "naics3",
    ] + visibility_cols + [y for y in OUTCOMES if y in cols_present]

    sdf = (
        df0
        .where(F.col("occupation_analysis_sample") == 1)
        .select(*keep)
    )

    pdf = sdf.toPandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan)

    # Build predetermined occupation visibility index from pre-period occupation means.
    pre = pdf[
        (pdf["year"] >= args.pre_start_year)
        & (pdf["year"] <= args.pre_end_year)
    ].copy()

    occ_pre = pre.groupby("occupation", as_index=False)[visibility_cols].mean()

    for c in visibility_cols:
        sd = occ_pre[c].std()
        if pd.notna(sd) and sd > 0:
            occ_pre[f"z_{c}"] = (occ_pre[c] - occ_pre[c].mean()) / sd
        else:
            occ_pre[f"z_{c}"] = 0.0

    z_cols = [f"z_{c}" for c in visibility_cols]
    occ_pre["visibility_index"] = occ_pre[z_cols].mean(axis=1)

    pdf = pdf.merge(
        occ_pre[["occupation", "visibility_index"]],
        on="occupation",
        how="left",
    )

    pdf["parent_year_fe"] = pdf["parent_rcid"].astype(str) + "_y" + pdf["year"].astype(int).astype(str)
    pdf["occupation_year_fe"] = pdf["occupation"].astype(str) + "_y" + pdf["year"].astype(int).astype(str)

    pdf["log_n_workers"] = np.where(pdf["n_workers"] > 0, np.log(pdf["n_workers"]), np.nan)
    pdf["treat_visibility"] = pdf["pa_posting_log1p"] * pdf["visibility_index"]

    meta = {
        "panel_dir": args.panel_dir,
        "out_dir": args.out_dir,
        "rows_loaded": int(len(pdf)),
        "visibility_cols": visibility_cols,
        "pre_start_year": args.pre_start_year,
        "pre_end_year": args.pre_end_year,
        "n_parents": int(pdf["parent_rcid"].nunique()),
        "n_occupations": int(pdf["occupation"].nunique()),
    }
    save_json(meta, os.path.join(args.out_dir, "00_metadata.json"))

    occ_pre.to_csv(os.path.join(args.out_dir, "01_occupation_visibility_index.csv"), index=False)

    results = []
    diagnostics = []

    for outcome in OUTCOMES:
        if outcome not in pdf.columns:
            continue

        use = pdf.dropna(
            subset=[
                outcome,
                "treat_visibility",
                "log_n_workers",
                "parent_year_fe",
                "occupation_year_fe",
            ]
        ).copy()

        if use.empty:
            continue

        # Residualize y, treatment, and level control on parent-year and occupation-year FE.
        resid = residualize_twfe(
            use,
            cols=[outcome, "treat_visibility", "log_n_workers"],
            fe1="parent_year_fe",
            fe2="occupation_year_fe",
        )

        y = resid[outcome].to_numpy()
        X = resid[["treat_visibility", "log_n_workers"]].to_numpy()

        fit = fit_cluster_ols(y, X, use["parent_rcid"].to_numpy())
        if fit is None:
            continue

        beta, se, t, p, n, G, r2 = fit
        terms = ["pa_posting_log1p_x_visibility_index", "log_n_workers"]

        for i, term in enumerate(terms):
            results.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": float(beta[i]),
                    "std_err": float(se[i]),
                    "t_stat": float(t[i]),
                    "p_value": float(p[i]),
                    "nobs": int(n),
                    "n_clusters": int(G),
                    "r2_resid": float(r2),
                    "fixed_effects": "parent_year + occupation_year",
                }
            )

        diagnostics.append(
            {
                "outcome": outcome,
                "nobs": int(n),
                "n_clusters": int(G),
                "mean_y": float(use[outcome].mean()),
                "sd_y": float(use[outcome].std()),
                "mean_treat_visibility": float(use["treat_visibility"].mean()),
                "sd_treat_visibility": float(use["treat_visibility"].std()),
            }
        )

    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "10_parent_occ_twfe_results.csv"), index=False)
    pd.DataFrame(diagnostics).to_csv(os.path.join(args.out_dir, "11_parent_occ_diagnostics.csv"), index=False)

    spark.stop()


if __name__ == "__main__":
    main()
