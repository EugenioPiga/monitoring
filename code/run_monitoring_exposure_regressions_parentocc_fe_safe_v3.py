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
    "d5_log_workers", "d5_exit_rate", "d5_hire_rate", "d5_skill_count_sd", "d5_skill_bundle_dispersion",
    "d5_skill_hhi_mean", "d5_specialist_share",
    "exit_rate", "hire_rate", "skill_count_sd", "skill_bundle_dispersion", "skill_hhi_mean", "specialist_share",
    "hr_to_employee_ratio", "managers_to_employee_ratio",
    "n_promotions", "promotion_rate", "promotion_rate_continuers",
]

EXPOSURES = [
    "monitoring_exposure_average", "monitoring_exposure_concentration", "monitoring_similarity_average",
    "pa_visibility_internal_oldformula", "pa_visibility_external_oldformula",
    "pa_visibility_internal_loginside", "pa_visibility_external_loginside",
    "ai_visibility_internal_oldformula", "ai_visibility_external_oldformula",
    "ai_visibility_internal_loginside", "ai_visibility_external_loginside",
]


def parse_args():
    p = argparse.ArgumentParser(description="Table-5-style regressions with parent x occupation FE and no log cell-size RHS control.")
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument(
        "--fe-spec",
        choices=["parent_occ_year", "parent_occ_parent_year_occ_year"],
        default="parent_occ_parent_year_occ_year",
        help="Default keeps the previous parent-year + occupation-year structure and adds parent x occupation FE.",
    )
    p.add_argument("--min-nobs", type=int, default=500)
    return p.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(o, p):
    ensure_dir(str(Path(p).parent))
    with open(p, "w") as f:
        json.dump(o, f, indent=2, sort_keys=True, default=str)


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pval(t):
    return 2 * (1 - ncdf(abs(t)))


def resid_multi_fe(df: pd.DataFrame, cols: list[str], fe_cols: list[str], max_iter: int, tol: float) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    last = None
    for _ in range(max_iter):
        for fe in fe_cols:
            keys = df[fe]
            for c in cols:
                out[c] = out[c] - out.groupby(keys, sort=False)[c].transform("mean")
        norm = float(np.sqrt(np.nanmean(out[cols].to_numpy() ** 2)))
        if last is not None and abs(last - norm) < tol:
            break
        last = norm
    return out


def fit_cluster(y, X, cl):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    cl = np.asarray(cl)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X, cl = y[ok], X[ok], cl[ok]
    n, k = X.shape
    if n <= k:
        return None
    inv = np.linalg.pinv(X.T @ X)
    b = inv @ (X.T @ y)
    u = y - X @ b
    meat = np.zeros((k, k))
    groups = pd.unique(cl)
    for g in groups:
        idx = np.where(cl == g)[0]
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    G = len(groups)
    corr = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = corr * inv @ meat @ inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
    p = np.array([pval(x) if np.isfinite(x) else np.nan for x in t])
    r2 = 1 - np.sum(u ** 2) / np.sum((y - y.mean()) ** 2) if np.sum((y - y.mean()) ** 2) > 0 else np.nan
    return b, se, t, p, n, G, r2


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    spark = SparkSession.builder.appName("monitoring_exposure_table5_parentocc_fe").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    df0 = spark.read.parquet(args.panel_dir)

    available_outcomes = [c for c in OUTCOMES if c in df0.columns]
    available_exposures = [c for c in EXPOSURES if c in df0.columns]
    if not available_outcomes:
        raise RuntimeError("None of the requested outcomes are in the panel.")
    if not available_exposures:
        raise RuntimeError("None of the requested exposure variables are in the panel.")

    base_cols = ["parent_rcid", "occupation", "year"]
    select_cols = sorted(set(base_cols + available_outcomes + available_exposures + ["occupation_onet_similarity", "n_workers"]))
    pdf = df0.select(*[c for c in select_cols if c in df0.columns]).toPandas().replace([np.inf, -np.inf], np.nan)
    pdf = pdf.dropna(subset=["parent_rcid", "occupation", "year"]).copy()
    pdf["parent_rcid"] = pdf["parent_rcid"].astype(str)
    pdf["occupation"] = pdf["occupation"].astype(str)
    pdf["year"] = pdf["year"].astype(int)
    pdf["parent_year_fe"] = pdf["parent_rcid"] + "_y" + pdf["year"].astype(str)
    pdf["occupation_year_fe"] = pdf["occupation"] + "_y" + pdf["year"].astype(str)
    pdf["parent_occupation_fe"] = pdf["parent_rcid"] + "_occ_" + pdf["occupation"]
    pdf["year_fe"] = pdf["year"].astype(str)

    if args.fe_spec == "parent_occ_year":
        fe_cols = ["parent_occupation_fe", "year_fe"]
    else:
        fe_cols = ["parent_occupation_fe", "parent_year_fe", "occupation_year_fe"]

    rows = []
    diags = []
    for y in available_outcomes:
        for x in available_exposures:
            use = pdf.dropna(subset=[y, x] + fe_cols + ["parent_rcid"]).copy()
            if len(use) < args.min_nobs:
                continue
            sd = use[x].std()
            if not (pd.notna(sd) and sd > 0):
                continue
            sx = f"std_{x}"
            use[sx] = (use[x] - use[x].mean()) / sd
            r = resid_multi_fe(use, [y, sx], fe_cols, args.max_iter, args.tol)
            out = fit_cluster(r[y], r[[sx]], use["parent_rcid"])
            if out is None:
                continue
            b, se, t, p, n, G, r2 = out
            rows.append({
                "outcome": y,
                "exposure": x,
                "term": sx,
                "coef": float(b[0]),
                "std_err": float(se[0]),
                "t_stat": float(t[0]),
                "p_value": float(p[0]),
                "nobs": int(n),
                "n_clusters": int(G),
                "r2_resid": float(r2),
                "fixed_effects": " + ".join(fe_cols),
                "rhs_controls": "none; log_n_workers deliberately excluded",
            })
        y_use = pdf.dropna(subset=[y]).copy()
        if not y_use.empty:
            diags.append({
                "outcome": y,
                "nobs_nonmissing": int(len(y_use)),
                "n_clusters": int(y_use["parent_rcid"].nunique()),
                "y_mean": float(y_use[y].mean()),
                "y_sd": float(y_use[y].std()),
            })

    res = pd.DataFrame(rows)
    diag = pd.DataFrame(diags)
    res.to_csv(os.path.join(args.out_dir, "10_monitoring_exposure_parentocc_fe_results.csv"), index=False)
    diag.to_csv(os.path.join(args.out_dir, "11_monitoring_exposure_parentocc_fe_diagnostics.csv"), index=False)
    save_json(
        {
            "panel_dir": args.panel_dir,
            "out_dir": args.out_dir,
            "outcomes": available_outcomes,
            "exposures": available_exposures,
            "fixed_effects": fe_cols,
            "note": "Each regression includes parent x occupation FE. log_n_workers is not included on the RHS.",
        },
        os.path.join(args.out_dir, "00_metadata.json"),
    )
    print(f"[INFO] Wrote {len(res):,} regression rows to {args.out_dir}")
    spark.stop()


if __name__ == "__main__":
    main()
