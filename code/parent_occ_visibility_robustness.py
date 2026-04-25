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
import matplotlib.pyplot as plt
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

OUTCOME_LABELS = {
    "d5_log_workers": "5y log workers",
    "d5_exit_rate": "5y exit rate",
    "d5_hire_rate": "5y hire rate",
    "d5_skill_count_sd": "5y skill-count SD",
    "d5_skill_bundle_dispersion": "5y skill-bundle dispersion",
    "d5_skill_hhi_mean": "5y skill HHI",
    "d5_specialist_share": "5y specialist share",
}

EVENT_LEVEL_OUTCOMES = [
    "exit_rate",
    "hire_rate",
    "skill_count_sd",
    "skill_bundle_dispersion",
    "skill_hhi_mean",
    "specialist_share",
    "n_workers",
]

VISIBILITY_GROUPS = {
    "composite": [
        "mean_user_has_data_skill",
        "mean_user_has_software_skill",
        "mean_user_has_hr_skill",
        "mean_user_has_hr_technology_skill",
        "mean_user_has_employee_feedback_tool_skill",
    ],
    "data_software": [
        "mean_user_has_data_skill",
        "mean_user_has_software_skill",
    ],
    "hr_monitoring": [
        "mean_user_has_hr_skill",
        "mean_user_has_hr_technology_skill",
        "mean_user_has_employee_feedback_tool_skill",
    ],
    "hrtech_feedback": [
        "mean_user_has_hr_technology_skill",
        "mean_user_has_employee_feedback_tool_skill",
    ],
}

SAMPLES = [
    "baseline",
    "min_workers_20",
    "min_workers_50",
    "drop_top_data_software_decile",
    "drop_top_hr_monitoring_decile",
    "drop_top_any_visibility_decile",
]


def parse_args():
    p = argparse.ArgumentParser(description="Memory-safe robustness for parent-occupation visibility analysis.")
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pre-start-year", type=int, default=2016)
    p.add_argument("--pre-end-year", type=int, default=2018)
    p.add_argument("--event-window", type=int, default=3)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-8)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pvalue_from_t(t: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(t)))


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def residualize_twfe(df: pd.DataFrame, cols: List[str], fe1: str, fe2: str, max_iter: int, tol: float) -> pd.DataFrame:
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


def build_visibility_indices(spark, sdf, pre_start: int, pre_end: int):
    raw_cols = sorted(set(sum(VISIBILITY_GROUPS.values(), [])))
    present_raw = [c for c in raw_cols if c in sdf.columns]

    if not present_raw:
        raise ValueError("No visibility proxy columns found in panel.")

    pre = (
        sdf
        .where((F.col("year") >= pre_start) & (F.col("year") <= pre_end))
        .groupBy("occupation")
        .agg(*[F.avg(c).alias(c) for c in present_raw])
    )

    occ = pre.toPandas()
    occ = occ.replace([np.inf, -np.inf], np.nan)

    available_groups = {}
    for name, cols in VISIBILITY_GROUPS.items():
        present = [c for c in cols if c in occ.columns]
        if present:
            available_groups[name] = present

    for c in present_raw:
        sd = occ[c].std()
        if pd.notna(sd) and sd > 0:
            occ[f"z_{c}"] = (occ[c] - occ[c].mean()) / sd
        else:
            occ[f"z_{c}"] = 0.0

    for name, cols in available_groups.items():
        occ[f"visibility_{name}"] = occ[[f"z_{c}" for c in cols]].mean(axis=1)

    if "visibility_data_software" in occ.columns:
        occ["top_decile_data_software"] = (
            occ["visibility_data_software"] >= occ["visibility_data_software"].quantile(0.90)
        ).astype(int)
    else:
        occ["top_decile_data_software"] = 0

    if "visibility_hr_monitoring" in occ.columns:
        occ["top_decile_hr_monitoring"] = (
            occ["visibility_hr_monitoring"] >= occ["visibility_hr_monitoring"].quantile(0.90)
        ).astype(int)
    else:
        occ["top_decile_hr_monitoring"] = 0

    keep = (
        ["occupation"]
        + [c for c in occ.columns if c.startswith("visibility_")]
        + ["top_decile_data_software", "top_decile_hr_monitoring"]
    )

    occ_small = occ[keep].copy()
    occ_spark = spark.createDataFrame(occ_small)

    return occ, occ_spark, available_groups


def apply_sample_filter(sdf, sample_name: str):
    out = sdf

    if sample_name == "baseline":
        return out

    if sample_name == "min_workers_20":
        return out.where(F.col("n_workers") >= 20)

    if sample_name == "min_workers_50":
        return out.where(F.col("n_workers") >= 50)

    if sample_name == "drop_top_data_software_decile":
        return out.where(F.col("top_decile_data_software") != 1)

    if sample_name == "drop_top_hr_monitoring_decile":
        return out.where(F.col("top_decile_hr_monitoring") != 1)

    if sample_name == "drop_top_any_visibility_decile":
        return out.where((F.col("top_decile_data_software") != 1) & (F.col("top_decile_hr_monitoring") != 1))

    raise ValueError(f"Unknown sample: {sample_name}")


def collect_estimation_pdf(sdf, outcome: str, sample_name: str, visibility_names: List[str]) -> pd.DataFrame:
    idx_cols = [f"visibility_{v}" for v in visibility_names if f"visibility_{v}" in sdf.columns]
    base_cols = [
        "parent_rcid",
        "occupation",
        "year",
        "pa_posting_log1p",
        "n_workers",
        outcome,
    ] + idx_cols

    tmp = apply_sample_filter(sdf, sample_name)
    tmp = (
        tmp
        .where(F.col(outcome).isNotNull())
        .where(F.col("pa_posting_log1p").isNotNull())
        .where(F.col("n_workers").isNotNull())
        .select(*base_cols)
    )

    pdf = tmp.toPandas()
    if pdf.empty:
        return pdf

    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    pdf["parent_year_fe"] = pdf["parent_rcid"].astype(str) + "_y" + pdf["year"].astype(int).astype(str)
    pdf["occupation_year_fe"] = pdf["occupation"].astype(str) + "_y" + pdf["year"].astype(int).astype(str)
    pdf["log_n_workers"] = np.where(pdf["n_workers"] > 0, np.log(pdf["n_workers"]), np.nan)
    return pdf


def estimate_from_pdf(
    pdf: pd.DataFrame,
    outcome: str,
    visibility_name: str,
    sample_name: str,
    winsorized: bool,
    max_iter: int,
    tol: float,
) -> List[Dict]:
    idx_col = f"visibility_{visibility_name}"
    if idx_col not in pdf.columns:
        return []

    use = pdf.dropna(
        subset=[outcome, "pa_posting_log1p", idx_col, "log_n_workers", "parent_year_fe", "occupation_year_fe"]
    ).copy()

    if use.empty:
        return []

    if winsorized:
        use[outcome] = winsorize(use[outcome])

    use["treat_visibility"] = use["pa_posting_log1p"] * use[idx_col]

    resid = residualize_twfe(
        use,
        cols=[outcome, "treat_visibility", "log_n_workers"],
        fe1="parent_year_fe",
        fe2="occupation_year_fe",
        max_iter=max_iter,
        tol=tol,
    )

    y = resid[outcome].to_numpy()
    X = resid[["treat_visibility", "log_n_workers"]].to_numpy()

    fit = fit_cluster_ols(y, X, use["parent_rcid"].to_numpy())
    if fit is None:
        return []

    beta, se, t, p, n, G, r2 = fit
    terms = [f"pa_posting_log1p_x_visibility_{visibility_name}", "log_n_workers"]

    rows = []
    for i, term in enumerate(terms):
        rows.append(
            {
                "sample": sample_name,
                "visibility_index": visibility_name,
                "winsorized_outcome": int(winsorized),
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
    return rows


def make_coefplot(results: pd.DataFrame, out_dir: str):
    sub = results[
        (results["sample"] == "baseline")
        & (results["visibility_index"] == "composite")
        & (results["winsorized_outcome"] == 0)
        & (results["term"] == "pa_posting_log1p_x_visibility_composite")
    ].copy()

    if sub.empty:
        return

    sub["label"] = sub["outcome"].map(OUTCOME_LABELS)
    sub = sub.sort_values("coef")
    sub["lo"] = sub["coef"] - 1.96 * sub["std_err"]
    sub["hi"] = sub["coef"] + 1.96 * sub["std_err"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(sub))
    ax.errorbar(sub["coef"], y, xerr=1.96 * sub["std_err"], fmt="o", capsize=3)
    ax.axvline(0, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel(r"Coefficient on $\log(1+\mathrm{PA}) \times$ visibility")
    ax.set_title("Parent-occupation TWFE estimates")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "20_coefplot_baseline_composite.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_binscatter(pdf: pd.DataFrame, outcome: str, visibility_name: str, out_dir: str, max_iter: int, tol: float):
    idx_col = f"visibility_{visibility_name}"
    if idx_col not in pdf.columns:
        return

    use = pdf.dropna(subset=[outcome, "pa_posting_log1p", idx_col, "log_n_workers"]).copy()
    if use.empty:
        return

    use["treat_visibility"] = use["pa_posting_log1p"] * use[idx_col]

    resid = residualize_twfe(
        use,
        cols=[outcome, "treat_visibility", "log_n_workers"],
        fe1="parent_year_fe",
        fe2="occupation_year_fe",
        max_iter=max_iter,
        tol=tol,
    )

    tmp = pd.DataFrame({"x": resid["treat_visibility"], "y": resid[outcome]}).dropna()
    if tmp.empty or tmp["x"].nunique() < 10:
        return

    tmp["bin"] = pd.qcut(tmp["x"], q=20, duplicates="drop")
    b = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean")).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(b["x"], b["y"], s=35)
    z = np.polyfit(b["x"], b["y"], 1)
    xx = np.linspace(b["x"].min(), b["x"].max(), 100)
    ax.plot(xx, z[0] * xx + z[1], linewidth=1.8)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(r"Residualized $\log(1+\mathrm{PA}) \times$ visibility")
    ax.set_ylabel(f"Residualized {OUTCOME_LABELS.get(outcome, outcome)}")
    ax.set_title(f"Residualized binscatter: {OUTCOME_LABELS.get(outcome, outcome)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"21_binscatter_{outcome}_{visibility_name}.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_event_high_low(sdf, level_outcome: str, event_window: int, out_dir: str):
    if level_outcome not in sdf.columns or "visibility_composite" not in sdf.columns or "event_time_posting" not in sdf.columns:
        return

    tmp = (
        sdf
        .where(F.col("event_time_posting").isNotNull())
        .where((F.col("event_time_posting") >= -event_window) & (F.col("event_time_posting") <= event_window))
        .where(F.col(level_outcome).isNotNull())
        .select("event_time_posting", "visibility_composite", level_outcome)
        .toPandas()
    )

    if tmp.empty:
        return

    med = tmp["visibility_composite"].median()
    tmp["visibility_group"] = np.where(tmp["visibility_composite"] >= med, "High visibility", "Low visibility")

    g = (
        tmp.groupby(["event_time_posting", "visibility_group"], as_index=False)[level_outcome]
        .mean()
        .rename(columns={level_outcome: "mean_y"})
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, sub in g.groupby("visibility_group"):
        sub = sub.sort_values("event_time_posting")
        ax.plot(sub["event_time_posting"], sub["mean_y"], marker="o", linewidth=1.8, label=name)

    ax.axvline(0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Event time relative to first PA posting adoption")
    ax.set_ylabel(level_outcome)
    ax.set_title(f"Event-time path by visibility: {level_outcome}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"22_event_high_low_{level_outcome}.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    spark = SparkSession.builder.appName("parent_occ_visibility_robustness_safe").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df0 = spark.read.parquet(args.panel_dir)

    raw_vis_cols = sorted(set(sum(VISIBILITY_GROUPS.values(), [])))
    needed = [
        "parent_rcid",
        "occupation",
        "year",
        "occupation_analysis_sample",
        "pa_posting_log1p",
        "n_workers",
        "event_time_posting",
    ]
    needed += [c for c in raw_vis_cols if c in df0.columns]
    needed += [c for c in OUTCOMES + EVENT_LEVEL_OUTCOMES if c in df0.columns]
    needed = sorted(set([c for c in needed if c in df0.columns]))

    sdf = (
        df0
        .where(F.col("occupation_analysis_sample") == 1)
        .select(*needed)
    )

    occ_idx_pdf, occ_idx_spark, available_groups = build_visibility_indices(
        spark,
        sdf,
        args.pre_start_year,
        args.pre_end_year,
    )

    sdf = sdf.join(occ_idx_spark, on="occupation", how="left").cache()
    _ = sdf.count()

    meta = {
        "panel_dir": args.panel_dir,
        "out_dir": args.out_dir,
        "rows_loaded": sdf.count(),
        "n_parents": sdf.select("parent_rcid").distinct().count(),
        "n_occupations": sdf.select("occupation").distinct().count(),
        "pre_start_year": args.pre_start_year,
        "pre_end_year": args.pre_end_year,
        "available_visibility_groups": available_groups,
        "note": "Memory-safe version: collects one outcome/sample at a time, not the full panel.",
    }
    save_json(meta, os.path.join(args.out_dir, "00_metadata.json"))
    occ_idx_pdf.to_csv(os.path.join(args.out_dir, "01_occupation_visibility_indices.csv"), index=False)

    visibility_names = list(available_groups.keys())
    all_rows = []

    for sample_name in SAMPLES:
        for outcome in OUTCOMES:
            if outcome not in sdf.columns:
                continue

            print(f"[INFO] Collecting sample={sample_name}, outcome={outcome}", flush=True)
            pdf = collect_estimation_pdf(sdf, outcome, sample_name, visibility_names)

            if pdf.empty:
                continue

            for visibility_name in visibility_names:
                all_rows.extend(
                    estimate_from_pdf(
                        pdf,
                        outcome,
                        visibility_name,
                        sample_name,
                        winsorized=False,
                        max_iter=args.max_iter,
                        tol=args.tol,
                    )
                )
                all_rows.extend(
                    estimate_from_pdf(
                        pdf,
                        outcome,
                        visibility_name,
                        sample_name,
                        winsorized=True,
                        max_iter=args.max_iter,
                        tol=args.tol,
                    )
                )

            if sample_name == "baseline" and outcome in ["d5_exit_rate", "d5_hire_rate", "d5_skill_count_sd", "d5_specialist_share"]:
                make_binscatter(pdf, outcome, "composite", args.out_dir, args.max_iter, args.tol)

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(args.out_dir, "10_parent_occ_visibility_robustness.csv"), index=False)

    if not results.empty:
        main = results[
            (results["term"] == "pa_posting_log1p_x_visibility_composite")
            & (results["visibility_index"] == "composite")
            & (results["sample"] == "baseline")
            & (results["winsorized_outcome"] == 0)
        ].copy()
        main.to_csv(os.path.join(args.out_dir, "11_main_baseline_composite.csv"), index=False)

        alt = results[
            (results["sample"] == "baseline")
            & (results["winsorized_outcome"] == 0)
            & (results["term"].str.startswith("pa_posting_log1p_x_visibility_"))
        ].copy()
        alt.to_csv(os.path.join(args.out_dir, "12_alt_visibility_indices.csv"), index=False)

        strict = results[
            (results["visibility_index"] == "composite")
            & (results["winsorized_outcome"] == 0)
            & (results["term"] == "pa_posting_log1p_x_visibility_composite")
        ].copy()
        strict.to_csv(os.path.join(args.out_dir, "13_sample_robustness_composite.csv"), index=False)

        win = results[
            (results["visibility_index"] == "composite")
            & (results["sample"] == "baseline")
            & (results["winsorized_outcome"] == 1)
            & (results["term"] == "pa_posting_log1p_x_visibility_composite")
        ].copy()
        win.to_csv(os.path.join(args.out_dir, "14_winsorized_composite.csv"), index=False)

        make_coefplot(results, args.out_dir)

    for level_outcome in ["exit_rate", "hire_rate", "specialist_share", "skill_bundle_dispersion", "n_workers"]:
        make_event_high_low(sdf, level_outcome, args.event_window, args.out_dir)

    sdf.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
