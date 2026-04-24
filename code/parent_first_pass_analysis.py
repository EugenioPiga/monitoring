#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F


OUTCOMES = [
    "d5_log_workforce",
    "d5_log_avg_salary",
    "d5_hire_rate",
    "d5_exit_rate",
    "d5_skill_count_sd",
    "d5_skill_bundle_dispersion",
]

OUTCOME_LABELS = {
    "d5_log_workforce": "5y log employment growth",
    "d5_log_avg_salary": "5y log average salary growth",
    "d5_hire_rate": "5y change in hire rate",
    "d5_exit_rate": "5y change in exit rate",
    "d5_skill_count_sd": "5y change in skill-count dispersion",
    "d5_skill_bundle_dispersion": "5y change in skill-bundle dispersion",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Economist-style analysis on collapsed parent-year panel.")
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-years", type=int, default=0)
    p.add_argument("--min-avg-workforce", type=float, default=0.0)
    p.add_argument("--event-window", type=int, default=3)
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


def winsorize_series(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)


def load_filtered_base(spark: SparkSession, panel_dir: str, min_years: int, min_avg_workforce: float):
    df = spark.read.parquet(panel_dir)
    df = df.where(F.col("analysis_sample") == 1)

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


def build_yearly_diagnostics(df_spark) -> pd.DataFrame:
    out = (
        df_spark.groupBy("year")
        .agg(
            F.countDistinct("parent_rcid").alias("n_parents"),
            F.avg("workforce_weighted").alias("mean_workforce"),
            F.avg("pa_posting_log1p").alias("mean_pa_posting_log1p"),
            F.avg("has_people_analytics_posting_any_enriched_by_year").alias("share_adopted"),
            F.sum("is_first_people_analytics_posting_year_any_enriched").alias("first_adoptions"),
            F.avg("skill_count_sd").alias("mean_skill_count_sd"),
            F.avg("skill_bundle_dispersion").alias("mean_skill_bundle_dispersion"),
        )
        .orderBy("year")
        .toPandas()
    )
    return out


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
        "first_people_analytics_posting_year_any_enriched",
        "event_time_posting",
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
        pdf["level_control"] = pdf["log_workforce"] if "log_workforce" in pdf.columns else np.nan
    elif outcome == "d5_log_avg_salary":
        if "avg_salary" in pdf.columns:
            pdf["level_control"] = np.where(pdf["avg_salary"] > 0, np.log(pdf["avg_salary"]), np.nan)
        else:
            pdf["level_control"] = np.nan
    else:
        pdf["level_control"] = pdf["log_workforce"] if "log_workforce" in pdf.columns else np.nan

    # Keep only one support indicator; the two were basically collinear in your earlier run
    pdf["support_control"] = pdf["has_position_data"] if "has_position_data" in pdf.columns else 1.0

    pdf["fe_ind_year"] = pdf["naics3"].astype(str) + "_y" + pdf["year"].astype(int).astype(str)
    return pdf


def fit_ols_clustered(pdf: pd.DataFrame, outcome: str, with_fe: bool, standardized: bool = False, winsorize_y: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    use = pdf.dropna(subset=[outcome, "pa_posting_log1p", "level_control"]).copy()

    if winsorize_y:
        use[outcome] = winsorize_series(use[outcome])

    if standardized:
        x_sd = use["pa_posting_log1p"].std()
        if pd.notna(x_sd) and x_sd > 0:
            use["pa_posting_log1p"] = (use["pa_posting_log1p"] - use["pa_posting_log1p"].mean()) / x_sd

    rhs = ["pa_posting_log1p", "level_control", "support_control"]
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
        return pd.DataFrame(), pd.DataFrame()

    XtX = Xmat.T @ Xmat
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xmat.T @ y)
    resid = y - Xmat @ beta

    unique_groups = pd.unique(groups)
    meat = np.zeros((k, k))
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        Xg = Xmat[idx, :]
        ug = resid[idx]
        Xgu = Xg.T @ ug
        meat += np.outer(Xgu, Xgu)

    G = len(unique_groups)
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if (G > 1 and n > k) else 1.0
    V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    tvals = beta / se
    pvals = np.array([two_sided_p_from_z(t) if np.isfinite(t) else np.nan for t in tvals])

    ybar = np.mean(y)
    tss = np.sum((y - ybar) ** 2)
    rss = np.sum(resid ** 2)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    model_name = "OLS_FE"
    if standardized:
        model_name += "_STD"
    if winsorize_y:
        model_name += "_WIN"

    out_rows = []
    for i, name in enumerate(X.columns):
        if name in ["const"] or name.startswith("fe_"):
            continue
        out_rows.append(
            {
                "model": model_name if with_fe else model_name.replace("OLS_FE", "OLS"),
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

    diagnostics = pd.DataFrame(
        [{
            "outcome": outcome,
            "model": model_name if with_fe else model_name.replace("OLS_FE", "OLS"),
            "nobs": int(n),
            "n_clusters": int(G),
            "r2": float(r2),
            "y_mean": float(np.mean(y)),
            "y_sd": float(np.std(y)),
        }]
    )

    return pd.DataFrame(out_rows), diagnostics


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return y - X @ beta


def make_binscatter(pdf: pd.DataFrame, outcome: str, path: str) -> None:
    use = pdf.dropna(subset=[outcome, "pa_posting_log1p", "level_control"]).copy()
    if use.empty:
        return

    # Residualize outcome and treatment on controls + FE
    controls = use[["level_control", "support_control"]].copy()
    fe = pd.get_dummies(use["fe_ind_year"], prefix="fe", drop_first=True, dtype=float)
    X = pd.concat([pd.DataFrame({"const": np.ones(len(use))}), controls, fe], axis=1).to_numpy(dtype=float)

    y_res = residualize(use[outcome].astype(float).to_numpy(), X)
    x_res = residualize(use["pa_posting_log1p"].astype(float).to_numpy(), X)

    tmp = pd.DataFrame({"x_res": x_res, "y_res": y_res}).dropna()
    if tmp.empty:
        return

    tmp["bin"] = pd.qcut(tmp["x_res"], q=20, duplicates="drop")
    b = tmp.groupby("bin", as_index=False).agg(x=("x_res", "mean"), y=("y_res", "mean"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(b["x"], b["y"], s=35)
    z = np.polyfit(b["x"], b["y"], 1)
    xx = np.linspace(b["x"].min(), b["x"].max(), 100)
    yy = z[0] * xx + z[1]
    ax.plot(xx, yy, linewidth=1.8)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Residualized adoption intensity")
    ax.set_ylabel(f"Residualized {OUTCOME_LABELS[outcome]}")
    ax.set_title(f"Residualized binscatter: {OUTCOME_LABELS[outcome]}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_event_study_plot(pdf: pd.DataFrame, outcome: str, event_window: int, path: str) -> None:
    if "event_time_posting" not in pdf.columns:
        return

    use = pdf.dropna(subset=["event_time_posting", outcome]).copy()
    use = use[(use["event_time_posting"] >= -event_window) & (use["event_time_posting"] <= event_window)]
    use = use[use["first_people_analytics_posting_year_any_enriched"].notna()].copy()

    if use.empty:
        return

    grouped = (
        use.groupby("event_time_posting")[outcome]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "y_mean", "std": "y_sd", "count": "n"})
    )
    grouped["se"] = grouped["y_sd"] / np.sqrt(np.maximum(grouped["n"], 1))
    grouped["lo"] = grouped["y_mean"] - 1.96 * grouped["se"]
    grouped["hi"] = grouped["y_mean"] + 1.96 * grouped["se"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grouped["event_time_posting"], grouped["y_mean"], marker="o", linewidth=1.8)
    ax.fill_between(grouped["event_time_posting"], grouped["lo"], grouped["hi"], alpha=0.2)
    ax.axvline(0, linestyle="--", linewidth=1.0)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Event time relative to first enriched posting adoption")
    ax.set_ylabel(OUTCOME_LABELS[outcome])
    ax.set_title(f"Descriptive event-time path: {OUTCOME_LABELS[outcome]}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_diagnostics(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["year"], df["share_adopted"], marker="o", linewidth=1.8, label="Share adopted")
    ax.plot(df["year"], df["mean_pa_posting_log1p"], marker="o", linewidth=1.8, label="Mean log(1+PA)")
    ax2 = ax.twinx()
    ax2.plot(df["year"], df["first_adoptions"], marker="s", linewidth=1.5, linestyle="--", label="First adoptions")

    ax.set_xlabel("Year")
    ax.set_ylabel("Adoption intensity / share")
    ax2.set_ylabel("First adoptions")
    ax.set_title("Adoption diffusion in the estimation sample")
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_summary(results: pd.DataFrame, path: str, model_name: str = "OLS_FE") -> None:
    sub = results[(results["model"] == model_name) & (results["term"] == "pa_posting_log1p")].copy()
    if sub.empty:
        return

    sub["label"] = sub["outcome"].map(OUTCOME_LABELS)
    sub = sub.sort_values("coef")
    sub["lo"] = sub["coef"] - 1.96 * sub["std_err"]
    sub["hi"] = sub["coef"] + 1.96 * sub["std_err"]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(sub))
    ax.errorbar(sub["coef"], y, xerr=1.96 * sub["std_err"], fmt="o", capsize=3)
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel("Coefficient on log(1 + enriched PA postings)")
    ax.set_title("Baseline FE estimates across outcomes")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    spark = SparkSession.builder.appName("parent_first_pass_analysis").getOrCreate()
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
        "event_window": args.event_window,
    }
    save_json(meta, os.path.join(args.out_dir, "00_analysis_metadata.json"))

    yearly = build_yearly_diagnostics(df)
    yearly.to_csv(os.path.join(args.out_dir, "01_yearly_diagnostics.csv"), index=False)
    plot_yearly_diagnostics(yearly, os.path.join(args.out_dir, "01_adoption_diffusion.png"))

    all_results = []
    all_diag = []

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue

        pdf = prepare_outcome_pdf(df, outcome)
        if pdf.empty:
            continue

        # Baseline FE
        res, diag = fit_ols_clustered(pdf, outcome, with_fe=True, standardized=False, winsorize_y=False)
        if not res.empty:
            all_results.append(res)
            all_diag.append(diag)

        # Standardized treatment FE
        res_std, diag_std = fit_ols_clustered(pdf, outcome, with_fe=True, standardized=True, winsorize_y=False)
        if not res_std.empty:
            all_results.append(res_std)
            all_diag.append(diag_std)

        # Winsorized outcome FE
        res_win, diag_win = fit_ols_clustered(pdf, outcome, with_fe=True, standardized=False, winsorize_y=True)
        if not res_win.empty:
            all_results.append(res_win)
            all_diag.append(diag_win)

        # OLS without FE as benchmark
        res_nofe, diag_nofe = fit_ols_clustered(pdf, outcome, with_fe=False, standardized=False, winsorize_y=False)
        if not res_nofe.empty:
            all_results.append(res_nofe)
            all_diag.append(diag_nofe)

        # Graphs for the most informative outcomes
        if outcome in ["d5_log_workforce", "d5_exit_rate", "d5_skill_count_sd"]:
            make_binscatter(pdf, outcome, os.path.join(args.out_dir, f"binscatter_{outcome}.png"))
            make_event_study_plot(pdf, outcome, args.event_window, os.path.join(args.out_dir, f"event_{outcome}.png"))

    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        results.to_csv(os.path.join(args.out_dir, "10_results_all_specs.csv"), index=False)
        plot_coefficient_summary(results, os.path.join(args.out_dir, "11_coefplot_baseline_fe.png"), model_name="OLS_FE")

    if all_diag:
        pd.concat(all_diag, ignore_index=True).to_csv(os.path.join(args.out_dir, "12_model_diagnostics.csv"), index=False)

    spark.stop()


if __name__ == "__main__":
    main()
