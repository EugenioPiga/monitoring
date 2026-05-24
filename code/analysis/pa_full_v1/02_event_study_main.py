#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import parse_common_args, resolve_paths, setup_logger, load_panel, write_manifest

OUTCOMES = [
    "log_n_workers",
    "d5_log_workers",
    "exit_rate",
    "hire_rate",
    "d5_exit_rate",
    "d5_hire_rate",
    "promotion_rate",
    "promotion_rate_continuers",
    "n_promotions",
    "n_continuing_workers",
    "skill_count_sd",
    "skill_bundle_dispersion",
    "skill_hhi_mean",
    "specialist_share",
    "d5_skill_count_sd",
    "d5_skill_bundle_dispersion",
    "d5_skill_hhi_mean",
    "d5_specialist_share",
    "hr_to_employee_ratio",
    "managers_to_employee_ratio",
    "n_hr_positions",
    "n_managers",
    "avg_salary",
    "log_avg_salary",
    "d5_log_avg_salary",
]

BIN_MIN = -6
BIN_MAX = 6
OMIT_K = -1

def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pval_from_t(t: float) -> float:
    return 2.0 * (1.0 - ncdf(abs(float(t))))

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def winsorize_series(x: pd.Series, p: float = 0.001) -> pd.Series:
    if x.dropna().empty:
        return x
    lo, hi = x.quantile([p, 1 - p])
    return x.clip(lo, hi)

def add_event_bins(df: pd.DataFrame, event_col: str = "event_time_posting") -> pd.DataFrame:
    out = df.copy()
    out["event_time_binned"] = out[event_col]
    out.loc[out["event_time_binned"] < BIN_MIN, "event_time_binned"] = BIN_MIN
    out.loc[out["event_time_binned"] > BIN_MAX, "event_time_binned"] = BIN_MAX
    return out

def make_event_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[str]]:
    ks = [k for k in range(BIN_MIN, BIN_MAX + 1) if k != OMIT_K]
    out = df.copy()
    cols = []
    for k in ks:
        c = f"ev_{k}"
        out[c] = (out["event_time_binned"] == k).astype(float)
        cols.append(c)
    return out, ks, cols

def demean_two_way(v: np.ndarray, g1: np.ndarray, g2: np.ndarray, w: np.ndarray | None, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    # Alternating weighted demeaning by g1 and g2
    x = v.astype(float).copy()
    prev = np.nan
    if w is None:
        w = np.ones_like(x)
    for _ in range(max_iter):
        # group 1
        s_w = pd.Series(w).groupby(g1).transform("sum").to_numpy()
        s_xw = pd.Series(x * w).groupby(g1).transform("sum").to_numpy()
        m1 = np.divide(s_xw, s_w, out=np.zeros_like(s_xw), where=s_w > 0)
        x = x - m1

        # group 2
        s_w2 = pd.Series(w).groupby(g2).transform("sum").to_numpy()
        s_xw2 = pd.Series(x * w).groupby(g2).transform("sum").to_numpy()
        m2 = np.divide(s_xw2, s_w2, out=np.zeros_like(s_xw2), where=s_w2 > 0)
        x = x - m2

        cur = float(np.nanmean(x * x))
        if np.isfinite(prev) and abs(cur - prev) < tol:
            break
        prev = cur
    return x

def cluster_robust_ols(y: np.ndarray, X: np.ndarray, cluster: np.ndarray, w: np.ndarray | None = None):
    if w is None:
        w = np.ones(len(y))
    w = np.asarray(w, float)
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    cl = np.asarray(cluster)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w)
    y, X, w, cl = y[ok], X[ok], w[ok], cl[ok]

    if len(y) == 0:
        return None

    sw = np.sqrt(np.clip(w, 0, None))
    yw = y * sw
    Xw = X * sw[:, None]

    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.pinv(XtX)
    b = XtX_inv @ (Xw.T @ yw)
    u = y - X @ b

    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = pd.unique(cl)
    for g in groups:
        idx = np.where(cl == g)[0]
        Xg = X[idx] * w[idx][:, None]
        ug = u[idx]
        sg = Xg.T @ ug
        meat += np.outer(sg, sg)

    G = len(groups)
    n, k = X.shape
    corr = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
    p = np.array([pval_from_t(v) if np.isfinite(v) else np.nan for v in t])

    return {
        "beta": b,
        "se": se,
        "t": t,
        "p": p,
        "nobs": int(n),
        "n_clusters": int(G),
        "ok_mask": ok,
        "vcov": V,
    }

def wald_test_pretrend(beta: np.ndarray, V: np.ndarray, ks: list[int]) -> tuple[float, int, float]:
    pre_idx = [i for i, k in enumerate(ks) if k <= -2]
    if len(pre_idx) == 0:
        return np.nan, 0, np.nan
    b = beta[pre_idx]
    VV = V[np.ix_(pre_idx, pre_idx)]
    VV_inv = np.linalg.pinv(VV)
    stat = float(b.T @ VV_inv @ b)
    df = len(pre_idx)
    # chi-square approx p-value via normal approx fallback
    # (for simplicity, use Monte-less approximation if scipy absent)
    # Wilson-Hilferty transform:
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df)) if df > 0 else np.nan
    p = 1 - ncdf(z) if np.isfinite(z) else np.nan
    return stat, df, p

def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def run_spec(df: pd.DataFrame, outcome: str, weight_mode: str) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    cols_needed = ["parent_rcid", "occupation", "year", "event_time_binned", outcome]
    if weight_mode == "n_workers":
        cols_needed.append("n_workers")
    use = df[ensure_cols(df, cols_needed)].copy()

    use = use.dropna(subset=["parent_rcid", "occupation", "year", "event_time_binned", outcome]).copy()
    if len(use) < 500:
        return None, None

    use["parent_rcid"] = use["parent_rcid"].astype(str)
    use["occupation"] = use["occupation"].astype(str)
    use["year"] = use["year"].astype(int)
    use["parent_occ_fe"] = use["parent_rcid"] + "||" + use["occupation"]
    use["year_fe"] = use["year"].astype(str)

    if weight_mode == "n_workers":
        use["w"] = use["n_workers"].fillna(0.0)
        use = use[use["w"] > 0].copy()
    else:
        use["w"] = 1.0

    # winsorize outcome lightly
    use[outcome] = winsorize_series(use[outcome], p=0.001)

    use, ks, ev_cols = make_event_dummies(use)

    # residualize y and event dummies by two-way FE
    g1 = use["parent_occ_fe"].to_numpy()
    g2 = use["year_fe"].to_numpy()
    w = use["w"].to_numpy()

    y_res = demean_two_way(use[outcome].to_numpy(), g1, g2, w if weight_mode == "n_workers" else None)

    X_res_cols = []
    for c in ev_cols:
        xr = demean_two_way(use[c].to_numpy(), g1, g2, w if weight_mode == "n_workers" else None)
        X_res_cols.append(xr)
    X_res = np.column_stack(X_res_cols)

    fit = cluster_robust_ols(y_res, X_res, cluster=use["parent_rcid"].to_numpy(), w=w if weight_mode == "n_workers" else None)
    if fit is None:
        return None, None

    beta, se, t, p = fit["beta"], fit["se"], fit["t"], fit["p"]
    lo = beta - 1.96 * se
    hi = beta + 1.96 * se

    rows = []
    for i, k in enumerate(ks):
        rows.append(
            {
                "outcome": outcome,
                "weight_mode": weight_mode,
                "event_time": int(k),
                "estimate": float(beta[i]),
                "std_error": float(se[i]),
                "t_stat": float(t[i]),
                "p_value": float(p[i]),
                "ci_low_95": float(lo[i]),
                "ci_high_95": float(hi[i]),
                "stars": stars(float(p[i])),
                "nobs": fit["nobs"],
                "n_clusters": fit["n_clusters"],
                "n_parent_occ": int(use["parent_occ_fe"].nunique()),
                "n_parents": int(use["parent_rcid"].nunique()),
                "n_years": int(use["year"].nunique()),
                "fe_spec": "parent_occ_fe + year_fe",
                "omitted_event_time": OMIT_K,
            }
        )
    coef_df = pd.DataFrame(rows)

    stat, df_w, p_w = wald_test_pretrend(beta, fit["vcov"], ks)

    # post summaries
    def post_avg(kmin, kmax):
        idx = [i for i, kk in enumerate(ks) if kk >= kmin and kk <= kmax]
        if not idx:
            return (np.nan, np.nan, np.nan)
        b = float(np.mean(beta[idx]))
        s = float(np.sqrt(np.mean(se[idx] ** 2)))
        tval = b / s if s > 0 else np.nan
        pval = pval_from_t(tval) if np.isfinite(tval) else np.nan
        return (b, s, pval)

    p02 = post_avg(0, 2)
    p04 = post_avg(0, 4)
    p06 = post_avg(0, 6)

    summary = {
        "outcome": outcome,
        "weight_mode": weight_mode,
        "nobs": fit["nobs"],
        "n_clusters": fit["n_clusters"],
        "pretrend_wald_stat": stat,
        "pretrend_df": df_w,
        "pretrend_p_value": p_w,
        "post_avg_0_2": p02[0],
        "post_se_0_2": p02[1],
        "post_p_0_2": p02[2],
        "post_avg_0_4": p04[0],
        "post_se_0_4": p04[1],
        "post_p_0_4": p04[2],
        "post_avg_0_6": p06[0],
        "post_se_0_6": p06[1],
        "post_p_0_6": p06[2],
    }
    return coef_df, summary

def plot_event_study(coef_df: pd.DataFrame, out_path: Path, title: str):
    if coef_df.empty:
        return
    d = coef_df.sort_values("event_time")
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        d["event_time"],
        d["estimate"],
        yerr=1.96 * d["std_error"],
        fmt="o-",
        capsize=3,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Event time")
    plt.ylabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="PA full v1 main event-study estimation.")
    parse_common_args(parser)
    args = parser.parse_args()

    paths = resolve_paths(args)
    out_dir = paths.output_root / "02_event_study_main"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("02_event_study_main", out_dir)
    logger.info("Loading panel from %s", paths.input_panel)

    # IMPORTANT: input panel may not contain all requested outcomes (e.g., wage vars before rebuild).
    # Ask PyArrow only for columns that actually exist.
    import pyarrow.dataset as ds
    schema_cols = [f.name for f in ds.dataset(str(paths.input_panel), format="parquet").schema]

    requested_cols = list(
        {
            "parent_rcid",
            "occupation",
            "year",
            "event_time_posting",
            "n_workers",
            *OUTCOMES,
        }
    )
    need_cols = [c for c in requested_cols if c in schema_cols]

    df = load_panel(columns=need_cols, panel_path=paths.input_panel)
    df = add_event_bins(df, event_col="event_time_posting")

    available_outcomes = [c for c in OUTCOMES if c in df.columns]
    logger.info("Available outcomes: %s", available_outcomes)

    coef_frames = []
    summaries = []

    for outcome in available_outcomes:
        for weight_mode in ["unweighted", "n_workers"]:
            logger.info("Estimating outcome=%s, weight_mode=%s", outcome, weight_mode)
            coef_df, summ = run_spec(df, outcome, weight_mode)
            if coef_df is None:
                logger.warning("Skipped outcome=%s, weight_mode=%s (insufficient support)", outcome, weight_mode)
                continue
            coef_frames.append(coef_df)
            summaries.append(summ)

            plot_event_study(
                coef_df,
                fig_dir / f"event_study__{outcome}__{weight_mode}.png",
                title=f"{outcome} ({weight_mode})",
            )

    if coef_frames:
        coefs = pd.concat(coef_frames, ignore_index=True)
    else:
        coefs = pd.DataFrame()

    summ_df = pd.DataFrame(summaries)

    coefs.to_csv(out_dir / "01_event_study_coefficients.csv", index=False)
    summ_df.to_csv(out_dir / "02_event_study_pretrend_and_postsummary.csv", index=False)

    # Family panel plot (simple): one figure per weight mode, key outcomes
    key_outcomes = [c for c in ["exit_rate", "hire_rate", "promotion_rate", "skill_count_sd", "specialist_share", "log_n_workers"] if c in available_outcomes]
    for wm in ["unweighted", "n_workers"]:
        sub = coefs[coefs["weight_mode"] == wm] if not coefs.empty else pd.DataFrame()
        if sub.empty or not key_outcomes:
            continue
        n = len(key_outcomes)
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow), squeeze=False)
        for i, oc in enumerate(key_outcomes):
            ax = axes[i // ncol][i % ncol]
            d = sub[sub["outcome"] == oc].sort_values("event_time")
            if d.empty:
                ax.axis("off")
                continue
            ax.errorbar(d["event_time"], d["estimate"], yerr=1.96 * d["std_error"], fmt="o-", capsize=3)
            ax.axhline(0, color="black", linewidth=1)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(oc)
            ax.set_xlabel("Event time")
            ax.set_ylabel("Coef")
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        plt.tight_layout()
        plt.savefig(fig_dir / f"panel_key_outcomes__{wm}.png", dpi=150)
        plt.close()

    (out_dir / "README.md").write_text(
        "Main event-study estimates.\n"
        f"Spec: Y_pot = sum_k!=(-1) beta_k 1[event_time=k] + parent_occ_FE + year_FE + e.\n"
        f"Event bins: [{BIN_MIN}, {BIN_MAX}], omitted={OMIT_K}. Cluster by parent_rcid.\n"
        "Weight modes: unweighted, n_workers.\n"
    )

    write_manifest(
        out_dir / "manifest.json",
        {
            "module": "02_event_study_main",
            "input_panel": str(paths.input_panel),
            "n_rows_input": int(len(df)),
            "available_outcomes": available_outcomes,
            "output_files": [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()],
        },
    )

    logger.info("Done. Outputs at %s", out_dir)


if __name__ == "__main__":
    main()
