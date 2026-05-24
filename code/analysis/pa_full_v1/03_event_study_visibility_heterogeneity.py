#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as ds

from common import parse_common_args, resolve_paths, setup_logger, load_panel, write_manifest

OUTCOMES = [
    "exit_rate",
    "hire_rate",
    "promotion_rate",
    "promotion_rate_continuers",
    "skill_count_sd",
    "skill_bundle_dispersion",
    "skill_hhi_mean",
    "specialist_share",
    "log_n_workers",
]

BIN_MIN = -6
BIN_MAX = 6
OMIT_K = -1

def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pval_from_t(t: float) -> float:
    return 2.0 * (1.0 - ncdf(abs(float(t))))

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

def add_event_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_time_binned"] = out["event_time_posting"]
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

def demean_two_way(v: np.ndarray, g1: np.ndarray, g2: np.ndarray, w: np.ndarray | None, max_iter=50, tol=1e-10) -> np.ndarray:
    x = v.astype(float).copy()
    if w is None:
        w = np.ones_like(x)
    prev = np.nan
    for _ in range(max_iter):
        sw1 = pd.Series(w).groupby(g1).transform("sum").to_numpy()
        sx1 = pd.Series(x * w).groupby(g1).transform("sum").to_numpy()
        m1 = np.divide(sx1, sw1, out=np.zeros_like(sx1), where=sw1 > 0)
        x = x - m1

        sw2 = pd.Series(w).groupby(g2).transform("sum").to_numpy()
        sx2 = pd.Series(x * w).groupby(g2).transform("sum").to_numpy()
        m2 = np.divide(sx2, sw2, out=np.zeros_like(sx2), where=sw2 > 0)
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

    return {"beta": b, "se": se, "t": t, "p": p, "nobs": int(n), "n_clusters": int(G), "vcov": V}

def wald_pretrend(beta: np.ndarray, V: np.ndarray, ks: list[int], block_offset: int, block_len: int):
    # test all pre-treatment interaction coefficients (k <= -2) within one block
    idx = [block_offset + i for i, k in enumerate(ks) if k <= -2 and i < block_len]
    if not idx:
        return (np.nan, 0, np.nan)
    b = beta[idx]
    VV = V[np.ix_(idx, idx)]
    VV_inv = np.linalg.pinv(VV)
    stat = float(b.T @ VV_inv @ b)
    df = len(idx)
    z = ((stat / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df)) if df > 0 else np.nan
    p = 1 - ncdf(z) if np.isfinite(z) else np.nan
    return stat, df, p

def zscore(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def plot_theta(df: pd.DataFrame, out_path: Path, title: str):
    if df.empty:
        return
    d = df.sort_values("event_time")
    plt.figure(figsize=(7,4))
    plt.errorbar(d["event_time"], d["estimate"], yerr=1.96*d["std_error"], fmt="o-", capsize=3)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Event time")
    plt.ylabel("Interaction coefficient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_model(use: pd.DataFrame, outcome: str, reg_type: str):
    use = use.dropna(subset=[outcome, "event_time_binned", "parent_rcid", "occupation", "year"]).copy()
    if len(use) < 1000:
        return None, None

    use["parent_rcid"] = use["parent_rcid"].astype(str)
    use["occupation"] = use["occupation"].astype(str)
    use["year"] = use["year"].astype(int)
    use["parent_occ_fe"] = use["parent_rcid"] + "||" + use["occupation"]
    use["year_fe"] = use["year"].astype(str)

    use, ks, ev_cols = make_event_dummies(use)

    if reg_type == "high_internal":
        if "high_internal" not in use.columns:
            return None, None
        inter_cols = []
        for c in ev_cols:
            ic = f"{c}_x_hi_int"
            use[ic] = use[c] * use["high_internal"]
            inter_cols.append(ic)
        X_cols = inter_cols
        label = "event_time x high_internal"

    elif reg_type == "high_external":
        if "high_external" not in use.columns:
            return None, None
        inter_cols = []
        for c in ev_cols:
            ic = f"{c}_x_hi_ext"
            use[ic] = use[c] * use["high_external"]
            inter_cols.append(ic)
        X_cols = inter_cols
        label = "event_time x high_external"

    elif reg_type == "z_internal":
        if "z_internal" not in use.columns:
            return None, None
        inter_cols = []
        for c in ev_cols:
            ic = f"{c}_x_zint"
            use[ic] = use[c] * use["z_internal"]
            inter_cols.append(ic)
        X_cols = inter_cols
        label = "event_time x z_internal"

    elif reg_type == "z_external":
        if "z_external" not in use.columns:
            return None, None
        inter_cols = []
        for c in ev_cols:
            ic = f"{c}_x_zext"
            use[ic] = use[c] * use["z_external"]
            inter_cols.append(ic)
        X_cols = inter_cols
        label = "event_time x z_external"

    elif reg_type == "joint_z":
        if "z_internal" not in use.columns or "z_external" not in use.columns:
            return None, None
        inter_cols = []
        for c in ev_cols:
            i1 = f"{c}_x_zint"
            i2 = f"{c}_x_zext"
            use[i1] = use[c] * use["z_internal"]
            use[i2] = use[c] * use["z_external"]
            inter_cols.extend([i1, i2])
        X_cols = inter_cols
        label = "event_time x z_internal + event_time x z_external"
    else:
        return None, None

    g1 = use["parent_occ_fe"].to_numpy()
    g2 = use["year_fe"].to_numpy()
    w = np.ones(len(use))

    y_res = demean_two_way(use[outcome].to_numpy(), g1, g2, w)
    X_res = np.column_stack([demean_two_way(use[c].to_numpy(), g1, g2, w) for c in X_cols])

    fit = cluster_robust_ols(y_res, X_res, cluster=use["parent_rcid"].to_numpy(), w=w)
    if fit is None:
        return None, None

    rows = []
    if reg_type == "joint_z":
        # split into two blocks
        n_k = len(ks)
        for block_name, offset in [("z_internal", 0), ("z_external", n_k)]:
            for i, k in enumerate(ks):
                j = offset + i
                rows.append({
                    "outcome": outcome,
                    "reg_type": reg_type,
                    "component": block_name,
                    "event_time": int(k),
                    "estimate": float(fit["beta"][j]),
                    "std_error": float(fit["se"][j]),
                    "t_stat": float(fit["t"][j]),
                    "p_value": float(fit["p"][j]),
                    "stars": stars(float(fit["p"][j])),
                    "nobs": fit["nobs"],
                    "n_clusters": fit["n_clusters"],
                    "spec_label": label,
                })
            stat, ddf, pp = wald_pretrend(fit["beta"], fit["vcov"], ks, offset, len(ks))
            rows.append({
                "outcome": outcome,
                "reg_type": reg_type,
                "component": block_name,
                "event_time": 999,
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "stars": "",
                "nobs": fit["nobs"],
                "n_clusters": fit["n_clusters"],
                "spec_label": f"PRETREND_TEST stat={stat:.4f}, df={ddf}, p={pp:.4g}",
            })
    else:
        for i, k in enumerate(ks):
            rows.append({
                "outcome": outcome,
                "reg_type": reg_type,
                "component": reg_type,
                "event_time": int(k),
                "estimate": float(fit["beta"][i]),
                "std_error": float(fit["se"][i]),
                "t_stat": float(fit["t"][i]),
                "p_value": float(fit["p"][i]),
                "stars": stars(float(fit["p"][i])),
                "nobs": fit["nobs"],
                "n_clusters": fit["n_clusters"],
                "spec_label": label,
            })
        stat, ddf, pp = wald_pretrend(fit["beta"], fit["vcov"], ks, 0, len(ks))
        rows.append({
            "outcome": outcome,
            "reg_type": reg_type,
            "component": reg_type,
            "event_time": 999,
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "stars": "",
            "nobs": fit["nobs"],
            "n_clusters": fit["n_clusters"],
            "spec_label": f"PRETREND_TEST stat={stat:.4f}, df={ddf}, p={pp:.4g}",
        })

    out = pd.DataFrame(rows)
    return out, use

def main():
    parser = argparse.ArgumentParser(description="Visibility heterogeneity event studies (PA full v1).")
    parse_common_args(parser)
    args = parser.parse_args()

    paths = resolve_paths(args)
    out_dir = paths.output_root / "03_event_study_visibility_heterogeneity"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("03_event_study_visibility_heterogeneity", out_dir)
    logger.info("Loading panel from %s", paths.input_panel)

    schema_cols = [f.name for f in ds.dataset(str(paths.input_panel), format="parquet").schema]
    needed = [c for c in list({
        "parent_rcid","occupation","year","event_time_posting",
        "occ_visibility_internal_static","occ_visibility_external_static",
        *OUTCOMES
    }) if c in schema_cols]
    df = load_panel(columns=needed, panel_path=paths.input_panel)
    df = add_event_bins(df)

    # Build heterogeneity vars
    if "occ_visibility_internal_static" in df.columns:
        med_int = df["occ_visibility_internal_static"].median()
        df["high_internal"] = (df["occ_visibility_internal_static"] >= med_int).astype(float)
        df["z_internal"] = zscore(df["occ_visibility_internal_static"])
    if "occ_visibility_external_static" in df.columns:
        med_ext = df["occ_visibility_external_static"].median()
        df["high_external"] = (df["occ_visibility_external_static"] >= med_ext).astype(float)
        df["z_external"] = zscore(df["occ_visibility_external_static"])

    available_outcomes = [o for o in OUTCOMES if o in df.columns]
    logger.info("Available outcomes for heterogeneity: %s", available_outcomes)

    all_rows = []
    specs = ["high_internal", "high_external", "z_internal", "z_external", "joint_z"]

    for y in available_outcomes:
        for spec in specs:
            logger.info("Estimating outcome=%s spec=%s", y, spec)
            out, used = run_model(df.copy(), y, spec)
            if out is None:
                logger.warning("Skipped outcome=%s spec=%s", y, spec)
                continue
            all_rows.append(out)

            # plot only coefficient rows (event_time != 999)
            if spec == "joint_z":
                for comp in ["z_internal","z_external"]:
                    pp = out[(out["event_time"] != 999) & (out["component"] == comp)].copy()
                    plot_theta(pp, fig_dir / f"theta__{y}__{spec}__{comp}.png", f"{y}: {comp}")
            else:
                pp = out[out["event_time"] != 999].copy()
                plot_theta(pp, fig_dir / f"theta__{y}__{spec}.png", f"{y}: {spec}")

    final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    final.to_csv(out_dir / "01_visibility_heterogeneity_coefficients.csv", index=False)

    # Summary ranking table: avg post (0..4) internal vs external
    rank_rows = []
    if not final.empty:
        for y in available_outcomes:
            sub_i = final[(final["outcome"] == y) & (final["reg_type"] == "z_internal") & (final["event_time"].between(0,4))]
            sub_e = final[(final["outcome"] == y) & (final["reg_type"] == "z_external") & (final["event_time"].between(0,4))]
            if not sub_i.empty and not sub_e.empty:
                bi = sub_i["estimate"].mean()
                be = sub_e["estimate"].mean()
                rank_rows.append({
                    "outcome": y,
                    "mean_post_0_4_internal_z": bi,
                    "mean_post_0_4_external_z": be,
                    "internal_minus_external": bi - be,
                })
    pd.DataFrame(rank_rows).sort_values("internal_minus_external", ascending=False).to_csv(
        out_dir / "02_internal_vs_external_post_effect_ranking.csv", index=False
    )

    (out_dir / "README.md").write_text(
        "Visibility heterogeneity event-study module.\n"
        "FE: parent_occ + year. Cluster: parent_rcid.\n"
        "Specs: high_internal, high_external, z_internal, z_external, joint_z.\n"
        "Event bins: -6..6, omit -1.\n"
    )

    write_manifest(
        out_dir / "manifest.json",
        {
            "module": "03_event_study_visibility_heterogeneity",
            "input_panel": str(paths.input_panel),
            "available_outcomes": available_outcomes,
            "output_files": [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()],
        },
    )

    logger.info("Done. Outputs in %s", out_dir)

if __name__ == "__main__":
    main()
