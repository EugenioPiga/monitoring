#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import parse_common_args, resolve_paths, setup_logger, load_panel, write_manifest

EXPOSURES = [
    "monitoring_exposure_average",
    "monitoring_exposure_concentration",
    "monitoring_similarity_average",
    "pa_visibility_internal_oldformula",
    "pa_visibility_external_oldformula",
    "pa_visibility_internal_loginside",
    "pa_visibility_external_loginside",
    "ai_visibility_internal_oldformula",
    "ai_visibility_external_oldformula",
]

OUTCOME_FAMILIES = {
    "employment_flows": [
        "log_n_workers", "d5_log_workers",
        "exit_rate", "hire_rate", "d5_exit_rate", "d5_hire_rate",
    ],
    "internal_personnel": [
        "promotion_rate", "promotion_rate_continuers", "n_promotions", "n_continuing_workers",
    ],
    "skill_allocation": [
        "skill_count_sd", "skill_bundle_dispersion", "skill_hhi_mean", "specialist_share",
        "d5_skill_count_sd", "d5_skill_bundle_dispersion", "d5_skill_hhi_mean", "d5_specialist_share",
    ],
    "org_structure": [
        "hr_to_employee_ratio", "managers_to_employee_ratio", "n_hr_positions", "n_managers",
    ],
    "wage_margin": [
        "avg_salary", "log_avg_salary", "d5_log_avg_salary",
    ],
}

BAD_CONTROL_OUTCOMES = {
    "log_n_workers", "d5_log_workers", "n_promotions", "n_continuing_workers",
    "n_hr_positions", "n_managers"
}

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

def zscore(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

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

    return {
        "beta": b, "se": se, "t": t, "p": p,
        "nobs": int(n), "n_clusters": int(G)
    }

def run_one(df: pd.DataFrame, outcome: str, exposure: str, weighted: bool, add_size_control: bool):
    cols = ["parent_rcid", "occupation", "year", outcome, exposure]
    if weighted:
        cols.append("n_workers")
    if add_size_control and outcome != "log_n_workers":
        cols.append("log_n_workers")

    use = df[[c for c in cols if c in df.columns]].copy()
    if outcome not in use.columns or exposure not in use.columns:
        return None

    use = use.dropna(subset=["parent_rcid", "occupation", "year", outcome, exposure]).copy()
    if len(use) < 1000:
        return None

    use["parent_rcid"] = use["parent_rcid"].astype(str)
    use["occupation"] = use["occupation"].astype(str)
    use["year"] = use["year"].astype(int)
    use["parent_occ_fe"] = use["parent_rcid"] + "||" + use["occupation"]
    use["year_fe"] = use["year"].astype(str)

    use["x_main"] = zscore(use[exposure])
    if use["x_main"].isna().all():
        return None

    x_cols = ["x_main"]
    if add_size_control and "log_n_workers" in use.columns and outcome not in BAD_CONTROL_OUTCOMES and outcome != "log_n_workers":
        use["x_size"] = zscore(use["log_n_workers"])
        if not use["x_size"].isna().all():
            x_cols.append("x_size")

    use = use.dropna(subset=[outcome] + x_cols).copy()
    if len(use) < 1000:
        return None

    w = np.ones(len(use))
    if weighted:
        if "n_workers" not in use.columns:
            return None
        w = use["n_workers"].fillna(0.0).to_numpy()
        keep = w > 0
        use = use.loc[keep].copy()
        w = w[keep]
        if len(use) < 1000:
            return None

    g1 = use["parent_occ_fe"].to_numpy()
    g2 = use["year_fe"].to_numpy()

    y_res = demean_two_way(use[outcome].to_numpy(), g1, g2, w if weighted else None)
    X_res = np.column_stack([demean_two_way(use[c].to_numpy(), g1, g2, w if weighted else None) for c in x_cols])

    fit = cluster_robust_ols(y_res, X_res, cluster=use["parent_rcid"].to_numpy(), w=w if weighted else None)
    if fit is None:
        return None

    rows = []
    for j, term in enumerate(x_cols):
        rows.append({
            "outcome": outcome,
            "exposure": exposure,
            "term": term,
            "weighted": int(weighted),
            "weight_var": "n_workers" if weighted else "none",
            "add_log_n_workers_control": int(add_size_control),
            "coef": float(fit["beta"][j]),
            "std_error": float(fit["se"][j]),
            "t_stat": float(fit["t"][j]),
            "p_value": float(fit["p"][j]),
            "stars": stars(float(fit["p"][j])),
            "nobs": fit["nobs"],
            "n_clusters": fit["n_clusters"],
            "n_parent_occ": int(use["parent_occ_fe"].nunique()),
            "n_parents": int(use["parent_rcid"].nunique()),
            "n_occupations": int(use["occupation"].nunique()),
            "n_years": int(use["year"].nunique()),
            "fe_spec": "parent_occ_fe + year_fe",
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="PA full v1 exposure-intensity regressions.")
    parse_common_args(parser)
    args = parser.parse_args()

    paths = resolve_paths(args)
    out_dir = paths.output_root / "04_exposure_intensity_regressions"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("04_exposure_intensity_regressions", out_dir)
    logger.info("Loading panel from %s", paths.input_panel)

    schema_cols = [f.name for f in ds.dataset(str(paths.input_panel), format="parquet").schema]
    requested = list({"parent_rcid","occupation","year","n_workers","log_n_workers", *EXPOSURES, *sum(OUTCOME_FAMILIES.values(), [])})
    need_cols = [c for c in requested if c in schema_cols]

    df = load_panel(columns=need_cols, panel_path=paths.input_panel)
    logger.info("Loaded rows=%s cols=%s", len(df), len(df.columns))

    available_exposures = [x for x in EXPOSURES if x in df.columns]
    available_families = {k: [y for y in ys if y in df.columns] for k, ys in OUTCOME_FAMILIES.items()}

    all_rows = []
    for fam, outcomes in available_families.items():
        logger.info("Family=%s outcomes=%s", fam, outcomes)
        fam_rows = []
        for y in outcomes:
            for x in available_exposures:
                # spec A: unweighted no extra control
                r1 = run_one(df, y, x, weighted=False, add_size_control=False)
                if r1 is not None:
                    r1["family"] = fam
                    r1["spec"] = "A_unweighted"
                    fam_rows.append(r1)

                # spec B: weighted by n_workers
                r2 = run_one(df, y, x, weighted=True, add_size_control=False)
                if r2 is not None:
                    r2["family"] = fam
                    r2["spec"] = "B_weighted_n_workers"
                    fam_rows.append(r2)

                # spec C: unweighted + size control (if valid)
                r3 = run_one(df, y, x, weighted=False, add_size_control=True)
                if r3 is not None:
                    r3["family"] = fam
                    r3["spec"] = "C_unweighted_plus_log_n_workers_control"
                    fam_rows.append(r3)

        fam_df = pd.concat(fam_rows, ignore_index=True) if fam_rows else pd.DataFrame()
        fam_df.to_csv(out_dir / f"01_coefficients_{fam}.csv", index=False)
        all_rows.append(fam_df)

    combined = pd.concat([d for d in all_rows if not d.empty], ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(out_dir / "02_coefficients_all_families.csv", index=False)

    if not combined.empty:
        summary = (
            combined[combined["term"] == "x_main"]
            .groupby(["family","outcome","exposure","spec"], as_index=False)
            .agg(
                mean_coef=("coef","mean"),
                mean_abs_t=("t_stat", lambda z: float(np.nanmean(np.abs(z)))),
                min_p=("p_value","min"),
                n_models=("coef","size"),
            )
            .sort_values(["family","outcome","spec","exposure"])
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(out_dir / "03_summary_table.csv", index=False)

    # lightweight "latex-ready" table as csv with display strings
    if not combined.empty:
        disp = combined[combined["term"] == "x_main"].copy()
        disp["coef_se"] = disp.apply(lambda r: f"{r['coef']:.4f}{r['stars']} ({r['std_error']:.4f})", axis=1)
        disp = disp[["family","outcome","exposure","spec","coef_se","nobs","n_clusters"]]
    else:
        disp = pd.DataFrame()
    disp.to_csv(out_dir / "04_regression_table_display.csv", index=False)

    (out_dir / "README.md").write_text(
        "Exposure-intensity regressions (PA full v1)\n"
        "Baseline FE spec implemented: parent_occ_fe + year_fe.\n"
        "SE clustered by parent_rcid.\n"
        "Specs: A unweighted, B weighted(n_workers), C + log_n_workers control where valid.\n"
        "Note: Additional requested FE/clustering variants can be added in next iteration.\n"
    )

    write_manifest(
        out_dir / "manifest.json",
        {
            "module": "04_exposure_intensity_regressions",
            "input_panel": str(paths.input_panel),
            "available_exposures": available_exposures,
            "available_families": available_families,
            "output_files": [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()],
        },
    )

    logger.info("Done. Outputs in %s", out_dir)

if __name__ == "__main__":
    main()
