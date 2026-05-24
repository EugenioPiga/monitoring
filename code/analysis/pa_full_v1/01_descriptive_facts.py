#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    parse_common_args,
    resolve_paths,
    setup_logger,
    load_panel,
    write_manifest,
)

OUTCOMES = [
    "log_n_workers",
    "d5_log_workers",
    "exit_rate",
    "hire_rate",
    "d5_exit_rate",
    "d5_hire_rate",
    "promotion_rate",
    "promotion_rate_continuers",
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
]

EXPOSURE_VIS_VARS = [
    "monitoring_exposure_average",
    "monitoring_exposure_concentration",
    "monitoring_similarity_average",
    "occ_visibility_internal_static",
    "occ_visibility_external_static",
    "pa_visibility_internal_oldformula",
    "pa_visibility_external_oldformula",
    "pa_visibility_internal_loginside",
    "pa_visibility_external_loginside",
    "ai_visibility_internal_oldformula",
    "ai_visibility_external_oldformula",
    "ai_visibility_internal_loginside",
    "ai_visibility_external_loginside",
]

BINSCAT_X_VARS = [
    "occ_visibility_internal_static",
    "occ_visibility_external_static",
    "monitoring_exposure_average",
    "monitoring_exposure_concentration",
]


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _safe_qcut(s: pd.Series, q: int = 4) -> pd.Series:
    try:
        return pd.qcut(s, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.nan, index=s.index)


def _save_hist(series: pd.Series, out_path: Path, title: str) -> None:
    x = series.dropna()
    if x.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=50)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_binned_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    n_bins: int = 20,
    weight_col: str | None = None,
) -> None:
    use = df[[x_col, y_col] + ([weight_col] if weight_col and weight_col in df.columns else [])].dropna()
    if use.empty:
        return
    try:
        use = use.copy()
        use["bin"] = pd.qcut(use[x_col], q=n_bins, duplicates="drop")
        g = use.groupby("bin", observed=True)
        if weight_col and weight_col in use.columns:
            tmp = g.apply(
                lambda z: pd.Series(
                    {
                        "x": np.average(z[x_col], weights=z[weight_col]) if z[weight_col].sum() > 0 else z[x_col].mean(),
                        "y": np.average(z[y_col], weights=z[weight_col]) if z[weight_col].sum() > 0 else z[y_col].mean(),
                        "n": len(z),
                    }
                )
            ).reset_index(drop=True)
        else:
            tmp = g.agg(x=(x_col, "mean"), y=(y_col, "mean"), n=(y_col, "size")).reset_index(drop=True)

        if tmp.empty:
            return

        plt.figure(figsize=(6, 4))
        plt.scatter(tmp["x"], tmp["y"], s=np.clip(tmp["n"], 10, 300), alpha=0.8)
        plt.plot(tmp["x"], tmp["y"], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Binned scatter: {y_col} vs {x_col}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="PA full v1 descriptive facts.")
    parse_common_args(parser)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    paths = resolve_paths(args)
    out_dir = paths.output_root / "01_descriptive_facts"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("01_descriptive_facts", out_dir)
    logger.info("Loading panel from %s", paths.input_panel)

    # Load only needed columns for speed
    need_cols = list(
        {
            "parent_rcid",
            "occupation",
            "year",
            "n_workers",
            "pa_posting_log1p",
            "pa_adoption_count_for_exposure",
            *OUTCOMES,
            *EXPOSURE_VIS_VARS,
        }
    )
    df = load_panel(columns=need_cols, panel_path=paths.input_panel)

    logger.info("Loaded %s rows and %s columns", len(df), len(df.columns))

    # 1) PA adoption over time
    adoption_cols = _available(df, ["year", "pa_posting_log1p", "pa_adoption_count_for_exposure", "parent_rcid"])
    if set(["year", "parent_rcid"]).issubset(adoption_cols):
        adoption = df.groupby("year", as_index=False).agg(
            n_rows=("parent_rcid", "size"),
            n_parents=("parent_rcid", "nunique"),
            mean_pa_posting_log1p=("pa_posting_log1p", "mean") if "pa_posting_log1p" in df.columns else ("parent_rcid", "size"),
            mean_pa_adoption_count=("pa_adoption_count_for_exposure", "mean") if "pa_adoption_count_for_exposure" in df.columns else ("parent_rcid", "size"),
        )
        adoption.to_csv(out_dir / "01_pa_adoption_over_time.csv", index=False)
    else:
        adoption = pd.DataFrame()
        adoption.to_csv(out_dir / "01_pa_adoption_over_time.csv", index=False)

    # 2-3) Distribution summaries + histograms
    dist_rows = []
    for c in _available(df, EXPOSURE_VIS_VARS):
        s = df[c].dropna()
        if s.empty:
            continue
        dist_rows.append(
            {
                "variable": c,
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
        _save_hist(s, fig_dir / f"hist_{c}.png", f"Distribution: {c}")

    pd.DataFrame(dist_rows).to_csv(out_dir / "02_exposure_visibility_distributions.csv", index=False)

    # 4) Correlation matrix among exposure/visibility vars
    corr_vars = _available(df, EXPOSURE_VIS_VARS)
    if corr_vars:
        corr = df[corr_vars].corr(numeric_only=True)
        corr.to_csv(out_dir / "03_corr_exposure_visibility.csv")
    else:
        pd.DataFrame().to_csv(out_dir / "03_corr_exposure_visibility.csv", index=False)

    # 5-7) Outcome means by quartiles
    avail_outcomes = _available(df, OUTCOMES)

    def make_quartile_table(split_var: str, out_name: str) -> None:
        if split_var not in df.columns or not avail_outcomes:
            pd.DataFrame().to_csv(out_dir / out_name, index=False)
            return
        tmp = df[[split_var, *avail_outcomes]].copy()
        tmp["quartile"] = _safe_qcut(tmp[split_var], q=4)
        out = tmp.groupby("quartile", dropna=False, as_index=False)[avail_outcomes].mean(numeric_only=True)
        out.to_csv(out_dir / out_name, index=False)

    make_quartile_table("occ_visibility_internal_static", "04_outcome_means_by_internal_visibility_quartile.csv")
    make_quartile_table("occ_visibility_external_static", "05_outcome_means_by_external_visibility_quartile.csv")
    make_quartile_table("monitoring_exposure_average", "06_outcome_means_by_pa_exposure_quartile.csv")

    # 8-9) Binned scatterplots (unweighted + weighted by n_workers)
    for x in _available(df, BINSCAT_X_VARS):
        for y in avail_outcomes:
            _save_binned_scatter(
                df=df,
                x_col=x,
                y_col=y,
                out_path=fig_dir / f"binscatter_unweighted__{y}__vs__{x}.png",
                n_bins=args.n_bins,
                weight_col=None,
            )
            if "n_workers" in df.columns:
                _save_binned_scatter(
                    df=df,
                    x_col=x,
                    y_col=y,
                    out_path=fig_dir / f"binscatter_weighted_nworkers__{y}__vs__{x}.png",
                    n_bins=args.n_bins,
                    weight_col="n_workers",
                )

    # 10) Manifest
    files = [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()]
    write_manifest(
        out_dir / "manifest.json",
        {
            "module": "01_descriptive_facts",
            "input_panel": str(paths.input_panel),
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "available_outcomes": avail_outcomes,
            "available_exposure_visibility_vars": corr_vars,
            "output_files": files,
        },
    )

    logger.info("Descriptive facts complete. Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
