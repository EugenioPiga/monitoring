#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.revelio_analysis_utils import build_analysis_paths, ensure_analysis_directories, ensure_directory, load_json, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot safe-v3 parent-occupation event-study coefficients.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visibility-results-dir", default=None)
    parser.add_argument("--visibility-output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--mode", choices=["base", "visibility", "both"], default="both")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.size": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def estimator_tag(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def plot_base(frame: pd.DataFrame, title: str, subtitle: str) -> plt.Figure:
    ordered = frame.sort_values("event_time").copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        ordered["event_time"],
        ordered["estimate"],
        yerr=1.96 * ordered["std_error"],
        fmt="o-",
        color="#1b4965",
        ecolor="#1b4965",
        elinewidth=1.2,
        capsize=3,
        linewidth=2,
        markersize=5,
    )
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Event time")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"{title}\n{subtitle}")
    return fig


def plot_visibility(frame: pd.DataFrame, title: str, subtitle: str) -> plt.Figure:
    ordered = frame.sort_values("event_time").copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.errorbar(
        ordered["event_time"],
        ordered["estimate"],
        yerr=1.96 * ordered["std_error"],
        fmt="o-",
        color="#8c2f39",
        ecolor="#8c2f39",
        elinewidth=1.2,
        capsize=3,
        linewidth=2,
        markersize=5,
    )
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Event time")
    ax.set_ylabel("Coefficient on event time x 1 SD visibility")
    ax.set_title(title)
    ax.text(
        0.01,
        0.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    return fig


def make_base_figures(coefficients: pd.DataFrame, output_dir: Path) -> None:
    ensure_directory(output_dir)
    appendix_path = output_dir / "event_study_appendix.pdf"
    with PdfPages(appendix_path) as appendix:
        for (estimator, outcome), subset in coefficients.groupby(["estimator", "outcome"], sort=False):
            subset = subset.sort_values("event_time").copy()
            if subset.empty:
                continue
            title = subset["outcome_label"].iloc[0] if "outcome_label" in subset.columns else outcome
            subtitle = subset["estimator_label"].iloc[0] if "estimator_label" in subset.columns else estimator
            fig = plot_base(subset, title, subtitle)
            stem = f"event_study__{estimator_tag(estimator)}__{outcome}"
            fig.savefig(output_dir / f"{stem}.png", dpi=240)
            fig.savefig(output_dir / f"{stem}.pdf")
            appendix.savefig(fig)
            plt.close(fig)


def make_visibility_figures(coefficients: pd.DataFrame, output_dir: Path) -> None:
    ensure_directory(output_dir)
    appendix_path = output_dir / "visibility_event_study_appendix.pdf"
    with PdfPages(appendix_path) as appendix:
        for (estimator, outcome, visibility_variable), subset in coefficients.groupby(["estimator", "outcome", "visibility_variable"], sort=False):
            subset = subset.sort_values("event_time").copy()
            if subset.empty:
                continue
            visibility_label = subset["visibility_label"].iloc[0] if "visibility_label" in subset.columns and subset["visibility_label"].notna().any() else visibility_variable
            outcome_label = subset["outcome_label"].iloc[0] if "outcome_label" in subset.columns else outcome
            title = f"Effect of PA adoption by {visibility_label}"
            subtitle = (
                f"Outcome: {outcome_label}\n"
                "Coefficient is event-time dummy x 1 SD visibility; FE: parent x occupation, parent x year, occupation x year"
            )
            fig = plot_visibility(subset, title, subtitle)
            stem = f"visibility_event_study__{estimator_tag(estimator)}__{outcome}__{estimator_tag(str(visibility_variable))}"
            fig.savefig(output_dir / f"{stem}.png", dpi=240)
            fig.savefig(output_dir / f"{stem}.pdf")
            appendix.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root, output_relative_root=config["output_relative_root"])
    ensure_analysis_directories(paths)
    configure_matplotlib()

    logger = setup_logging("03_make_event_study_figures", paths.logs_root)
    results_dir = Path(args.results_dir) if args.results_dir else paths.results_root
    output_dir = Path(args.output_dir) if args.output_dir else paths.figures_root
    visibility_results_dir = Path(args.visibility_results_dir) if args.visibility_results_dir else paths.visibility_results_root
    visibility_output_dir = Path(args.visibility_output_dir) if args.visibility_output_dir else paths.visibility_figures_root

    if args.mode in {"base", "both"}:
        coefficients_path = results_dir / "02_event_study_coefficients.csv"
        if not coefficients_path.exists():
            raise FileNotFoundError(f"Coefficient table not found: {coefficients_path}")
        coefficients = pd.read_csv(coefficients_path)
        make_base_figures(coefficients, output_dir)
        logger.info("Base event-study figure outputs written to %s", output_dir)

    if args.mode in {"visibility", "both"}:
        coefficients_path = visibility_results_dir / "02_visibility_event_study_coefficients.csv"
        if not coefficients_path.exists():
            raise FileNotFoundError(f"Visibility coefficient table not found: {coefficients_path}")
        coefficients = pd.read_csv(coefficients_path)
        if coefficients.empty:
            raise ValueError(f"Visibility coefficient table is empty: {coefficients_path}")
        make_visibility_figures(coefficients, visibility_output_dir)
        logger.info("Visibility event-study figure outputs written to %s", visibility_output_dir)


if __name__ == "__main__":
    main()
