#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"

import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from plotting.revelio_plotting import configure_matplotlib, plot_comparison_event_study, plot_single_event_study, save_figure
from utils.revelio_analysis_utils import build_analysis_paths, ensure_analysis_directories, ensure_directory, load_json, setup_logging
from utils.revelio_event_study_design import outcome_frame


BASELINE_SPEC = "spec1_main_twfe"
COMPARISON_SPECS = {
    "spec1_main_twfe": "Main treatment",
    "spec4_position_twfe": "Position treatment",
    "spec4_posting_twfe": "Posting treatment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create event-study figures from regression output tables.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    return parser.parse_args()


def ensure_required_files(results_dir: Path) -> Path:
    coefficients_path = results_dir / "twfe_event_study_coefficients.csv"
    if not coefficients_path.exists():
        raise FileNotFoundError(f"Coefficient table not found: {coefficients_path}")
    return coefficients_path


def build_group_grid(frame: pd.DataFrame, outcome_specs: pd.DataFrame, title: str) -> plt.Figure:
    n_plots = len(outcome_specs)
    n_cols = 2
    n_rows = int(math.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.25 * n_rows))
    axes = axes.flatten()
    for ax in axes:
        ax.set_visible(False)

    for idx, (_, outcome_row) in enumerate(outcome_specs.iterrows()):
        ax = axes[idx]
        ax.set_visible(True)
        subset = frame[frame["outcome"] == outcome_row["name"]].sort_values("event_time")
        if subset.empty:
            ax.text(0.5, 0.5, "No estimates", ha="center", va="center")
            ax.set_title(outcome_row["label"])
            continue
        ax.fill_between(subset["event_time"], subset["ci_low"], subset["ci_high"], color="#1b4965", alpha=0.18)
        ax.plot(subset["event_time"], subset["estimate"], marker="o", color="#1b4965", linewidth=2)
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.axvline(-1.0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(outcome_row["label"])
        ax.set_xlabel("Event time")
        ax.set_ylabel("Estimate")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def write_group_outputs(
    *,
    group_name: str,
    coefficients: pd.DataFrame,
    outcome_specs: pd.DataFrame,
    output_dir: Path,
    appendix: PdfPages,
) -> None:
    group_frame = coefficients[coefficients["outcome"].isin(outcome_specs["name"])].copy()
    if group_frame.empty:
        return
    grid = build_group_grid(group_frame, outcome_specs, f"{group_name.title()} outcomes | Main treatment")
    save_figure(grid, output_dir / f"{group_name}_outcomes_main_treatment.png", output_dir / f"{group_name}_outcomes_main_treatment.pdf")
    appendix.savefig(grid)
    plt.close(grid)

    for _, outcome_row in outcome_specs.iterrows():
        subset = group_frame[group_frame["outcome"] == outcome_row["name"]].copy()
        if subset.empty:
            continue
        figure = plot_single_event_study(
            subset,
            title=outcome_row["label"],
            outcome_label="Estimate",
            subtitle="Main treatment | Spec 1",
        )
        save_figure(
            figure,
            output_dir / f"event_study__{group_name}__{outcome_row['name']}.png",
            output_dir / f"event_study__{group_name}__{outcome_row['name']}.pdf",
        )
        appendix.savefig(figure)
        plt.close(figure)


def write_comparison_outputs(coefficients: pd.DataFrame, output_dir: Path, appendix: PdfPages) -> None:
    comparison = coefficients[coefficients["spec_id"].isin(COMPARISON_SPECS)].copy()
    if comparison.empty:
        return
    comparison["comparison_label"] = comparison["spec_id"].map(COMPARISON_SPECS)
    chosen_outcomes = ["exit_rate", "hire_rate", "log_workforce", "avg_seniority"]
    comparison = comparison[comparison["outcome"].isin(chosen_outcomes)].copy()
    if comparison.empty:
        return

    pages = []
    for outcome_name, outcome_frame in comparison.groupby("outcome", sort=False):
        outcome_label = outcome_frame["outcome_label"].iloc[0]
        figure = plot_comparison_event_study(
            outcome_frame,
            title=f"Treatment-definition comparison | {outcome_label}",
            outcome_label="Estimate",
            hue_col="comparison_label",
        )
        save_figure(
            figure,
            output_dir / f"treatment_comparison__{outcome_name}.png",
            output_dir / f"treatment_comparison__{outcome_name}.pdf",
        )
        appendix.savefig(figure)
        pages.append(figure)
        plt.close(figure)

    if pages:
        summary = comparison.copy()
        outcome_order = {name: idx for idx, name in enumerate(chosen_outcomes)}
        summary["outcome_order"] = summary["outcome"].map(outcome_order)
        unique_outcomes = [name for name in chosen_outcomes if name in set(summary["outcome"])]
        fig, axes = plt.subplots(len(unique_outcomes), 1, figsize=(8.5, 4 * len(unique_outcomes)))
        if len(unique_outcomes) == 1:
            axes = [axes]
        for ax, outcome_name in zip(axes, unique_outcomes):
            sub = summary[summary["outcome"] == outcome_name].copy()
            for label, spec_frame in sub.groupby("comparison_label", sort=False):
                ordered = spec_frame.sort_values("event_time")
                ax.plot(ordered["event_time"], ordered["estimate"], marker="o", linewidth=2, label=label)
                ax.fill_between(ordered["event_time"], ordered["ci_low"], ordered["ci_high"], alpha=0.12)
            ax.axhline(0.0, color="#333333", linewidth=1)
            ax.axvline(-1.0, color="#777777", linestyle="--", linewidth=1)
            ax.set_title(sub["outcome_label"].iloc[0])
            ax.set_xlabel("Event time")
            ax.set_ylabel("Estimate")
        axes[0].legend(loc="best")
        fig.suptitle("Treatment-definition comparisons", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_figure(fig, output_dir / "treatment_definition_comparisons.png", output_dir / "treatment_definition_comparisons.pdf")
        appendix.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = build_analysis_paths(args.project_root)
    ensure_analysis_directories(paths)
    configure_matplotlib()

    logger = setup_logging("03_make_event_study_figures", paths.logs_root)
    results_dir = Path(args.results_dir) if args.results_dir else paths.event_study_root / "results"
    output_dir = Path(args.output_dir) if args.output_dir else paths.figures_root / "event_study"
    ensure_directory(output_dir)

    coefficients_path = ensure_required_files(results_dir)
    coefficients = pd.read_csv(coefficients_path)
    coefficients = coefficients[coefficients["spec_id"] == BASELINE_SPEC].copy()
    config = load_json(args.config_path)
    outcomes = outcome_frame(config)

    appendix_path = output_dir / "event_study_appendix.pdf"
    with PdfPages(appendix_path) as appendix:
        write_group_outputs(
            group_name="primary",
            coefficients=coefficients[coefficients["outcome_group"] == "primary"].copy(),
            outcome_specs=outcomes[outcomes["group"] == "primary"].copy(),
            output_dir=output_dir,
            appendix=appendix,
        )
        write_group_outputs(
            group_name="wages",
            coefficients=coefficients[coefficients["outcome_group"] == "wages"].copy(),
            outcome_specs=outcomes[outcomes["group"] == "wages"].copy(),
            output_dir=output_dir,
            appendix=appendix,
        )
        write_group_outputs(
            group_name="composition",
            coefficients=coefficients[coefficients["outcome_group"] == "composition"].copy(),
            outcome_specs=outcomes[outcomes["group"] == "composition"].copy(),
            output_dir=output_dir,
            appendix=appendix,
        )
        comparison_coefficients = pd.read_csv(coefficients_path)
        write_comparison_outputs(comparison_coefficients, output_dir, appendix)

    logger.info("Figure outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
