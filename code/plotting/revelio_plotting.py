from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.size": 10,
            "grid.alpha": 0.25,
            "grid.color": "#666666",
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, png_path: str | Path, pdf_path: str | Path | None = None, dpi: int = 240) -> None:
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path)


def plot_single_event_study(
    frame: pd.DataFrame,
    *,
    title: str,
    outcome_label: str,
    subtitle: str | None = None,
    color: str = "#1b4965",
) -> plt.Figure:
    ordered = frame.sort_values("event_time").copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(ordered["event_time"], ordered["ci_low"], ordered["ci_high"], color=color, alpha=0.18)
    ax.plot(ordered["event_time"], ordered["estimate"], color=color, marker="o", linewidth=2)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.axvline(-1.0, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("Event time")
    ax.set_ylabel(outcome_label)
    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title)
    return fig


def plot_comparison_event_study(
    frame: pd.DataFrame,
    *,
    title: str,
    outcome_label: str,
    hue_col: str,
) -> plt.Figure:
    palette = ["#1b4965", "#ca6702", "#2a9d8f", "#7f5539"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for idx, (series_name, series_frame) in enumerate(frame.groupby(hue_col, sort=False)):
        ordered = series_frame.sort_values("event_time")
        color = palette[idx % len(palette)]
        ax.plot(ordered["event_time"], ordered["estimate"], marker="o", linewidth=2, color=color, label=str(series_name))
        ax.fill_between(ordered["event_time"], ordered["ci_low"], ordered["ci_high"], color=color, alpha=0.12)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.axvline(-1.0, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("Event time")
    ax.set_ylabel(outcome_label)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig
