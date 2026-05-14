from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from utils.revelio_analysis_utils import year_from_any


def outcome_frame(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(config.get("outcomes", []))
    if frame.empty:
        return pd.DataFrame(columns=["name", "group", "label", "priority"])
    return frame.sort_values(["priority", "name"]).reset_index(drop=True)


def treatment_frame(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(config.get("treatments", []))
    if frame.empty:
        return pd.DataFrame(columns=["name", "label", "first_treat_col", "first_indicator_col"])
    return frame.reset_index(drop=True)


def classify_years(year_metrics: pd.DataFrame, config: dict[str, Any], current_year: int) -> pd.DataFrame:
    inspection = config["inspection_defaults"]
    frame = year_metrics.copy()
    frame["year"] = frame["year"].map(year_from_any)
    frame = frame.dropna(subset=["year"]).copy()
    frame["year"] = frame["year"].astype(int)
    frame = frame.sort_values("year").reset_index(drop=True)

    min_valid_year = inspection.get("min_valid_year", 1990)
    max_valid_year = current_year + inspection.get("max_future_year_buffer", 0)
    frame["valid_calendar_year"] = (frame["year"] >= min_valid_year) & (frame["year"] <= max_valid_year)

    for col in [
        "row_count",
        "distinct_firms",
        "position_rows",
        "posting_rows",
        "both_rows",
        "main_adoptions",
        "position_adoptions",
        "posting_adoptions",
    ]:
        if col not in frame.columns:
            frame[col] = 0
        frame[col] = frame[col].fillna(0)

    peak_rows = max(float(frame["row_count"].max()), 1.0)
    peak_firms = max(float(frame["distinct_firms"].max()), 1.0)
    peak_positions = max(float(frame["position_rows"].max()), 1.0)
    peak_postings = max(float(frame["posting_rows"].max()), 1.0)

    frame["rows_share_of_peak"] = frame["row_count"] / peak_rows
    frame["firms_share_of_peak"] = frame["distinct_firms"] / peak_firms
    frame["position_share_of_peak"] = frame["position_rows"] / peak_positions
    frame["posting_share_of_peak"] = frame["posting_rows"] / peak_postings

    frame["modern_support_main"] = (
        frame["valid_calendar_year"]
        & (frame["rows_share_of_peak"] >= inspection.get("min_rows_share_of_peak", 0.40))
        & (frame["firms_share_of_peak"] >= inspection.get("min_firms_share_of_peak", 0.40))
        & (frame["position_share_of_peak"] >= inspection.get("min_position_share_of_peak", 0.40))
    )
    frame["modern_support_posting"] = (
        frame["valid_calendar_year"]
        & (frame["posting_share_of_peak"] >= inspection.get("min_posting_share_of_peak", 0.15))
        & (frame["posting_rows"] >= inspection.get("min_posting_rows", 50_000))
    )
    frame["tiny_tail_year"] = frame["row_count"] < inspection.get("tail_year_min_rows", 1_000)
    return frame


def _rolling_median(values: pd.Series, index: int, width: int = 3) -> float | None:
    if index <= 0:
        return None
    start = max(0, index - width)
    prior = values.iloc[start:index]
    if prior.empty:
        return None
    return float(prior.median())


def _pick_end_year(frame: pd.DataFrame, config: dict[str, Any], fallback_year: int) -> int:
    inspection = config["inspection_defaults"]
    valid = frame[frame["valid_calendar_year"]].copy().sort_values("year").reset_index(drop=True)
    if valid.empty:
        return fallback_year

    end_index = len(valid) - 1
    while end_index >= 0:
        row = valid.iloc[end_index]
        if row["tiny_tail_year"]:
            end_index -= 1
            continue

        rows_median = _rolling_median(valid["row_count"], end_index)
        adoptions_median = _rolling_median(valid["main_adoptions"], end_index)
        postings_median = _rolling_median(valid["posting_rows"], end_index)
        is_anomalous = False

        if rows_median is not None and rows_median > 0:
            if row["row_count"] < rows_median * inspection.get("anomalous_end_year_drop_threshold", 0.45):
                is_anomalous = True
            if row["row_count"] > rows_median * inspection.get("anomalous_end_year_spike_threshold", 1.40):
                if adoptions_median is not None and adoptions_median > 0:
                    adoption_ratio = row["main_adoptions"] / adoptions_median
                    if adoption_ratio < inspection.get("anomalous_end_year_adoption_drop_threshold", 0.40):
                        is_anomalous = True

        if postings_median is not None and postings_median > 0 and adoptions_median is not None and adoptions_median > 0:
            posting_ratio = row["posting_rows"] / postings_median
            adoption_ratio = row["main_adoptions"] / adoptions_median
            if (
                posting_ratio < inspection.get("anomalous_end_year_drop_threshold", 0.45)
                and adoption_ratio < inspection.get("anomalous_end_year_adoption_drop_threshold", 0.40)
            ):
                is_anomalous = True

        if not is_anomalous:
            return int(row["year"])
        end_index -= 1

    return fallback_year


def recommend_windows(year_metrics: pd.DataFrame, config: dict[str, Any], current_year: int) -> dict[str, Any]:
    classified = classify_years(year_metrics, config, current_year)
    inspection = config["inspection_defaults"]

    main_candidates = classified[classified["modern_support_main"]]
    main_start = int(main_candidates["year"].min()) if not main_candidates.empty else inspection.get("default_main_start_year", 2010)

    posting_candidates = classified[classified["modern_support_posting"]]
    posting_start = int(posting_candidates["year"].min()) if not posting_candidates.empty else inspection.get("default_posting_start_year", 2017)
    position_start = main_start

    fallback_end = inspection.get("default_main_end_year", min(current_year, 2022))
    common_end = _pick_end_year(classified, config, fallback_end)
    if common_end < main_start:
        common_end = fallback_end

    windows = {
        "main": {
            "start_year": int(main_start),
            "end_year": int(common_end),
            "basis": "rows_firms_and_position_support_with_tail_year_check",
        },
        "position": {
            "start_year": int(position_start),
            "end_year": int(common_end),
            "basis": "same_as_main_because_position_coverage_is_the_backbone_of_the_panel",
        },
        "posting": {
            "start_year": int(max(posting_start, main_start)),
            "end_year": int(common_end),
            "basis": "first_year_with_meaningful_posting_coverage_and_nontrivial_adoption_counts",
        },
    }
    return {
        "classified_years": classified,
        "recommended_windows": windows,
    }


def choose_supported_event_window(
    support_frame: pd.DataFrame,
    candidate_windows: Sequence[int],
    *,
    min_event_bin_treated_firms: int,
    min_event_bin_rows: int,
) -> int:
    frame = support_frame.copy()
    if frame.empty:
        return min(candidate_windows)
    required_bins_template = lambda width: [k for k in range(-width, width + 1) if k != -1]
    for width in sorted(candidate_windows, reverse=True):
        required_bins = required_bins_template(width)
        subset = frame[frame["event_time"].isin(required_bins)].copy()
        bins_present = set(subset["event_time"].tolist())
        if set(required_bins) != bins_present:
            continue
        if (
            (subset["treated_firms"] >= min_event_bin_treated_firms).all()
            and (subset["treated_rows"] >= min_event_bin_rows).all()
        ):
            return int(width)
    return int(min(candidate_windows))


def grouped_outcomes(config: dict[str, Any], groups: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    frame = outcome_frame(config)
    if groups is not None:
        frame = frame[frame["group"].isin(groups)].copy()
    output: dict[str, list[dict[str, Any]]] = {}
    for group_name, group_frame in frame.groupby("group", sort=False):
        output[group_name] = group_frame.to_dict(orient="records")
    return output
