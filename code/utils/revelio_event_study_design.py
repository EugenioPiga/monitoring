from __future__ import annotations

import re
from typing import Any

import pandas as pd

from utils.revelio_analysis_utils import year_from_any


def outcome_frame(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(config.get("outcomes", []))
    if frame.empty:
        return pd.DataFrame(columns=["name", "group", "label", "priority", "optional"])
    if "optional" not in frame.columns:
        frame["optional"] = False
    return frame.sort_values(["priority", "name"]).reset_index(drop=True)


def required_outcomes(config: dict[str, Any]) -> list[str]:
    frame = outcome_frame(config)
    return frame.loc[~frame["optional"].fillna(False), "name"].tolist()


def optional_outcomes(config: dict[str, Any]) -> list[str]:
    frame = outcome_frame(config)
    return frame.loc[frame["optional"].fillna(False), "name"].tolist()


def event_time_values(config: dict[str, Any], *, include_omitted: bool = False) -> list[int]:
    settings = config["event_time"]
    lower = int(settings["bin_min"])
    upper = int(settings["bin_max"])
    omitted = int(settings["omit_event_time"])
    values = list(range(lower, upper + 1))
    if include_omitted:
        return values
    return [value for value in values if value != omitted]


def event_dummy_name(event_time: int, *, prefix: str = "event") -> str:
    if event_time < 0:
        return f"{prefix}_m{abs(event_time)}"
    return f"{prefix}_p{event_time}"


def visibility_settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("visibility_event_studies", {}))


def visibility_enabled(config: dict[str, Any]) -> bool:
    return bool(visibility_settings(config).get("enabled", False))


def visibility_candidate_patterns(config: dict[str, Any]) -> list[str]:
    settings = visibility_settings(config)
    patterns = settings.get("candidate_patterns")
    if isinstance(patterns, list) and patterns:
        return [str(pattern) for pattern in patterns]
    return [
        "visibility",
        "visible",
        "monitor",
        "monitoring",
        "exposure",
        "internal",
        "external",
        "task",
        "onet",
        "concentration",
        "average",
    ]


def configured_visibility_variables(config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = visibility_settings(config)
    values = settings.get("visibility_variables", [])
    parsed: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, str):
            parsed.append({"name": value, "label": value})
        elif isinstance(value, dict) and value.get("name"):
            parsed.append(
                {
                    "name": str(value["name"]),
                    "label": str(value.get("label", value["name"])),
                    "source_level": str(value.get("source_level", "occupation")),
                }
            )
    return parsed


def visibility_variable_names(config: dict[str, Any]) -> list[str]:
    return [item["name"] for item in configured_visibility_variables(config)]


def safe_visibility_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_").lower()
    return safe or "visibility"


def visibility_interaction_name(event_time: int, visibility_name: str, *, prefix: str = "event") -> str:
    return f"{event_dummy_name(event_time, prefix=prefix)}_x_{safe_visibility_name(visibility_name)}"


def build_joint_year_frame(parent_year: pd.DataFrame, parent_occ: pd.DataFrame) -> pd.DataFrame:
    py = parent_year.copy()
    po = parent_occ.copy()
    py["year"] = py["year"].map(year_from_any)
    po["year"] = po["year"].map(year_from_any)
    py = py.dropna(subset=["year"]).copy()
    po = po.dropna(subset=["year"]).copy()
    py["year"] = py["year"].astype(int)
    po["year"] = po["year"].astype(int)

    merged = py.merge(po, on="year", how="outer", suffixes=("_parent_year", "_parent_occ")).sort_values("year")
    for column in [
        "parent_year_rows",
        "parent_year_approx_parents",
        "parent_year_analysis_rows",
        "parent_year_adoptions",
        "parent_occ_rows",
        "parent_occ_approx_parents",
        "parent_occ_approx_occupations",
        "parent_occ_analysis_rows",
    ]:
        if column not in merged.columns:
            merged[column] = 0
        merged[column] = merged[column].fillna(0)
    return merged.reset_index(drop=True)


def recommend_estimation_window(year_frame: pd.DataFrame, config: dict[str, Any], *, current_year: int) -> dict[str, Any]:
    inspection = config["inspection_defaults"]
    frame = year_frame.copy()
    frame["valid_calendar_year"] = frame["year"].between(
        int(inspection.get("min_valid_year", 2000)),
        int(current_year + inspection.get("max_future_year_buffer", 0)),
    )

    peak_parent_occ_rows = max(float(frame["parent_occ_rows"].max()), 1.0)
    peak_parent_occ_parents = max(float(frame["parent_occ_approx_parents"].max()), 1.0)
    peak_parent_year_parents = max(float(frame["parent_year_approx_parents"].max()), 1.0)

    frame["parent_occ_rows_share_of_peak"] = frame["parent_occ_rows"] / peak_parent_occ_rows
    frame["parent_occ_parents_share_of_peak"] = frame["parent_occ_approx_parents"] / peak_parent_occ_parents
    frame["parent_year_parents_share_of_peak"] = frame["parent_year_approx_parents"] / peak_parent_year_parents
    frame["usable_support"] = (
        frame["valid_calendar_year"]
        & (frame["parent_occ_rows_share_of_peak"] >= float(inspection.get("min_parent_occ_rows_share_of_peak", 0.30)))
        & (frame["parent_occ_parents_share_of_peak"] >= float(inspection.get("min_parent_occ_parents_share_of_peak", 0.30)))
        & (frame["parent_year_parents_share_of_peak"] >= float(inspection.get("min_parent_year_parents_share_of_peak", 0.30)))
    )

    usable = frame.loc[frame["usable_support"]].copy()
    if usable.empty:
        start_year = int(inspection.get("default_start_year", int(frame["year"].min())))
        end_year = int(inspection.get("default_end_year", int(frame["year"].max())))
        basis = "fallback_defaults"
    else:
        start_year = int(usable["year"].min())
        end_year = int(_pick_end_year(usable, inspection))
        basis = "joint_parent_year_and_parent_occ_support"

    frame["outside_recommended_window"] = (frame["year"] < start_year) | (frame["year"] > end_year)
    frame["tiny_tail_year"] = frame["parent_occ_rows"] < float(inspection.get("tail_year_min_rows", 1000))
    return {
        "recommended_window": {
            "start_year": start_year,
            "end_year": end_year,
            "basis": basis,
        },
        "classified_years": frame,
    }


def _pick_end_year(frame: pd.DataFrame, inspection: dict[str, Any]) -> int:
    ordered = frame.sort_values("year").reset_index(drop=True)
    for index in range(len(ordered) - 1, -1, -1):
        row = ordered.iloc[index]
        if row["parent_occ_rows"] < float(inspection.get("tail_year_min_rows", 1000)):
            continue
        if index == 0:
            return int(row["year"])
        prior = ordered.iloc[max(0, index - 3):index]
        if prior.empty:
            return int(row["year"])
        prior_rows_median = float(prior["parent_occ_rows"].median())
        prior_adoption_median = float(prior["parent_year_adoptions"].median()) if "parent_year_adoptions" in prior.columns else 0.0
        if prior_rows_median > 0 and row["parent_occ_rows"] < prior_rows_median * float(inspection.get("anomalous_end_year_drop_threshold", 0.45)):
            continue
        if prior_adoption_median > 0 and row["parent_year_adoptions"] < prior_adoption_median * float(inspection.get("anomalous_end_year_adoption_drop_threshold", 0.40)):
            continue
        return int(row["year"])
    return int(ordered["year"].max())
