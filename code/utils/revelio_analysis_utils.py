from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    code_root: Path
    configs_root: Path
    docs_root: Path
    logs_root: Path
    processed_root: Path
    output_root: Path
    inspection_root: Path
    sample_root: Path
    results_root: Path
    figures_root: Path
    tables_root: Path
    visibility_sample_root: Path
    visibility_results_root: Path
    visibility_figures_root: Path
    visibility_tables_root: Path


def bootstrap_code_path(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    project_root = Path(project_root).resolve()
    code_root = project_root / "code"
    code_str = str(code_root)
    if code_str not in sys.path:
        sys.path.insert(0, code_str)
    return code_root


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_analysis_paths(
    project_root: str | Path,
    *,
    output_relative_root: str = "processed/final/event_studies_pa_posting_parent_occ_safe_v3",
) -> AnalysisPaths:
    root = Path(project_root).resolve()
    processed_root = root / "processed"
    output_root = root / output_relative_root
    return AnalysisPaths(
        project_root=root,
        code_root=root / "code",
        configs_root=root / "configs",
        docs_root=root / "docs",
        logs_root=root / "logs",
        processed_root=processed_root,
        output_root=output_root,
        inspection_root=output_root / "inspection",
        sample_root=output_root / "sample",
        results_root=output_root / "results",
        figures_root=output_root / "figures",
        tables_root=output_root / "tables",
        visibility_sample_root=output_root / "visibility_sample",
        visibility_results_root=output_root / "visibility_results",
        visibility_figures_root=output_root / "visibility_figures",
        visibility_tables_root=output_root / "visibility_tables",
    )


def ensure_analysis_directories(paths: AnalysisPaths) -> None:
    for path in [
        paths.code_root / "analysis",
        paths.code_root / "utils",
        paths.code_root / "plotting",
        paths.configs_root,
        paths.docs_root,
        paths.logs_root,
        paths.processed_root,
        paths.output_root,
        paths.inspection_root,
        paths.sample_root,
        paths.results_root,
        paths.figures_root,
        paths.tables_root,
        paths.visibility_sample_root,
        paths.visibility_results_root,
        paths.visibility_figures_root,
        paths.visibility_tables_root,
    ]:
        ensure_directory(path)


def setup_logging(script_name: str, log_dir: str | Path) -> logging.Logger:
    ensure_directory(log_dir)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(Path(log_dir) / f"{script_name}.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def write_json(obj: Any, path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)


def write_text(text: str, path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(text, encoding="utf-8")


def write_pandas_csv(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index=index)


def write_pandas_latex(frame: pd.DataFrame, path: str | Path, *, index: bool = False, float_format: str = "%.4f") -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(frame.to_latex(index=index, float_format=float_format), encoding="utf-8")


def create_spark(
    app_name: str,
    *,
    shuffle_partitions: int | None = None,
    tmpdir: str | None = None,
    extra_conf: dict[str, str] | None = None,
):
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    builder = builder.config("spark.sql.session.timeZone", "UTC")
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.files.ignoreCorruptFiles", "false")

    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    if tmpdir:
        builder = builder.config("spark.local.dir", tmpdir)
    if extra_conf:
        for key, value in extra_conf.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def parse_comma_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


def year_from_any(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def append_restriction(
    records: list[dict[str, Any]],
    *,
    step: str,
    before_rows: int,
    after_rows: int,
    reason: str,
    detail: str | None = None,
) -> None:
    records.append(
        {
            "step": step,
            "before_rows": int(before_rows),
            "after_rows": int(after_rows),
            "dropped_rows": int(before_rows - after_rows),
            "reason": reason,
            "detail": detail or "",
        }
    )


def restrictions_to_markdown(records: list[dict[str, Any]], title: str) -> str:
    lines = [f"# {title}", ""]
    if not records:
        lines.append("No restriction steps were recorded.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| Step | Before rows | After rows | Dropped rows | Reason | Detail |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {step} | {before_rows:,} | {after_rows:,} | {dropped_rows:,} | {reason} | {detail} |".format(
                **record
            )
        )
    return "\n".join(lines) + "\n"


def write_restriction_outputs(records: list[dict[str, Any]], csv_path: str | Path, markdown_path: str | Path, title: str) -> None:
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["step", "before_rows", "after_rows", "dropped_rows", "reason", "detail"])
    write_pandas_csv(frame, csv_path, index=False)
    write_text(restrictions_to_markdown(records, title), markdown_path)


def call_subprocess(command: list[str], logger: logging.Logger, env: dict[str, str] | None = None) -> None:
    logger.info("Launching command: %s", " ".join(command))
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if completed.stdout:
        for line in completed.stdout.splitlines():
            logger.info("[subprocess] %s", line)
    if completed.stderr:
        for line in completed.stderr.splitlines():
            logger.warning("[subprocess] %s", line)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def default_parent_year_path(project_root: str | Path, config: dict[str, Any]) -> Path:
    relative_path = config.get("parent_year_relative_path", "processed/final/parent_year_first_pass_paonly_safe_v3")
    return Path(project_root).resolve() / relative_path


def default_parent_occ_path(project_root: str | Path, config: dict[str, Any]) -> Path:
    relative_path = config.get("parent_occ_relative_path", "processed/final/parent_occupation_year_panel_paonly_safe_v3")
    return Path(project_root).resolve() / relative_path


def default_visibility_panel_path(project_root: str | Path, config: dict[str, Any]) -> Path:
    settings = config.get("visibility_event_studies", {})
    relative_path = settings.get(
        "visibility_panel_relative_path",
        "processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3",
    )
    return Path(project_root).resolve() / relative_path


def slurm_threads(default: int) -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
