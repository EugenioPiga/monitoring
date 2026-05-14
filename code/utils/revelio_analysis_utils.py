from __future__ import annotations

import json
import logging
import os
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
    processed_analysis_root: Path
    samples_root: Path
    event_study_root: Path
    tables_root: Path
    figures_root: Path
    diagnostics_root: Path


def bootstrap_code_path(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    project_root = Path(project_root).resolve()
    code_root = project_root / "code"
    code_str = str(code_root)
    if code_str not in sys.path:
        sys.path.insert(0, code_str)
    return code_root


def build_analysis_paths(project_root: str | Path) -> AnalysisPaths:
    root = Path(project_root).resolve()
    processed_root = root / "processed"
    processed_analysis_root = processed_root / "analysis"
    return AnalysisPaths(
        project_root=root,
        code_root=root / "code",
        configs_root=root / "configs",
        docs_root=root / "docs",
        logs_root=root / "logs",
        processed_root=processed_root,
        processed_analysis_root=processed_analysis_root,
        samples_root=processed_analysis_root / "samples",
        event_study_root=processed_analysis_root / "event_study",
        tables_root=processed_analysis_root / "tables",
        figures_root=processed_analysis_root / "figures",
        diagnostics_root=processed_analysis_root / "diagnostics",
    )


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_analysis_directories(paths: AnalysisPaths) -> None:
    for path in [
        paths.code_root / "analysis",
        paths.code_root / "utils",
        paths.code_root / "plotting",
        paths.code_root / "archive",
        paths.configs_root,
        paths.docs_root,
        paths.logs_root,
        paths.processed_root,
        paths.processed_analysis_root,
        paths.samples_root,
        paths.event_study_root,
        paths.tables_root,
        paths.figures_root,
        paths.diagnostics_root,
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


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def write_text(text: str, path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(text, encoding="utf-8")


def write_pandas_csv(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index=index)


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


def extract_naics_digits(column, digits: int = 2):
    from pyspark.sql import functions as F

    cleaned = F.regexp_extract(F.coalesce(column.cast("string"), F.lit("")), r"([0-9]+)", 1)
    return F.when(F.length(cleaned) >= digits, F.substring(cleaned, 1, digits))


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
    import subprocess

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


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()


def default_dataset_path(project_root: str | Path, config: dict[str, Any]) -> Path:
    relative_path = config.get("dataset_relative_path", "processed/final/firm_year_panel")
    return Path(project_root).resolve() / relative_path


def as_bool_flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    return default


def slurm_threads(default: int) -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default
