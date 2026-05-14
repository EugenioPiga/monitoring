#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"

import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.revelio_analysis_utils import (
    build_analysis_paths,
    call_subprocess,
    default_dataset_path,
    ensure_analysis_directories,
    ensure_directory,
    load_json,
    parse_comma_list,
    setup_logging,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Revelio event-study specifications through an R/fixest backend.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--r-script-path", default=str(PROJECT_ROOT / "code" / "analysis" / "02_estimate_event_studies_fixest.R"))
    parser.add_argument("--outcomes", default=None, help="Optional comma-separated subset of outcomes.")
    parser.add_argument("--treatments", default=None, help="Optional comma-separated subset of treatments.")
    parser.add_argument("--run-advanced", action="store_true", default=False, help="Run Sun-Abraham estimates in addition to baseline TWFE.")
    parser.add_argument("--run-heterogeneity", action="store_true", default=False, help="Run optional heterogeneity splits for primary outcomes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root)
    ensure_analysis_directories(paths)

    logger = setup_logging("02_estimate_event_studies", paths.logs_root)
    sample_path = Path(args.sample_path) if args.sample_path else paths.samples_root / "revelio_event_study_sample.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else paths.event_study_root
    ensure_directory(output_dir)

    r_script_path = Path(args.r_script_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample parquet not found: {sample_path}")
    if not r_script_path.exists():
        raise FileNotFoundError(f"R estimation script not found: {r_script_path}")
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript is not available on PATH. The estimation step requires R plus fixest dependencies.")

    selected_outcomes = parse_comma_list(args.outcomes)
    selected_treatments = parse_comma_list(args.treatments)

    manifest = {
        "project_root": str(paths.project_root),
        "sample_path": str(sample_path),
        "output_dir": str(output_dir),
        "config_path": str(Path(args.config_path).resolve()),
        "r_script_path": str(r_script_path.resolve()),
        "selected_outcomes": selected_outcomes,
        "selected_treatments": selected_treatments,
        "run_advanced": bool(args.run_advanced),
        "run_heterogeneity": bool(args.run_heterogeneity),
        "dataset_reference": str(default_dataset_path(args.project_root, config)),
    }
    write_json(manifest, output_dir / "run_manifest.json")

    command = [
        "Rscript",
        str(r_script_path),
        "--sample-path",
        str(sample_path),
        "--output-dir",
        str(output_dir),
        "--config-path",
        str(Path(args.config_path).resolve()),
        "--run-advanced",
        "1" if args.run_advanced else "0",
        "--run-heterogeneity",
        "1" if args.run_heterogeneity else "0",
    ]
    if selected_outcomes:
        command.extend(["--outcomes", ",".join(selected_outcomes)])
    if selected_treatments:
        command.extend(["--treatments", ",".join(selected_treatments)])

    logger.info("Delegating estimation to %s", r_script_path)
    call_subprocess(command, logger)
    logger.info("Estimation complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
