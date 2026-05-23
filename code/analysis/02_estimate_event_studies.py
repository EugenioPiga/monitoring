#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.revelio_analysis_utils import (
    build_analysis_paths,
    call_subprocess,
    ensure_analysis_directories,
    ensure_directory,
    load_json,
    parse_comma_list,
    setup_logging,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate parent-occupation safe-v3 event studies with an R/fixest backend.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--visibility-sample-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "revelio_event_study_config.json"))
    parser.add_argument("--r-script-path", default=str(PROJECT_ROOT / "code" / "analysis" / "02_estimate_event_studies_fixest.R"))
    parser.add_argument("--outcomes", default=None)
    parser.add_argument("--run-base", default="1", help="1 to run the base TWFE branch, 0 to skip.")
    parser.add_argument("--run-stacked", default="1", help="1 to run the stacked not-yet-treated estimator, 0 to skip.")
    parser.add_argument("--run-visibility", default="0", help="1 to run visibility-interacted estimators, 0 to skip.")
    parser.add_argument("--run-visibility-stacked", default="1", help="1 to run stacked visibility estimators, 0 to skip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    paths = build_analysis_paths(args.project_root, output_relative_root=config["output_relative_root"])
    ensure_analysis_directories(paths)

    logger = setup_logging("02_estimate_event_studies", paths.logs_root)
    sample_dir = Path(args.sample_dir) if args.sample_dir else paths.sample_root
    visibility_sample_dir = Path(args.visibility_sample_dir) if args.visibility_sample_dir else paths.visibility_sample_root
    output_dir = Path(args.output_dir) if args.output_dir else paths.output_root
    ensure_directory(output_dir)

    base_sample_path = sample_dir / "parent_occ_event_study_sample.parquet"
    stacked_sample_path = sample_dir / "parent_occ_event_study_stacked_sample.parquet"
    r_script_path = Path(args.r_script_path)

    if args.run_base in {"1", "true", "TRUE", "True"} and not base_sample_path.exists():
        raise FileNotFoundError(f"Base sample parquet not found: {base_sample_path}")
    if not r_script_path.exists():
        raise FileNotFoundError(f"R estimation script not found: {r_script_path}")
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript is not available on PATH. The estimation step requires R plus fixest dependencies.")

    selected_outcomes = parse_comma_list(args.outcomes)
    manifest = {
        "project_root": str(paths.project_root),
        "sample_dir": str(sample_dir),
        "visibility_sample_dir": str(visibility_sample_dir),
        "base_sample_path": str(base_sample_path),
        "stacked_sample_path": str(stacked_sample_path),
        "output_dir": str(output_dir),
        "config_path": str(Path(args.config_path).resolve()),
        "r_script_path": str(r_script_path.resolve()),
        "selected_outcomes": selected_outcomes,
        "run_base": str(args.run_base),
        "run_stacked": str(args.run_stacked),
        "run_visibility": str(args.run_visibility),
        "run_visibility_stacked": str(args.run_visibility_stacked),
    }
    write_json(manifest, output_dir / "results" / "00_run_manifest.json")

    command = [
        "Rscript",
        str(r_script_path),
        "--sample-dir",
        str(sample_dir),
        "--visibility-sample-dir",
        str(visibility_sample_dir),
        "--output-dir",
        str(output_dir),
        "--config-path",
        str(Path(args.config_path).resolve()),
        "--run-base",
        str(args.run_base),
        "--run-stacked",
        str(args.run_stacked),
        "--run-visibility",
        str(args.run_visibility),
        "--run-visibility-stacked",
        str(args.run_visibility_stacked),
    ]
    if selected_outcomes:
        command.extend(["--outcomes", ",".join(selected_outcomes)])

    logger.info("Delegating estimation to %s", r_script_path)
    call_subprocess(command, logger)
    logger.info("Estimation complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
