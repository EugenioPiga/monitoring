from __future__ import annotations
import json, logging, subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow.dataset as ds

@dataclass
class Paths:
    project_root: Path
    input_panel: Path
    output_root: Path

def parse_common_args(parser):
    parser.add_argument("--project-root", default="/labs/khanna/predictive_capital/revelio_people_analytics")
    parser.add_argument("--input-panel", default=None)
    parser.add_argument("--output-root", default=None)
    return parser

def resolve_paths(args) -> Paths:
    project_root = Path(args.project_root).resolve()
    input_panel = Path(args.input_panel) if args.input_panel else project_root / "processed" / "final" / "monitoring_exposure_parent_occ_year_paonly_safe_v3"
    output_root = Path(args.output_root) if args.output_root else project_root / "processed" / "final" / "pa_empirical_strategy_full_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    return Paths(project_root=project_root, input_panel=input_panel, output_root=output_root)

def setup_logger(name: str, out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(out_dir / f"{name}.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def load_panel(columns: list[str] | None = None, panel_path: Path | str = "") -> pd.DataFrame:
    dataset = ds.dataset(str(panel_path), format="parquet")
    table = dataset.to_table(columns=columns)
    return table.to_pandas()

def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))

def git_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
