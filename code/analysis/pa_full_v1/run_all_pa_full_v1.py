#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from common import parse_common_args, resolve_paths, write_manifest, git_hash

SCRIPTS = [
"00_audit_repo_and_data.py","01_descriptive_facts.py","02_event_study_main.py","03_event_study_visibility_heterogeneity.py","04_exposure_intensity_regressions.py","05_mechanism_tests.py","06_identification_robustness.py","07_placebos_falsifications.py","08_outcome_transformations.py","09_build_paper_synthesis.py"
]

def main():
    p=argparse.ArgumentParser(); parse_common_args(p)
    p.add_argument("--skip", nargs="*", default=[])
    args=p.parse_args(); paths=resolve_paths(args)
    root=Path(__file__).resolve().parent
    ran=[]; failed=[]
    for s in SCRIPTS:
        if s in args.skip: continue
        cmd=[sys.executable, str(root/s), "--project-root", str(paths.project_root), "--input-panel", str(paths.input_panel), "--output-root", str(paths.output_root)]
        try:
            subprocess.run(cmd, check=True)
            ran.append(s)
        except subprocess.CalledProcessError:
            failed.append(s)
    write_manifest(paths.output_root/"manifest.json", {"git_hash": git_hash(paths.project_root), "input_panel": str(paths.input_panel), "output_root": str(paths.output_root), "scripts_ran": ran, "scripts_failed": failed})

if __name__=='__main__':
    main()
