#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from common import parse_common_args, resolve_paths, setup_logger, load_panel, write_manifest

NAME = Path(__file__).stem

def main():
    p=argparse.ArgumentParser(); parse_common_args(p); args=p.parse_args()
    paths=resolve_paths(args)
    out=paths.output_root/NAME; out.mkdir(parents=True,exist_ok=True)
    logger=setup_logger(NAME, out)
    df=load_panel(panel_path=paths.input_panel)
    support=pd.DataFrame({"rows":[len(df)],"parents":[df["parent_rcid"].nunique()],"occupations":[df["occupation"].nunique()],"years":[df["year"].nunique()]})
    support.to_csv(out/"00_support_snapshot.csv",index=False)
    (out/"README.md").write_text(f"{NAME} scaffold created.\nTODO: full econometric implementation.\n")
    write_manifest(out/"manifest.json", {"module": NAME, "status": "scaffold"})
    logger.info("Wrote scaffold outputs for %s", NAME)

if __name__=='__main__':
    main()
