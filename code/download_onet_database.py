#!/usr/bin/env python3
from __future__ import annotations
import argparse, urllib.request, zipfile
from pathlib import Path

ONET_TEXT_ZIP_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text.zip"

def parse_args():
    p = argparse.ArgumentParser(description="Download and extract the O*NET 30.2 text database.")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_30_2_text")
    p.add_argument("--url", default=ONET_TEXT_ZIP_URL)
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "db_30_2_text.zip"
    if not zip_path.exists():
        print(f"[INFO] Downloading {args.url}", flush=True)
        urllib.request.urlretrieve(args.url, zip_path)
    else:
        print(f"[INFO] Zip exists: {zip_path}", flush=True)
    if not (out / "Task Statements.txt").exists():
        print(f"[INFO] Extracting {zip_path}", flush=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out)
    print(f"[INFO] O*NET directory: {out}", flush=True)

if __name__ == "__main__":
    main()
