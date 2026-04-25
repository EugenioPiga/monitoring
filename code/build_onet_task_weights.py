#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession


def parse_args():
    p = argparse.ArgumentParser(description="Build O*NET occupation-task importance weights.")
    p.add_argument("--onet-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_30_2_text")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_task_weights")
    p.add_argument("--shuffle-partitions", type=int, default=200)
    p.add_argument("--coalesce", type=int, default=1)
    p.add_argument("--tmpdir", default=None)
    p.add_argument("--core-only", action="store_true")
    return p.parse_args()


def find_onet_file(onet_dir: Path, target_name: str) -> Path:
    """
    Recursively find an O*NET text file, because the ZIP sometimes extracts
    into a nested db_* folder.
    """
    target_norm = target_name.lower().replace(" ", "").replace("_", "")
    candidates = []

    for p in onet_dir.rglob("*"):
        if not p.is_file():
            continue
        name_norm = p.name.lower().replace(" ", "").replace("_", "")
        if name_norm == target_norm:
            candidates.append(p)

    if candidates:
        return candidates[0]

    # Softer fallback: match all words in the filename.
    words = target_name.lower().replace(".txt", "").split()
    for p in onet_dir.rglob("*.txt"):
        lname = p.name.lower()
        if all(w in lname for w in words):
            return p

    found = "\n".join(str(p) for p in onet_dir.rglob("*.txt"))
    raise FileNotFoundError(
        f"Could not find {target_name} under {onet_dir}.\n\n"
        f"Text files found:\n{found[:5000]}"
    )


def read_onet_txt(path: Path) -> pd.DataFrame:
    print(f"[INFO] Reading {path}")
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df


def main():
    args = parse_args()

    onet_dir = Path(args.onet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_statements_path = find_onet_file(onet_dir, "Task Statements.txt")
    task_ratings_path = find_onet_file(onet_dir, "Task Ratings.txt")
    occupation_data_path = find_onet_file(onet_dir, "Occupation Data.txt")

    print("[INFO] Found O*NET files:")
    print(f"  Task Statements: {task_statements_path}")
    print(f"  Task Ratings:    {task_ratings_path}")
    print(f"  Occupation Data: {occupation_data_path}")

    tasks = read_onet_txt(task_statements_path)
    ratings = read_onet_txt(task_ratings_path)
    occ = read_onet_txt(occupation_data_path)

    tasks.columns = [c.strip() for c in tasks.columns]
    ratings.columns = [c.strip() for c in ratings.columns]
    occ.columns = [c.strip() for c in occ.columns]

    required_tasks = ["O*NET-SOC Code", "Task ID", "Task", "Task Type"]
    required_ratings = ["O*NET-SOC Code", "Task ID", "Scale ID", "Data Value", "N", "Standard Error", "Recommend Suppress"]
    required_occ = ["O*NET-SOC Code", "Title", "Description"]

    tasks = ensure_cols(tasks, required_tasks)
    ratings = ensure_cols(ratings, required_ratings)
    occ = ensure_cols(occ, required_occ)

    # Use O*NET task importance ratings: Scale ID == IM.
    ratings_im = ratings[ratings["Scale ID"].astype(str).str.upper().eq("IM")].copy()
    ratings_im["Data Value"] = pd.to_numeric(ratings_im["Data Value"], errors="coerce")

    keep_tasks = tasks.copy()
    if args.core_only and "Task Type" in keep_tasks.columns:
        keep_tasks = keep_tasks[keep_tasks["Task Type"].astype(str).str.lower().eq("core")].copy()

    merged = keep_tasks.merge(
        ratings_im[
            [
                "O*NET-SOC Code",
                "Task ID",
                "Data Value",
                "N",
                "Standard Error",
                "Recommend Suppress",
            ]
        ],
        on=["O*NET-SOC Code", "Task ID"],
        how="left",
    )

    merged["importance"] = pd.to_numeric(merged["Data Value"], errors="coerce").fillna(1.0)
    merged["importance"] = merged["importance"].clip(lower=0.0)

    denom = merged.groupby("O*NET-SOC Code")["importance"].transform("sum")
    merged["task_weight"] = merged["importance"] / denom
    merged.loc[~merged["task_weight"].replace([float("inf"), -float("inf")], pd.NA).notna(), "task_weight"] = pd.NA

    occ_small = occ.rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Title": "onet_title",
            "Description": "onet_description",
        }
    )

    out = merged.rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Task ID": "task_id",
            "Task": "task_text",
            "Task Type": "task_type",
            "Data Value": "task_importance",
        }
    )

    out = out.merge(
        occ_small[["onet_soc_code", "onet_title", "onet_description"]],
        on="onet_soc_code",
        how="left",
    )

    out = ensure_cols(
        out,
        [
            "onet_soc_code",
            "onet_title",
            "onet_description",
            "task_id",
            "task_text",
            "task_type",
            "task_importance",
            "task_weight",
            "N",
            "Standard Error",
            "Recommend Suppress",
        ],
    )

    out = out[
        [
            "onet_soc_code",
            "onet_title",
            "onet_description",
            "task_id",
            "task_text",
            "task_type",
            "task_importance",
            "task_weight",
            "N",
            "Standard Error",
            "Recommend Suppress",
        ]
    ].copy()

    out["task_id"] = out["task_id"].astype(str)
    out["task_text"] = out["task_text"].fillna("")
    out["task_type"] = out["task_type"].fillna("")
    out["task_importance"] = pd.to_numeric(out["task_importance"], errors="coerce")
    out["task_weight"] = pd.to_numeric(out["task_weight"], errors="coerce")

    spark = (
        SparkSession.builder
        .appName("build_onet_task_weights")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    sdf = spark.createDataFrame(out)
    if args.coalesce and args.coalesce > 0:
        sdf = sdf.coalesce(args.coalesce)

    sdf.write.mode("overwrite").option("compression", "snappy").parquet(str(out_dir))

    meta = {
        "onet_dir": str(onet_dir),
        "out_dir": str(out_dir),
        "task_statements_path": str(task_statements_path),
        "task_ratings_path": str(task_ratings_path),
        "occupation_data_path": str(occupation_data_path),
        "n_rows": int(len(out)),
        "n_onet_occupations": int(out["onet_soc_code"].nunique()),
        "n_tasks": int(out["task_id"].nunique()),
        "core_only": bool(args.core_only),
    }

    with open(out_dir / "_metadata.json", "w") as f:
        import json
        json.dump(meta, f, indent=2, sort_keys=True)

    print("[INFO] Done O*NET task weights")
    print(meta)

    spark.stop()


if __name__ == "__main__":
    main()
