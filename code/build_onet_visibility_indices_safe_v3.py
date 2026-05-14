#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

INTERNAL_VISIBILITY_PATTERNS = [
    "monitor", "monitoring", "inspect", "inspection", "evaluate", "evaluation", "assess", "assessment",
    "compliance", "standards", "standardized", "document", "documentation", "record", "records", "recording",
    "report", "reports", "audit", "auditing", "quality", "control", "process", "procedures", "data", "information",
    "analyzing", "analysis", "tracking", "measure", "metrics", "supervis", "coordinate", "schedule", "administer",
    "computer", "software", "database", "system", "workflow", "operation", "operations",
]

EXTERNAL_VISIBILITY_PATTERNS = [
    "customer", "client", "public", "outside", "external", "communicat", "present", "presentation", "sell", "sales",
    "negotiate", "negotiation", "resolve conflicts", "interview", "advise", "consult", "consultation", "teach", "training",
    "coach", "mentor", "service", "services", "patient", "students", "stakeholder", "vendor", "supplier", "partner",
    "relationship", "interpersonal", "perform", "demonstrate", "write", "publish", "publication", "legal", "contract",
    "certify", "credential", "license", "publicly", "market", "deliver", "support", "care", "assist",
]

NEGATIVE_EXTERNAL_PATTERNS = [
    "confidential", "internal", "back office", "database", "records", "filing", "inventory", "warehouse"
]


def parse_args():
    p = argparse.ArgumentParser(description="Build static O*NET task-level internal and outside-market visibility indices.")
    p.add_argument("--onet-task-weights-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_task_weights")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_task_visibility_static")
    p.add_argument("--diagnostics-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/diagnostics/onet_task_visibility_static")
    p.add_argument("--shuffle-partitions", type=int, default=200)
    p.add_argument("--coalesce", type=int, default=1)
    return p.parse_args()


def norm(s: str | None) -> str:
    s = "" if s is None else str(s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def score_text(text: str, positive_patterns: list[str], negative_patterns: list[str] | None = None) -> float:
    text = norm(text)
    if not text:
        return 0.0
    score = 0.0
    for pat in positive_patterns:
        if re.search(r"\b" + re.escape(pat.lower()).replace(r"\ ", r"\s+") , text):
            score += 1.0
        elif pat.lower() in text:
            score += 0.6
    if negative_patterns:
        for pat in negative_patterns:
            if pat.lower() in text:
                score -= 0.35
    return max(score, 0.0)


def minmax(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    sd = float(x.std(ddof=0))
    if sd <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - float(x.mean())) / sd


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.diagnostics_dir).mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("build_onet_visibility_indices")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    onet = spark.read.parquet(args.onet_task_weights_dir).toPandas()
    required = ["onet_soc_code", "onet_title", "task_id", "task_text", "task_weight"]
    missing = [c for c in required if c not in onet.columns]
    if missing:
        raise ValueError(f"Missing required columns in O*NET task weights: {missing}")

    task = onet[["task_id", "task_text"]].drop_duplicates("task_id").copy()
    task["visibility_internal_raw"] = task["task_text"].map(lambda x: score_text(x, INTERNAL_VISIBILITY_PATTERNS))
    task["visibility_external_raw"] = task["task_text"].map(lambda x: score_text(x, EXTERNAL_VISIBILITY_PATTERNS, NEGATIVE_EXTERNAL_PATTERNS))
    task["visibility_internal_static"] = minmax(task["visibility_internal_raw"])
    task["visibility_external_static"] = minmax(task["visibility_external_raw"])
    task["visibility_internal_static_z"] = zscore(task["visibility_internal_static"])
    task["visibility_external_static_z"] = zscore(task["visibility_external_static"])

    # Join back to occupation-task weights so later scripts can use j=task weights w_oj.
    out = onet.merge(task, on=["task_id", "task_text"], how="left")
    out["task_weight"] = pd.to_numeric(out["task_weight"], errors="coerce").fillna(0.0)

    occ = (
        out.groupby(["onet_soc_code", "onet_title"], dropna=False)
        .apply(lambda g: pd.Series({
            "occ_visibility_internal_static": float(np.sum(g["task_weight"] * g["visibility_internal_static"])),
            "occ_visibility_external_static": float(np.sum(g["task_weight"] * g["visibility_external_static"])),
            "occ_visibility_internal_static_z": float(np.sum(g["task_weight"] * g["visibility_internal_static_z"])),
            "occ_visibility_external_static_z": float(np.sum(g["task_weight"] * g["visibility_external_static_z"])),
            "n_tasks": int(g["task_id"].nunique()),
        }))
        .reset_index()
    )

    spark_out = spark.createDataFrame(out)
    if args.coalesce and args.coalesce > 0:
        spark_out = spark_out.coalesce(args.coalesce)
    spark_out.write.mode("overwrite").option("compression", "snappy").parquet(args.out_dir)

    occ.to_csv(os.path.join(args.diagnostics_dir, "occupation_visibility_static.csv"), index=False)
    task.sort_values("visibility_internal_static", ascending=False).head(200).to_csv(os.path.join(args.diagnostics_dir, "top_internal_visibility_tasks.csv"), index=False)
    task.sort_values("visibility_external_static", ascending=False).head(200).to_csv(os.path.join(args.diagnostics_dir, "top_external_visibility_tasks.csv"), index=False)

    meta = {
        "onet_task_weights_dir": args.onet_task_weights_dir,
        "out_dir": args.out_dir,
        "diagnostics_dir": args.diagnostics_dir,
        "rows": int(len(out)),
        "tasks": int(task["task_id"].nunique()),
        "occupations": int(out["onet_soc_code"].nunique()),
        "note": "Static task-level visibility indices from O*NET Task Statements and task importance weights. Internal=monitorable/documented/process-based; external=public/client-facing/market-observable.",
    }
    with open(os.path.join(args.diagnostics_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(meta)
    spark.stop()


if __name__ == "__main__":
    main()
