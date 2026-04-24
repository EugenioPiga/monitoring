#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pyspark.sql import functions as F
from strategy_utils import create_spark, ensure_dir, save_json, write_parquet

def parse_args():
    p = argparse.ArgumentParser(description="Create small panel-preserving development samples.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id-col", default="parent_rcid")
    p.add_argument("--fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--shuffle-partitions", type=int, default=200)
    p.add_argument("--coalesce", type=int, default=1)
    p.add_argument("--tmpdir", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    spark = create_spark("make_revelio_dev_sample", args.threads, args.shuffle_partitions, args.tmpdir)

    df = spark.read.parquet(args.input_dir)
    ids = (
        df.select(args.id_col)
        .where(F.col(args.id_col).isNotNull())
        .distinct()
        .sample(False, args.fraction, seed=args.seed)
    )
    sample = df.join(ids, on=args.id_col, how="inner")
    write_parquet(sample, args.out_dir, args.coalesce)

    meta = {
        "input_dir": args.input_dir,
        "out_dir": args.out_dir,
        "id_col": args.id_col,
        "fraction": args.fraction,
        "rows": sample.count(),
        "ids": sample.select(args.id_col).distinct().count(),
    }
    save_json(meta, os.path.join(args.out_dir, "_sample_metadata.json"))
    print(meta)
    spark.stop()

if __name__ == "__main__":
    main()
