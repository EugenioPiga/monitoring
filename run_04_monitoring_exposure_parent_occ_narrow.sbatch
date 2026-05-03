#!/usr/bin/env bash
#SBATCH --job-name=monitor_exp_narrow
#SBATCH --output=logs/monitor_exp_narrow_%j.out
#SBATCH --error=logs/monitor_exp_narrow_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --mail-user=epiga@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_ROOT="/labs/khanna/predictive_capital/revelio_people_analytics"
CODE_DIR="$PROJECT_ROOT/code"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

source "$HOME/miniconda/bin/activate" revelio
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "[INFO] CONDA_PREFIX=$CONDA_PREFIX"
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/monitor_exposure_narrow"
SCRATCH="$SCRATCH_BASE/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"/{tmp,local,warehouse}

export TMPDIR="$SCRATCH/tmp"
export SPARK_LOCAL_DIRS="$SCRATCH/local"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export _JAVA_OPTIONS="-Djava.io.tmpdir=$TMPDIR -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+ExitOnOutOfMemoryError"
export SPARK_EVENTLOG_ENABLED="false"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
export PYSPARK_PYTHON="$(which python)"
export PYSPARK_DRIVER_PYTHON="$(which python)"

export PYSPARK_SUBMIT_ARGS="\
--conf spark.driver.memory=180g \
--conf spark.driver.maxResultSize=16g \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.default.parallelism=1000 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=true \
--conf spark.sql.autoBroadcastJoinThreshold=10485760 \
pyspark-shell"

NARROW_APPS="$PROJECT_ROOT/processed/final/monitoring_applications_parent_year_narrow"
OUT_DIR="$PROJECT_ROOT/processed/final/monitoring_exposure_parent_occ_year_narrow"
DIAG_DIR="$PROJECT_ROOT/processed/diagnostics/monitoring_exposure_parent_occ_year_narrow_${SLURM_JOB_ID}"

echo "[INFO] Creating narrow monitoring applications dataset: $NARROW_APPS"

rm -rf "$NARROW_APPS" || true

python - <<'PY'
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("make_narrow_monitoring_applications").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

src = "/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_applications_parent_year"
out = "/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_applications_parent_year_narrow"

drop_categories = [
    "compliance_quality_monitoring",
    "performance_review_management",
]

df = spark.read.parquet(src)
narrow = df.where(~F.col("application_category").isin(drop_categories))

narrow.write.mode("overwrite").option("compression", "snappy").parquet(out)

print("[INFO] Narrow rows:", narrow.count())
print("[INFO] Narrow parents:", narrow.select("parent_rcid").distinct().count())

narrow.groupBy("application_category") \
    .agg(
        F.countDistinct("parent_rcid").alias("n_parents"),
        F.sum("application_posting_count").alias("n_postings")
    ) \
    .orderBy(F.desc("n_postings")) \
    .show(50, False)

spark.stop()
PY

echo "[INFO] Building narrow O*NET monitoring exposure panel"

rm -rf "$OUT_DIR" || true
mkdir -p "$DIAG_DIR"

python -u "$CODE_DIR/build_monitoring_exposure_parent_occ_year.py" \
  --parent-occ-dir "$PROJECT_ROOT/processed/final/parent_occupation_year_panel" \
  --applications-dir "$NARROW_APPS" \
  --onet-task-weights-dir "$PROJECT_ROOT/processed/external/onet_task_weights" \
  --onet-dir "$PROJECT_ROOT/processed/external/onet_30_2_text" \
  --out-dir "$OUT_DIR" \
  --diagnostics-dir "$DIAG_DIR" \
  --start-year 2014 \
  --end-year 2022 \
  --similarity-threshold-quantile 0.95 \
  --min-similarity 0.05 \
  --shuffle-partitions 1000 \
  --coalesce 250 \
  --tmpdir "$SCRATCH"

echo "[INFO] Narrow exposure metadata:"
cat "$DIAG_DIR/00_metadata.json"

echo "[INFO] Narrow exposure yearly summary:"
cat "$DIAG_DIR"/03_yearly_summary_csv/part-*.csv | column -s, -t | sed -n '1,80p'

echo "[INFO] Narrow crosswalk preview:"
head -20 "$DIAG_DIR/01_revelio_occupation_to_onet_crosswalk.csv"

rm -rf "$SCRATCH" || true
echo "[INFO] Done."
