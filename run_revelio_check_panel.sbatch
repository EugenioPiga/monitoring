#!/usr/bin/env bash
#SBATCH --job-name=firm_panel_audit
#SBATCH --output=logs/firm_panel_audit_%j.out
#SBATCH --error=logs/firm_panel_audit_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --partition=general
#SBATCH --mail-user=epiga@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --signal=B:USR1@600

set -euo pipefail

PROJECT_ROOT="/labs/khanna/predictive_capital/revelio_people_analytics"
CODE_DIR="$PROJECT_ROOT/code"
LOG_DIR="$PROJECT_ROOT/logs"
OUT_DIR="$PROJECT_ROOT/processed/diagnostics/firm_year_audit_${SLURM_JOB_ID}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$PROJECT_ROOT"

source "$HOME/miniconda/bin/activate" revelio
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/firm_panel_audit"
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
--conf spark.driver.memory=140g \
--conf spark.driver.maxResultSize=16g \
--conf spark.sql.shuffle.partitions=1600 \
--conf spark.default.parallelism=1600 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
--conf spark.rpc.askTimeout=600s \
--conf spark.network.timeout=800s \
--conf spark.executor.heartbeatInterval=60s \
--conf spark.sql.broadcastTimeout=1200 \
pyspark-shell"

DATASET_PATH="$PROJECT_ROOT/processed/final/firm_year_panel"

python -u "$CODE_DIR/check_panel_audit.py" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUT_DIR" \
  --key-cols firm_key year \
  --year-col year \
  --top-n 100 \
  --shuffle-partitions 1600

echo "[INFO] Audit outputs written to: $OUT_DIR"
find "$OUT_DIR" -maxdepth 2 | sed -n '1,80p'

rm -rf "$SCRATCH" || true
echo "[INFO] Done."
