#!/usr/bin/env bash
#SBATCH --job-name=monitor_apps
#SBATCH --output=logs/monitor_apps_%j.out
#SBATCH --error=logs/monitor_apps_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
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
export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/monitor_apps"
SCRATCH="$SCRATCH_BASE/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"/{tmp,local,warehouse}
export TMPDIR="$SCRATCH/tmp"
export SPARK_LOCAL_DIRS="$SCRATCH/local"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export _JAVA_OPTIONS="-Djava.io.tmpdir=$TMPDIR -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+ExitOnOutOfMemoryError"
export SPARK_EVENTLOG_ENABLED="false"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
export PYSPARK_PYTHON="$(which python)"
export PYSPARK_DRIVER_PYTHON="$(which python)"
export PYSPARK_SUBMIT_ARGS="--conf spark.driver.memory=100g --conf spark.driver.maxResultSize=8g --conf spark.sql.shuffle.partitions=800 --conf spark.default.parallelism=800 --conf spark.sql.files.maxPartitionBytes=134217728 --conf spark.sql.adaptive.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true --conf spark.sql.autoBroadcastJoinThreshold=10485760 pyspark-shell"
python -V
OUT_DIR="$PROJECT_ROOT/processed/final/monitoring_applications_parent_year"
DIAG_DIR="$PROJECT_ROOT/processed/diagnostics/monitoring_applications_parent_year_${SLURM_JOB_ID}"
rm -rf "$OUT_DIR" || true
python -u "$CODE_DIR/build_monitoring_applications_from_postings.py" --postings-dir "$PROJECT_ROOT/processed/intermediate/postings_extracted" --parent-year-dir "$PROJECT_ROOT/processed/final/parent_year_first_pass" --out-dir "$OUT_DIR" --diagnostics-dir "$DIAG_DIR" --start-year 2014 --end-year 2023 --shuffle-partitions 800 --coalesce 120 --tmpdir "$SCRATCH"
cat "$DIAG_DIR/00_metadata.json"
rm -rf "$SCRATCH" || true
echo "[INFO] Done."
