#!/usr/bin/env bash
#SBATCH --job-name=pa_expos_v3
#SBATCH --output=logs/exposure_v3_%j.out
#SBATCH --error=logs/exposure_v3_%j.err
#SBATCH --time=3-00:00:00
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
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/monitoring_exposure_safe_v3"
SCRATCH="$SCRATCH_BASE/${SLURM_JOB_ID}"
rm -rf "$SCRATCH" 2>/dev/null || true
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
--conf spark.driver.memory=190g \
--conf spark.driver.maxResultSize=12g \
--conf spark.sql.shuffle.partitions=2400 \
--conf spark.default.parallelism=2400 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
pyspark-shell"

cleanup() {
  rm -rf "$SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT


OUT_DIR="$PROJECT_ROOT/processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3"
DIAG_DIR="$PROJECT_ROOT/processed/diagnostics/monitoring_exposure_parent_occ_year_paonly_safe_v3"
mkdir -p "$DIAG_DIR"

python -u "$CODE_DIR/build_monitoring_exposure_parent_occ_year_safe_v3.py" \
  --parent-occ-dir "$PROJECT_ROOT/processed/final/parent_occupation_year_panel_paonly_safe_v3" \
  --parent-year-dir "$PROJECT_ROOT/processed/final/parent_year_first_pass_paonly_safe_v3" \
  --applications-dir "$PROJECT_ROOT/processed/final/monitoring_applications_parent_year" \
  --onet-task-weights-dir "$PROJECT_ROOT/processed/external/onet_task_weights_safe_v3" \
  --visibility-dir "$PROJECT_ROOT/processed/external/onet_task_visibility_static_safe_v3" \
  --onet-dir "$PROJECT_ROOT/processed/external/onet_30_2_text" \
  --out-dir "$OUT_DIR" \
  --diagnostics-dir "$DIAG_DIR" \
  --tmpdir "$SCRATCH" \
  --start-year 2014 \
  --end-year 2022 \
  --similarity-threshold-quantile 0.95 \
  --min-similarity 0.05 \
  --shuffle-partitions 2400 \
  --coalesce 500

echo "[CHECK] monitoring_exposure_parent_occ_year_paonly_safe_v3 parquet files:"
find "$OUT_DIR" -name "*.parquet" | head || true
find "$OUT_DIR" -name "*.parquet" | wc -l || true


echo "[INFO] Done."
