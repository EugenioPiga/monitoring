#!/usr/bin/env bash
#SBATCH --job-name=augment_parent_v3
#SBATCH --output=logs/augment_parent_v3_%j.out
#SBATCH --error=logs/augment_parent_v3_%j.err
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
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

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/augment_parent_safe_v3"
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
--conf spark.driver.memory=80g \
--conf spark.driver.maxResultSize=8g \
--conf spark.sql.shuffle.partitions=1200 \
--conf spark.default.parallelism=1200 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
pyspark-shell"

cleanup() {
  rm -rf "$SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT


OUT_DIR="$PROJECT_ROOT/processed/final/parent_year_first_pass_ai_hr_manager_safe_v3"
DIAG_DIR="$PROJECT_ROOT/processed/diagnostics/parent_year_ai_hr_manager_safe_v3"
mkdir -p "$DIAG_DIR"

python -u "$CODE_DIR/augment_parent_year_with_ai_hr_manager_safe_v3.py" \
  --parent-year-dir "$PROJECT_ROOT/processed/final/parent_year_first_pass" \
  --firm-year-ai-dir "$PROJECT_ROOT/processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed" \
  --out-dir "$OUT_DIR" \
  --diagnostics-dir "$DIAG_DIR" \
  --shuffle-partitions 1200 \
  --coalesce 300

echo "[CHECK] parent_year_first_pass_ai_hr_manager_safe_v3 parquet files:"
find "$OUT_DIR" -name "*.parquet" | head || true
find "$OUT_DIR" -name "*.parquet" | wc -l || true


echo "[INFO] Done."
