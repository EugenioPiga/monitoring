#!/usr/bin/env bash
#SBATCH --job-name=evt_paocc_04
#SBATCH --output=logs/evt_paocc_04_%j.out
#SBATCH --error=logs/evt_paocc_04_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CODE_DIR="$PROJECT_ROOT/code"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
SAMPLE_DIR="${SAMPLE_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/sample}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/tables}"
VISIBILITY_SAMPLE_DIR="${VISIBILITY_SAMPLE_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_sample}"
VISIBILITY_RESULTS_DIR="${VISIBILITY_RESULTS_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_results}"
VISIBILITY_OUTPUT_DIR="${VISIBILITY_OUTPUT_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_tables}"
SPARK_BASE_DIR="${SPARK_BASE_DIR:-/labs/khanna/predictive_capital/scratch/revelio_event_study/${USER}}"
JOB_SCRATCH="$SPARK_BASE_DIR/${SLURM_JOB_ID}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

set +u
source "$HOME/miniconda/bin/activate" revelio
set -u
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

rm -rf "$JOB_SCRATCH" 2>/dev/null || true
mkdir -p "$JOB_SCRATCH"/{tmp,local,warehouse}
export TMPDIR="$JOB_SCRATCH/tmp"
export SPARK_LOCAL_DIRS="$JOB_SCRATCH/local"

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
--conf spark.driver.memory=20g \
--conf spark.driver.maxResultSize=4g \
--conf spark.sql.shuffle.partitions=400 \
--conf spark.default.parallelism=400 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
pyspark-shell"

cleanup() {
  rm -rf "$JOB_SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT

python -u "$CODE_DIR/analysis/04_make_event_study_tables.py" \
  --project-root "$PROJECT_ROOT" \
  --sample-dir "$SAMPLE_DIR" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --visibility-sample-dir "$VISIBILITY_SAMPLE_DIR" \
  --visibility-results-dir "$VISIBILITY_RESULTS_DIR" \
  --visibility-output-dir "$VISIBILITY_OUTPUT_DIR" \
  --config-path "$CONFIG_PATH" \
  --tmpdir "$JOB_SCRATCH" \
  --mode base

echo "[CHECK] table outputs:"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort | head || true
find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l || true

echo "[INFO] Done."
