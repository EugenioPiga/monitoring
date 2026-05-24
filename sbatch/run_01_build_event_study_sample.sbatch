#!/usr/bin/env bash
#SBATCH --job-name=evt_paocc_01
#SBATCH --output=logs/evt_paocc_01_%j.out
#SBATCH --error=logs/evt_paocc_01_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --mail-user=epiga@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CODE_DIR="$PROJECT_ROOT/code"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
PARENT_YEAR_DIR="${PARENT_YEAR_DIR:-$PROJECT_ROOT/processed/final/parent_year_first_pass_paonly_safe_v3}"
PARENT_OCC_DIR="${PARENT_OCC_DIR:-$PROJECT_ROOT/processed/final/parent_occupation_year_panel_paonly_safe_v3}"
VISIBILITY_PANEL_DIR="${VISIBILITY_PANEL_DIR:-$PROJECT_ROOT/processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3}"
INSPECTION_DIR="${INSPECTION_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/inspection}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/sample}"
VISIBILITY_OUTPUT_DIR="${VISIBILITY_OUTPUT_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_sample}"
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
--conf spark.driver.memory=96g \
--conf spark.driver.maxResultSize=8g \
--conf spark.sql.shuffle.partitions=1600 \
--conf spark.default.parallelism=1600 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
pyspark-shell"

cleanup() {
  rm -rf "$JOB_SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT

python -u "$CODE_DIR/analysis/01_build_event_study_sample.py" \
  --project-root "$PROJECT_ROOT" \
  --parent-year-dir "$PARENT_YEAR_DIR" \
  --parent-occ-dir "$PARENT_OCC_DIR" \
  --visibility-panel-dir "$VISIBILITY_PANEL_DIR" \
  --inspection-dir "$INSPECTION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --visibility-output-dir "$VISIBILITY_OUTPUT_DIR" \
  --config-path "$CONFIG_PATH" \
  --shuffle-partitions 1600 \
  --coalesce 400 \
  --tmpdir "$JOB_SCRATCH"

echo "[CHECK] sample outputs:"
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort | head || true
find "$OUTPUT_DIR" -maxdepth 2 -type f | wc -l || true

echo "[INFO] Done."
