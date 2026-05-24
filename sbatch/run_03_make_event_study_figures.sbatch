#!/usr/bin/env bash
#SBATCH --job-name=evt_paocc_03
#SBATCH --output=logs/evt_paocc_03_%j.out
#SBATCH --error=logs/evt_paocc_03_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --mail-user=epiga@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CODE_DIR="$PROJECT_ROOT/code"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/figures}"
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

cleanup() {
  rm -rf "$JOB_SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT

python -u "$CODE_DIR/plotting/03_make_event_study_figures.py" \
  --project-root "$PROJECT_ROOT" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --config-path "$CONFIG_PATH" \
  --mode base

echo "[CHECK] figure outputs:"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort | head || true
find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l || true

echo "[INFO] Done."
