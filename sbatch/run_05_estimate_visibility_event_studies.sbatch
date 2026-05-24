#!/usr/bin/env bash
#SBATCH --job-name=evt_paocc_05
#SBATCH --output=logs/evt_paocc_05_%j.out
#SBATCH --error=logs/evt_paocc_05_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --mail-user=epiga@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CODE_DIR="$PROJECT_ROOT/code"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
SAMPLE_DIR="${SAMPLE_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/sample}"
VISIBILITY_SAMPLE_DIR="${VISIBILITY_SAMPLE_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_sample}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3}"
RUN_VISIBILITY_STACKED="${RUN_VISIBILITY_STACKED:-1}"
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

cleanup() {
  rm -rf "$JOB_SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT

python -u "$CODE_DIR/analysis/02_estimate_event_studies.py" \
  --project-root "$PROJECT_ROOT" \
  --sample-dir "$SAMPLE_DIR" \
  --visibility-sample-dir "$VISIBILITY_SAMPLE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --config-path "$CONFIG_PATH" \
  --run-base 0 \
  --run-stacked 0 \
  --run-visibility 1 \
  --run-visibility-stacked "$RUN_VISIBILITY_STACKED"

echo "[CHECK] visibility estimation outputs:"
find "$OUTPUT_DIR/visibility_results" -maxdepth 1 -type f | sort | head || true
find "$OUTPUT_DIR/visibility_results" -maxdepth 1 -type f | wc -l || true

echo "[INFO] Done."
