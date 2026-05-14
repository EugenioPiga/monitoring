#!/bin/bash
#SBATCH --job-name=revelio_evt_03
#SBATCH --output=logs/revelio_evt_03_%j.out
#SBATCH --error=logs/revelio_evt_03_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/processed/analysis}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CONDA_ENV="${CONDA_ENV:-revelio}"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk}"
SPARK_BASE_DIR="${SPARK_BASE_DIR:-/labs/khanna/predictive_capital/scratch/revelio_event_study/${USER}}"
JOB_SCRATCH="${SPARK_BASE_DIR}/${SLURM_JOB_ID}_03_figures"
EVENT_STUDY_DIR="${EVENT_STUDY_DIR:-$ANALYSIS_ROOT/event_study}"
FIGURE_DIR="${FIGURE_DIR:-$ANALYSIS_ROOT/figures/event_study}"

mkdir -p "$LOG_DIR" "$ANALYSIS_ROOT" "$JOB_SCRATCH" "$JOB_SCRATCH/spark_local"
export JAVA_HOME
export TMPDIR="$JOB_SCRATCH"
export SPARK_LOCAL_DIRS="$JOB_SCRATCH/spark_local"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-python}"

cleanup() {
  rm -rf "$JOB_SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT

source "$HOME/miniconda/bin/activate" revelio

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
python code/analysis/03_make_event_study_figures.py \
  --project-root "$PROJECT_ROOT" \
  --config-path "$CONFIG_PATH" \
  --results-dir "$EVENT_STUDY_DIR/results" \
  --output-dir "$FIGURE_DIR"
