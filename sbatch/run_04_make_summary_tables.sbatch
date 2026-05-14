#!/bin/bash
#SBATCH --job-name=revelio_evt_04
#SBATCH --output=logs/revelio_evt_04_%j.out
#SBATCH --error=logs/revelio_evt_04_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/processed/analysis}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CONDA_ENV="${CONDA_ENV:-revelio}"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk}"
SPARK_BASE_DIR="${SPARK_BASE_DIR:-/scratch/${USER}/revelio_event_study}"
JOB_SCRATCH="${SPARK_BASE_DIR}/${SLURM_JOB_ID}_04_tables"
SAMPLE_PATH="${SAMPLE_PATH:-$ANALYSIS_ROOT/samples/revelio_event_study_sample.parquet}"
TABLE_DIR="${TABLE_DIR:-$ANALYSIS_ROOT/tables/event_study_summary}"

mkdir -p "$LOG_DIR" "$ANALYSIS_ROOT" "$JOB_SCRATCH" "$JOB_SCRATCH/spark_local"
export JAVA_HOME
export TMPDIR="$JOB_SCRATCH"
export SPARK_LOCAL_DIRS="$JOB_SCRATCH/spark_local"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-python}"

cleanup() {
  rm -rf "$JOB_SCRATCH"
}
trap cleanup EXIT

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Unable to locate conda.sh" >&2
  exit 1
fi
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
python code/analysis/04_make_summary_tables.py \
  --project-root "$PROJECT_ROOT" \
  --config-path "$CONFIG_PATH" \
  --sample-path "$SAMPLE_PATH" \
  --output-dir "$TABLE_DIR"
