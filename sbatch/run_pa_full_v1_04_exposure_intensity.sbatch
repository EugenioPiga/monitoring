#!/bin/bash
#SBATCH --job-name=pa_full04
#SBATCH --output=logs/pa_full04_%j.out
#SBATCH --error=logs/pa_full04_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=epiga@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
INPUT_PANEL="${INPUT_PANEL:-$PROJECT_ROOT/processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/processed/final/pa_empirical_strategy_full_v1}"
CODE_DIR="$PROJECT_ROOT/code/analysis/pa_full_v1"

mkdir -p "$PROJECT_ROOT/logs"

# Conda activation can fail under nounset; relax -u temporarily
set +u
source "$HOME/miniconda/bin/activate" revelio || true
set -u

python -u "$CODE_DIR/04_exposure_intensity_regressions.py" \
  --project-root "$PROJECT_ROOT" \
  --input-panel "$INPUT_PANEL" \
  --output-root "$OUTPUT_ROOT"
