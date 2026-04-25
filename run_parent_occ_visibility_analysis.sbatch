#!/usr/bin/env bash
#SBATCH --job-name=parent_occ_vis
#SBATCH --output=logs/parent_occ_vis_%j.out
#SBATCH --error=logs/parent_occ_vis_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] Using python: $(which python)"
python -V

OUT_DIR="$PROJECT_ROOT/processed/diagnostics/parent_occ_visibility_${SLURM_JOB_ID}"
PANEL_DIR="$PROJECT_ROOT/processed/final/parent_occupation_year_panel"

python -u "$CODE_DIR/parent_occ_visibility_analysis.py" \
  --panel-dir "$PANEL_DIR" \
  --out-dir "$OUT_DIR" \
  --pre-start-year 2016 \
  --pre-end-year 2018

echo "[INFO] Outputs written to $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 | sed -n '1,200p'
echo "[INFO] Done."
