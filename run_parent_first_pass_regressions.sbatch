#!/usr/bin/env bash
#SBATCH --job-name=parent_reg
#SBATCH --output=logs/parent_reg_%j.out
#SBATCH --error=logs/parent_reg_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
echo "[INFO] CONDA_PREFIX=$CONDA_PREFIX"
echo "[INFO] Using python: $(which python)"
python -V

OUT_DIR="$PROJECT_ROOT/processed/diagnostics/parent_regressions_${SLURM_JOB_ID}"
PANEL_DIR="$PROJECT_ROOT/processed/final/parent_year_first_pass"

python -u "$CODE_DIR/parent_first_pass_regressions.py" \
  --panel-dir "$PANEL_DIR" \
  --out-dir "$OUT_DIR" \
  --min-years 0 \
  --min-avg-workforce 0 \
  --with-fe

echo "[INFO] Outputs written to: $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 | sed -n '1,200p'
echo "[INFO] Done."
