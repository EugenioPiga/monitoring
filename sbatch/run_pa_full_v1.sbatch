#!/bin/bash
#SBATCH --job-name=pa_full_v1
#SBATCH --output=logs/pa_full_v1_%j.out
#SBATCH --error=logs/pa_full_v1_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CODE_DIR="$PROJECT_ROOT/code/analysis/pa_full_v1"
source "$HOME/miniconda/bin/activate" revelio || true
python -u "$CODE_DIR/run_all_pa_full_v1.py" --project-root "$PROJECT_ROOT"
