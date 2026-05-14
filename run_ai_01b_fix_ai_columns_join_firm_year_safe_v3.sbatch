#!/usr/bin/env bash
#SBATCH --job-name=ai_fix_firm
#SBATCH --output=logs/ai_fix_firm_year_v3_%j.out
#SBATCH --error=logs/ai_fix_firm_year_v3_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
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
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/ai_fix_firm_year_safe_v3"
SCRATCH="$SCRATCH_BASE/${SLURM_JOB_ID}"
rm -rf "$SCRATCH" 2>/dev/null || true
mkdir -p "$SCRATCH"/{tmp,local,warehouse}
export TMPDIR="$SCRATCH/tmp"
export SPARK_LOCAL_DIRS="$SCRATCH/local"

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
--conf spark.driver.memory=220g \
--conf spark.driver.maxResultSize=16g \
--conf spark.sql.shuffle.partitions=3200 \
--conf spark.default.parallelism=3200 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
pyspark-shell"

rm -rf "$PROJECT_ROOT/processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed"

python -u "$CODE_DIR/fix_ai_columns_join_firm_year_safe_v3.py" \
  --project-root "$PROJECT_ROOT" \
  --company-ref-dir "/labs/bharadwajlab/linkedin/company_ref" \
  --positions-dir "/labs/bharadwajlab/linkedin/individual_position" \
  --postings-dir "$PROJECT_ROOT/processed/intermediate/postings_extracted" \
  --base-firm-year-dir "$PROJECT_ROOT/processed/final/firm_year_panel_ai_hr_manager_safe_v3" \
  --out-dir "$PROJECT_ROOT/processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed" \
  --diagnostics-dir "$PROJECT_ROOT/processed/diagnostics/firm_year_ai_fix_safe_v3" \
  --analysis-end-year 2023 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --shuffle-partitions 3200 \
  --coalesce 800 \
  --tmpdir "$SCRATCH"

echo "[CHECK] fixed firm-year parquet files:"
find "$PROJECT_ROOT/processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed" -name "*.parquet" | head || true
find "$PROJECT_ROOT/processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed" -name "*.parquet" | wc -l || true

rm -rf "$SCRATCH" 2>/dev/null || true
echo "[INFO] Done."
