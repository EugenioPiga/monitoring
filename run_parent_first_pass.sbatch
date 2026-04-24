#!/usr/bin/env bash
#SBATCH --job-name=parent_first_pass
#SBATCH --output=logs/parent_first_pass_%j.out
#SBATCH --error=logs/parent_first_pass_%j.err
#SBATCH --time=7-00:00:00
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] CONDA_PREFIX=$CONDA_PREFIX"
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/parent_first_pass"
SCRATCH="$SCRATCH_BASE/${SLURM_JOB_ID}"
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
--conf spark.driver.maxResultSize=8g \
--conf spark.sql.shuffle.partitions=800 \
--conf spark.default.parallelism=800 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=true \
--conf spark.sql.autoBroadcastJoinThreshold=10485760 \
pyspark-shell"

OUT_DIR="$PROJECT_ROOT/processed/diagnostics/parent_first_pass_${SLURM_JOB_ID}"
PANEL_OUT="$PROJECT_ROOT/processed/final/parent_year_first_pass"

rm -rf "$PANEL_OUT" || true

python -u "$CODE_DIR/parent_first_pass.py" \
  --project-root "$PROJECT_ROOT" \
  --firm-year-dir "$PROJECT_ROOT/processed/final/firm_year_panel" \
  --worker-year-dir "$PROJECT_ROOT/processed/final/worker_year_panel" \
  --out-dir "$OUT_DIR" \
  --panel-out-dir "$PANEL_OUT" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --shuffle-partitions 800 \
  --start-year 2014 \
  --end-year 2022 \
  --min-workforce 5 \
  --coalesce 80 \
  --tmpdir "$SCRATCH"

echo "[INFO] Outputs written to: $OUT_DIR"
find "$OUT_DIR" -maxdepth 2 | sed -n '1,200p'

rm -rf "$SCRATCH" || true
echo "[INFO] Done."
