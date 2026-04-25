#!/usr/bin/env bash
#SBATCH --job-name=parent_occ_rob
#SBATCH --output=logs/parent_occ_rob_%j.out
#SBATCH --error=logs/parent_occ_rob_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
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

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/parent_occ_rob"
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
--conf spark.driver.memory=110g \
--conf spark.driver.maxResultSize=16g \
--conf spark.sql.shuffle.partitions=800 \
--conf spark.default.parallelism=800 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=true \
--conf spark.sql.autoBroadcastJoinThreshold=10485760 \
pyspark-shell"

OUT_DIR="$PROJECT_ROOT/processed/diagnostics/parent_occ_visibility_robustness_${SLURM_JOB_ID}"
PANEL_DIR="$PROJECT_ROOT/processed/final/parent_occupation_year_panel"

python -u "$CODE_DIR/parent_occ_visibility_robustness.py" \
  --panel-dir "$PANEL_DIR" \
  --out-dir "$OUT_DIR" \
  --pre-start-year 2016 \
  --pre-end-year 2018 \
  --event-window 3 \
  --max-iter 30 \
  --tol 1e-8

echo "[INFO] Outputs written to $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 | sed -n '1,200p'

rm -rf "$SCRATCH" || true
echo "[INFO] Done."
