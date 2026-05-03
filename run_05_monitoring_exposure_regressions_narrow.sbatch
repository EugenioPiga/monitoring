#!/usr/bin/env bash
#SBATCH --job-name=monitor_regs_narrow
#SBATCH --output=logs/monitor_regs_narrow_%j.out
#SBATCH --error=logs/monitor_regs_narrow_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
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

echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/monitor_regs_narrow"
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
--conf spark.driver.memory=100g \
--conf spark.driver.maxResultSize=16g \
--conf spark.sql.shuffle.partitions=800 \
--conf spark.default.parallelism=800 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=true \
--conf spark.sql.autoBroadcastJoinThreshold=10485760 \
pyspark-shell"

OUT_DIR="$PROJECT_ROOT/processed/diagnostics/monitoring_exposure_regressions_narrow_${SLURM_JOB_ID}"

python -u "$CODE_DIR/run_monitoring_exposure_regressions.py" \
  --panel-dir "$PROJECT_ROOT/processed/final/monitoring_exposure_parent_occ_year_narrow" \
  --out-dir "$OUT_DIR" \
  --max-iter 30 \
  --tol 1e-8

echo "[INFO] Outputs written to $OUT_DIR"
cat "$OUT_DIR/00_metadata.json"

echo "[INFO] Main results:"
cat "$OUT_DIR"/10_monitoring_exposure_table5_results.csv | column -s, -t | sed -n '1,120p'

echo "[INFO] Diagnostics:"
cat "$OUT_DIR"/11_monitoring_exposure_diagnostics.csv | column -s, -t | sed -n '1,80p'

find "$OUT_DIR" -maxdepth 1 -name "*.png" | sort

rm -rf "$SCRATCH" || true
echo "[INFO] Done."
