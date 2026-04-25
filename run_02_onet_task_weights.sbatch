#!/usr/bin/env bash
#SBATCH --job-name=onet_weights
#SBATCH --output=logs/onet_weights_%j.out
#SBATCH --error=logs/onet_weights_%j.err
#SBATCH --time=0-06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/onet_weights"
SCRATCH="$SCRATCH_BASE/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"/{tmp,local,warehouse}
export TMPDIR="$SCRATCH/tmp"
export SPARK_LOCAL_DIRS="$SCRATCH/local"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export _JAVA_OPTIONS="-Djava.io.tmpdir=$TMPDIR -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+ExitOnOutOfMemoryError"
export SPARK_EVENTLOG_ENABLED="false"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
export PYSPARK_PYTHON="$(which python)"
export PYSPARK_DRIVER_PYTHON="$(which python)"
export PYSPARK_SUBMIT_ARGS="--conf spark.driver.memory=16g --conf spark.driver.maxResultSize=4g --conf spark.sql.shuffle.partitions=200 --conf spark.default.parallelism=200 --conf spark.sql.files.maxPartitionBytes=134217728 --conf spark.sql.adaptive.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true --conf spark.sql.autoBroadcastJoinThreshold=10485760 pyspark-shell"
python -V
python -u "$CODE_DIR/build_onet_task_weights.py" --onet-dir "$PROJECT_ROOT/processed/external/onet_30_2_text" --out-dir "$PROJECT_ROOT/processed/external/onet_task_weights" --shuffle-partitions 200 --coalesce 1 --tmpdir "$SCRATCH"
cat "$PROJECT_ROOT/processed/external/onet_task_weights/_metadata.json"
rm -rf "$SCRATCH" || true
echo "[INFO] Done."
