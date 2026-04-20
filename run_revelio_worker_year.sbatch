#!/usr/bin/env bash
#SBATCH --job-name=revelio_worker_year
#SBATCH --output=logs/revelio_worker_year_%j.out
#SBATCH --error=logs/revelio_worker_year_%j.err
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
echo "[INFO] Using python: $(which python)"
python -V

export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "[JAVA] Using $(java -version 2>&1 | head -n 1)"

SCRATCH_BASE="/labs/khanna/predictive_capital/scratch/revelio_worker_year"
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

# Give Spark a large heap and avoid oversized partitions/tasks
export PYSPARK_SUBMIT_ARGS="\
--conf spark.driver.memory=220g \
--conf spark.driver.maxResultSize=16g \
--conf spark.sql.shuffle.partitions=3200 \
--conf spark.default.parallelism=3200 \
--conf spark.sql.files.maxPartitionBytes=134217728 \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=false \
--conf spark.sql.autoBroadcastJoinThreshold=10485760 \
pyspark-shell"

FIRM_YEAR_DIR="$PROJECT_ROOT/processed/final/firm_year_panel"
WORKER_YEAR_OUT="$PROJECT_ROOT/processed/final/worker_year_panel"

# Make sure upstream firm output really exists
FIRM_PARQUET_COUNT=$(find "$FIRM_YEAR_DIR" -name "*.parquet" | wc -l || true)
echo "[CHECK] firm_year parquet count: $FIRM_PARQUET_COUNT"
if [ "${FIRM_PARQUET_COUNT:-0}" -eq 0 ]; then
  echo "[ERROR] No parquet files found in $FIRM_YEAR_DIR"
  exit 1
fi

rm -rf "$WORKER_YEAR_OUT"

python -u "$CODE_DIR/create_revelio_worker_year.py" \
  --project-root "$PROJECT_ROOT" \
  --positions-dir "/labs/bharadwajlab/linkedin/individual_position" \
  --education-dir "/labs/bharadwajlab/linkedin/individual_education" \
  --skills-dir "/labs/bharadwajlab/linkedin/individual_skill" \
  --users-dir "/labs/bharadwajlab/linkedin/individual_user" \
  --firm-year-dir "$FIRM_YEAR_DIR" \
  --intermediate-dir "$PROJECT_ROOT/processed/intermediate/worker_year" \
  --out-dir "$WORKER_YEAR_OUT" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --shuffle-partitions 3200 \
  --coalesce 800 \
  --tmpdir "$SCRATCH"

echo "[CHECK] worker_year_panel parquet files:"
find "$WORKER_YEAR_OUT" -name "*.parquet" | head
find "$WORKER_YEAR_OUT" -name "*.parquet" | wc -l

rm -rf "$SCRATCH" || true
echo "[INFO] Done."
