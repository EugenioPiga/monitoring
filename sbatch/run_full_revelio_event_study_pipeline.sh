#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/processed/final/firm_year_panel}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/processed/analysis}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CONDA_ENV="${CONDA_ENV:-revelio}"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk}"
SPARK_BASE_DIR="${SPARK_BASE_DIR:-/labs/khanna/predictive_capital/scratch/revelio_event_study/${USER}}"
RUN_ADVANCED="${RUN_ADVANCED:-1}"
RUN_HETEROGENEITY="${RUN_HETEROGENEITY:-0}"

COMMON_EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATASET_PATH=${DATASET_PATH},ANALYSIS_ROOT=${ANALYSIS_ROOT},LOG_DIR=${LOG_DIR},CONDA_ENV=${CONDA_ENV},JAVA_HOME=${JAVA_HOME},SPARK_BASE_DIR=${SPARK_BASE_DIR},RUN_ADVANCED=${RUN_ADVANCED},RUN_HETEROGENEITY=${RUN_HETEROGENEITY}"

cd "$PROJECT_ROOT"

job0=$(sbatch --parsable --export="$COMMON_EXPORTS" sbatch/run_00_inspect_revelio_event_study_inputs.sbatch)
job1=$(sbatch --parsable --dependency=afterok:${job0} --export="$COMMON_EXPORTS" sbatch/run_01_build_event_study_sample.sbatch)
job2=$(sbatch --parsable --dependency=afterok:${job1} --export="$COMMON_EXPORTS" sbatch/run_02_estimate_event_studies.sbatch)
job3=$(sbatch --parsable --dependency=afterok:${job2} --export="$COMMON_EXPORTS" sbatch/run_03_make_event_study_figures.sbatch)
job4=$(sbatch --parsable --dependency=afterok:${job1} --export="$COMMON_EXPORTS" sbatch/run_04_make_summary_tables.sbatch)

printf 'Submitted jobs:\n'
printf '  00 inspect: %s\n' "$job0"
printf '  01 sample:  %s\n' "$job1"
printf '  02 estimate:%s\n' "$job2"
printf '  03 figures: %s\n' "$job3"
printf '  04 tables:  %s\n' "$job4"
