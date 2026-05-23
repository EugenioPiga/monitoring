#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/labs/khanna/predictive_capital/revelio_people_analytics}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/revelio_event_study_config.json}"
PARENT_YEAR_DIR="${PARENT_YEAR_DIR:-$PROJECT_ROOT/processed/final/parent_year_first_pass_paonly_safe_v3}"
PARENT_OCC_DIR="${PARENT_OCC_DIR:-$PROJECT_ROOT/processed/final/parent_occupation_year_panel_paonly_safe_v3}"
VISIBILITY_PANEL_DIR="${VISIBILITY_PANEL_DIR:-$PROJECT_ROOT/processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/processed/final/event_studies_pa_posting_parent_occ_safe_v3}"
INSPECTION_DIR="${INSPECTION_DIR:-$OUTPUT_ROOT/inspection}"
SAMPLE_DIR="${SAMPLE_DIR:-$OUTPUT_ROOT/sample}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_ROOT/results}"
FIGURES_DIR="${FIGURES_DIR:-$OUTPUT_ROOT/figures}"
TABLES_DIR="${TABLES_DIR:-$OUTPUT_ROOT/tables}"
VISIBILITY_SAMPLE_DIR="${VISIBILITY_SAMPLE_DIR:-$OUTPUT_ROOT/visibility_sample}"
VISIBILITY_RESULTS_DIR="${VISIBILITY_RESULTS_DIR:-$OUTPUT_ROOT/visibility_results}"
VISIBILITY_FIGURES_DIR="${VISIBILITY_FIGURES_DIR:-$OUTPUT_ROOT/visibility_figures}"
VISIBILITY_TABLES_DIR="${VISIBILITY_TABLES_DIR:-$OUTPUT_ROOT/visibility_tables}"
SPARK_BASE_DIR="${SPARK_BASE_DIR:-/labs/khanna/predictive_capital/scratch/revelio_event_study/${USER}}"
RUN_STACKED="${RUN_STACKED:-1}"
RUN_VISIBILITY_STACKED="${RUN_VISIBILITY_STACKED:-1}"

COMMON_EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG_PATH=${CONFIG_PATH},PARENT_YEAR_DIR=${PARENT_YEAR_DIR},PARENT_OCC_DIR=${PARENT_OCC_DIR},VISIBILITY_PANEL_DIR=${VISIBILITY_PANEL_DIR},OUTPUT_ROOT=${OUTPUT_ROOT},INSPECTION_DIR=${INSPECTION_DIR},SAMPLE_DIR=${SAMPLE_DIR},RESULTS_DIR=${RESULTS_DIR},FIGURES_DIR=${FIGURES_DIR},TABLES_DIR=${TABLES_DIR},VISIBILITY_SAMPLE_DIR=${VISIBILITY_SAMPLE_DIR},VISIBILITY_RESULTS_DIR=${VISIBILITY_RESULTS_DIR},VISIBILITY_FIGURES_DIR=${VISIBILITY_FIGURES_DIR},VISIBILITY_TABLES_DIR=${VISIBILITY_TABLES_DIR},SPARK_BASE_DIR=${SPARK_BASE_DIR},RUN_STACKED=${RUN_STACKED},RUN_VISIBILITY_STACKED=${RUN_VISIBILITY_STACKED}"

cd "$PROJECT_ROOT"

job0=$(sbatch --parsable --export="$COMMON_EXPORTS,OUTPUT_DIR=${INSPECTION_DIR}" sbatch/run_00_inspect_revelio_event_study_inputs.sbatch)
job1=$(sbatch --parsable --dependency=afterok:${job0} --export="$COMMON_EXPORTS,OUTPUT_DIR=${SAMPLE_DIR},VISIBILITY_OUTPUT_DIR=${VISIBILITY_SAMPLE_DIR}" sbatch/run_01_build_event_study_sample.sbatch)
job2=$(sbatch --parsable --dependency=afterok:${job1} --export="$COMMON_EXPORTS,OUTPUT_DIR=${OUTPUT_ROOT}" sbatch/run_02_estimate_event_studies.sbatch)
job3=$(sbatch --parsable --dependency=afterok:${job2} --export="$COMMON_EXPORTS,OUTPUT_DIR=${FIGURES_DIR}" sbatch/run_03_make_event_study_figures.sbatch)
job4=$(sbatch --parsable --dependency=afterok:${job2} --export="$COMMON_EXPORTS,OUTPUT_DIR=${TABLES_DIR}" sbatch/run_04_make_summary_tables.sbatch)
job5=$(sbatch --parsable --dependency=afterok:${job1} --export="$COMMON_EXPORTS,OUTPUT_DIR=${OUTPUT_ROOT}" sbatch/run_05_estimate_visibility_event_studies.sbatch)
job6=$(sbatch --parsable --dependency=afterok:${job5} --export="$COMMON_EXPORTS,OUTPUT_DIR=${VISIBILITY_FIGURES_DIR}" sbatch/run_06_make_visibility_event_study_figures.sbatch)
job7=$(sbatch --parsable --dependency=afterok:${job5} --export="$COMMON_EXPORTS,OUTPUT_DIR=${VISIBILITY_TABLES_DIR}" sbatch/run_07_make_visibility_summary_tables.sbatch)

printf 'Submitted jobs:\n'
printf '  00 inspect: %s\n' "$job0"
printf '  01 sample:  %s\n' "$job1"
printf '  02 estimate:%s\n' "$job2"
printf '  03 figures: %s\n' "$job3"
printf '  04 tables:  %s\n' "$job4"
printf '  05 vis est: %s\n' "$job5"
printf '  06 vis fig: %s\n' "$job6"
printf '  07 vis tab: %s\n' "$job7"
