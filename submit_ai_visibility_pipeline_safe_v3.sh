#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/labs/khanna/predictive_capital/revelio_people_analytics"
cd "$PROJECT_ROOT"
mkdir -p logs

# Optional flags:
#   RUN_ONET_DOWNLOAD=1 ./submit_ai_visibility_pipeline_safe_v3.sh
#   RUN_MONITORING_APPS=1 ./submit_ai_visibility_pipeline_safe_v3.sh
#   RUN_WORKER_YEAR=1 ./submit_ai_visibility_pipeline_safe_v3.sh
#   RUN_PARENT_FIRST_PASS=1 ./submit_ai_visibility_pipeline_safe_v3.sh
# By default, this script runs only the new safe_v3 jobs and uses existing base outputs.

RUN_ONET_DOWNLOAD=${RUN_ONET_DOWNLOAD:-0}
RUN_MONITORING_APPS=${RUN_MONITORING_APPS:-0}
RUN_WORKER_YEAR=${RUN_WORKER_YEAR:-0}
RUN_PARENT_FIRST_PASS=${RUN_PARENT_FIRST_PASS:-0}

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required sbatch: $f"
    exit 1
  fi
}

for f in \
  run_ai_01_firm_year_ai_hr_manager_safe_v3.sbatch \
  run_ai_02_augment_parent_year_safe_v3.sbatch \
  run_ai_03_parent_occupation_year_safe_v3.sbatch \
  run_ai_04_onet_task_weights_original_safe_v3.sbatch \
  run_ai_05_onet_visibility_safe_v3.sbatch \
  run_ai_06_monitoring_exposure_parent_occ_safe_v3.sbatch \
  run_ai_07_monitoring_exposure_regressions_safe_v3.sbatch; do
  require_file "$f"
done

if [[ "$RUN_WORKER_YEAR" == "1" ]]; then
  require_file run_revelio_worker_year.sbatch
  jid_worker=$(sbatch --parsable run_revelio_worker_year.sbatch)
else
  jid_worker="existing_or_skipped"
  if [[ ! -d processed/final/worker_year_panel ]]; then
    echo "[ERROR] Missing processed/final/worker_year_panel. Run with RUN_WORKER_YEAR=1 or run your existing worker-year sbatch first."
    exit 1
  fi
fi

if [[ "$RUN_PARENT_FIRST_PASS" == "1" ]]; then
  require_file run_parent_first_pass.sbatch
  if [[ "$jid_worker" == existing_or_skipped ]]; then
    jid_parent_base=$(sbatch --parsable run_parent_first_pass.sbatch)
  else
    jid_parent_base=$(sbatch --parsable --dependency=afterok:$jid_worker run_parent_first_pass.sbatch)
  fi
else
  jid_parent_base="existing_or_skipped"
  if [[ ! -d processed/final/parent_year_first_pass ]]; then
    echo "[ERROR] Missing processed/final/parent_year_first_pass. Run with RUN_PARENT_FIRST_PASS=1 or run your existing parent-first-pass sbatch first."
    exit 1
  fi
fi

if [[ "$RUN_ONET_DOWNLOAD" == "1" ]]; then
  require_file run_01_download_onet.sbatch
  jid_onet_download=$(sbatch --parsable run_01_download_onet.sbatch)
else
  jid_onet_download="existing_or_skipped"
  if [[ ! -d processed/external/onet_30_2_text ]]; then
    echo "[ERROR] Missing processed/external/onet_30_2_text. Run with RUN_ONET_DOWNLOAD=1 or run run_01_download_onet.sbatch first."
    exit 1
  fi
fi

if [[ "$RUN_MONITORING_APPS" == "1" ]]; then
  require_file run_03_monitoring_applications.sbatch
  jid_apps=$(sbatch --parsable run_03_monitoring_applications.sbatch)
else
  jid_apps="existing_or_skipped"
  if [[ ! -d processed/final/monitoring_applications_parent_year ]]; then
    echo "[ERROR] Missing processed/final/monitoring_applications_parent_year. Run with RUN_MONITORING_APPS=1 or run run_03_monitoring_applications.sbatch first."
    exit 1
  fi
fi

jid_firm=$(sbatch --parsable run_ai_01_firm_year_ai_hr_manager_safe_v3.sbatch)

if [[ "$jid_parent_base" == existing_or_skipped ]]; then
  jid_aug=$(sbatch --parsable --dependency=afterok:$jid_firm run_ai_02_augment_parent_year_safe_v3.sbatch)
else
  jid_aug=$(sbatch --parsable --dependency=afterok:$jid_firm:$jid_parent_base run_ai_02_augment_parent_year_safe_v3.sbatch)
fi

if [[ "$jid_worker" == existing_or_skipped ]]; then
  jid_pocc=$(sbatch --parsable --dependency=afterok:$jid_aug run_ai_03_parent_occupation_year_safe_v3.sbatch)
else
  jid_pocc=$(sbatch --parsable --dependency=afterok:$jid_aug:$jid_worker run_ai_03_parent_occupation_year_safe_v3.sbatch)
fi

if [[ "$jid_onet_download" == existing_or_skipped ]]; then
  jid_weights=$(sbatch --parsable run_ai_04_onet_task_weights_original_safe_v3.sbatch)
else
  jid_weights=$(sbatch --parsable --dependency=afterok:$jid_onet_download run_ai_04_onet_task_weights_original_safe_v3.sbatch)
fi

jid_visibility=$(sbatch --parsable --dependency=afterok:$jid_weights run_ai_05_onet_visibility_safe_v3.sbatch)

if [[ "$jid_apps" == existing_or_skipped ]]; then
  jid_exposure=$(sbatch --parsable --dependency=afterok:$jid_pocc:$jid_visibility run_ai_06_monitoring_exposure_parent_occ_safe_v3.sbatch)
else
  jid_exposure=$(sbatch --parsable --dependency=afterok:$jid_pocc:$jid_visibility:$jid_apps run_ai_06_monitoring_exposure_parent_occ_safe_v3.sbatch)
fi

jid_reg=$(sbatch --parsable --dependency=afterok:$jid_exposure run_ai_07_monitoring_exposure_regressions_safe_v3.sbatch)

cat <<MSG
Submitted safe_v3 pipeline:
  existing worker-year:                     $jid_worker
  existing parent first pass:               $jid_parent_base
  safe firm-year AI/HR/manager:             $jid_firm
  safe augmented parent-year:               $jid_aug
  safe parent-occupation-year:              $jid_pocc
  existing O*NET download:                  $jid_onet_download
  safe O*NET task weights:                  $jid_weights
  safe O*NET visibility:                    $jid_visibility
  existing monitoring applications:         $jid_apps
  safe monitoring exposure:                 $jid_exposure
  safe parent-occ FE regressions:           $jid_reg

Check queue:
  squeue -u $USER

Tail recent logs:
  ls -ltr logs/*v3* | tail
MSG
