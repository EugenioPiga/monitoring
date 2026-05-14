Main original pipeline:
1. run_revelio_firm_year.sbatch
2. run_revelio_worker_year.sbatch
3. run_parent_first_pass.sbatch
4. run_parent_occupation_year.sbatch
5. run_01_download_onet.sbatch
6. run_02_onet_task_weights.sbatch
7. run_03_monitoring_applications.sbatch
8. run_04_monitoring_exposure_parent_occ.sbatch
9. run_05_monitoring_exposure_regressions.sbatch

Safe-v3 full AI/PA pipeline:
1. run_ai_01_firm_year_ai_hr_manager_safe_v3.sbatch
2. run_ai_01b_fix_ai_columns_join_firm_year_safe_v3.sbatch
3. run_ai_02_augment_parent_year_safe_v3.sbatch
4. run_ai_03_parent_occupation_year_safe_v3.sbatch
5. run_ai_04_onet_task_weights_original_safe_v3.sbatch
6. run_ai_05_onet_visibility_safe_v3.sbatch
7. run_ai_06_monitoring_exposure_parent_occ_safe_v3.sbatch
8. run_ai_07_monitoring_exposure_regressions_safe_v3.sbatch

Safe-v3 PA-only branch:
Use this branch for new FE specs, HR/manager outcomes, promotions, skill dispersion, internal/external visibility, and PA exposure formulas, excluding AI adoption.
1. run_paonly_02_augment_parent_year_safe_v3.sbatch
2. run_paonly_03_parent_occupation_year_safe_v3.sbatch
3. run_paonly_06_monitoring_exposure_parent_occ_safe_v3.sbatch
4. run_paonly_07_monitoring_exposure_regressions_safe_v3.sbatch

Important outputs:
- Original firm-year:
  processed/final/firm_year_panel

- Safe-v3 firm-year with HR/manager but broken AI:
  processed/final/firm_year_panel_ai_hr_manager_safe_v3

- Safe-v3 firm-year with corrected AI:
  processed/final/firm_year_panel_ai_hr_manager_safe_v3_ai_fixed

- PA-only parent-year:
  processed/final/parent_year_first_pass_paonly_safe_v3

- PA-only parent-occ-year:
  processed/final/parent_occupation_year_panel_paonly_safe_v3

- PA-only exposure:
  processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3

- PA-only regressions:
  processed/final/regressions_monitoring_exposure_parentocc_fe_paonly_safe_v3
