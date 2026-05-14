Safe-v3 active scripts:

1. build_revelio_firm_year_ai_hr_manager_safe_v3.py
   Full firm-year reconstruction with HR/manager variables. Original AI regex was fixed after output; use fixed add-on below for corrected AI.

2. fix_ai_columns_join_firm_year_safe_v3.py
   Corrects AI columns by building AI-only aggregates and joining them onto the existing safe-v3 firm-year panel.

3. augment_parent_year_with_ai_hr_manager_safe_v3.py
   Adds safe-v3 firm-year variables to the parent-year panel.

4. build_parent_occupation_year_safe_v3.py
   Builds parent x occupation x year panel with promotions, HR/manager outcomes, and skill outcomes.

5. build_onet_task_weights_original_safe_v3.py
   Safe copy of the original O*NET task-weight builder.

6. build_onet_visibility_indices_safe_v3.py
   Builds static internal/external O*NET visibility indices.

7. build_monitoring_exposure_parent_occ_year_safe_v3.py
   Builds parent-occupation-year exposure variables, including internal/external and log-inside formula.

8. run_monitoring_exposure_regressions_parentocc_fe_safe_v3.py
   Runs new regressions with parent x occupation FE and new outcomes.
