# Parent-Occupation Event Study: Safe-v3

This event-study pipeline is built for the safe-v3 Revelio branch that uses:

- parent-year adoption timing from `processed/final/parent_year_first_pass_paonly_safe_v3`
- parent x occupation x year outcomes from `processed/final/parent_occupation_year_panel_paonly_safe_v3`
- visibility inputs from `processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3` when the mechanism branch is enabled

It deliberately does **not** use the older `processed/final/firm_year_panel` event-study setup.

## Treatment

The treatment is parent-level adoption of people-analytics postings:

- `T_p = first_people_analytics_posting_year_any_enriched`

Each parent x occupation cell inherits the same parent-level treatment year.

## Outcomes

Baseline outcomes:

- `exit_rate`
- `hire_rate`
- `promotion_rate`
- `promotion_rate_continuers`
- `skill_count_sd`
- `skill_hhi_mean`
- `specialist_share`
- `skill_bundle_dispersion`
- `managers_to_employee_ratio`

Optional five-year outcomes are included only if they exist and are nonmissing:

- `d5_exit_rate`
- `d5_hire_rate`
- `d5_skill_count_sd`
- `d5_skill_bundle_dispersion`
- `d5_skill_hhi_mean`
- `d5_specialist_share`

Excluded:

- `hr_to_employee_ratio`

That variable is currently excluded because the safe-v3 HR numerator is all zero.

## Baseline Design

The baseline event study is estimated on parent x occupation x year observations:

`y_{p,o,t} = sum_{k != -1} beta_k 1[t - T_p = k] + alpha_{p,o} + gamma_{o,t} + epsilon_{p,o,t}`

where:

- `alpha_{p,o}` is a parent x occupation fixed effect
- `gamma_{o,t}` is an occupation x calendar-year fixed effect
- the omitted event time is `k = -1`
- standard errors are clustered by `parent_rcid`

Why not parent x year fixed effects?

- Treatment timing varies at the parent-year level.
- Parent x year fixed effects would absorb the treatment event-time variation directly.

Why parent x occupation FE and occupation x year FE?

- Parent x occupation FE absorb persistent differences in the workforce structure of each parent-occupation cell.
- Occupation x year FE absorb broad labor-market shocks that hit occupations nationally over time.

## Event-Time Construction

Raw event time:

- `event_time_raw = year - first_people_analytics_posting_year_any_enriched`

Binning:

- `event_time_raw <= -6` becomes `-6`
- `event_time_raw >= 6` becomes `6`
- retain `-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5`
- omit `-1` in estimation

The estimated coefficients are therefore reported for:

- `-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6`

## Staggered Adoption

Two estimators are produced.

1. TWFE first pass

- parent x occupation FE
- occupation x year FE
- clustered by parent

This is useful for transparent first-pass plots, but it can be contaminated under staggered adoption.

2. Stacked not-yet-treated design

- cohorts are defined by the parent-level first adoption year
- for cohort `g`, treated parents satisfy `T_p = g`
- controls are never-treated parents or parents with `T_p > g + 6`
- the stacked sample keeps years `g - 6` through `g + 6`
- fixed effects are:
  - `cohort x parent x occupation`
  - `cohort x occupation x year`
- standard errors are still clustered by `parent_rcid`

This stacked estimator is the cleaner design in the presence of staggered adoption.

## Visibility-Interacted Event-Study Design

The baseline event study above estimates the **average** effect of parent-level PA adoption on parent x occupation x year outcomes.

The visibility-interacted branch estimates whether those effects are larger in occupations that are more visible or monitorable within the same parent-year.

The preferred mechanism specification is:

`y_{p,o,t} = sum_{k != -1} theta_k [1[t - T_p = k] x Visibility_o] + alpha_{p,o} + delta_{p,t} + gamma_{o,t} + epsilon_{p,o,t}`

where:

- `alpha_{p,o}` is a parent x occupation fixed effect
- `delta_{p,t}` is a parent x year fixed effect
- `gamma_{o,t}` is an occupation x year fixed effect
- the omitted event time is `k = -1`
- standard errors are clustered by `parent_rcid`

Why does this design include parent x year fixed effects?

- Parent-level PA adoption occurs at the parent-year level.
- Once parent x year fixed effects are included, all uninteracted parent-level event-time shocks are absorbed.
- Identification therefore comes only from comparing more visible and less visible occupations **within the same parent-year** around adoption.

That is the correct design for the monitoring / visibility mechanism.

The coefficient at event time `k` should be interpreted as:

- the differential effect of PA adoption for a 1 standard deviation increase in visibility,
- relative to event time `-1`,
- holding fixed the parent-year shock common to all occupations inside the parent.

Visibility variables are standardized inside the estimation sample:

- `visibility_std = (visibility - mean) / sd`

The pipeline also creates `high` / `low` visibility indicators using the median among nonmissing occupation-level observations. Those are used only for descriptive tables, not as the preferred regression design.

Default safe-v3 visibility variables:

- `occ_visibility_internal_static`
- `occ_visibility_external_static`

The inspection step searches live schemas first and writes `inspection/12_visibility_candidate_columns.csv` before the visibility branch is built.

## Output Layout

Outputs are written to:

- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/inspection`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/sample`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/results`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/figures`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/tables`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_sample`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_results`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_figures`
- `processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_tables`

## Run Order

1. Inspect the parent-year and parent-occupation-year inputs.
2. Build the joined event-study sample.
3. Estimate TWFE and stacked event studies.
4. Plot the coefficient paths.
5. Build compact diagnostics tables.
6. Estimate the visibility-interacted TWFE and stacked event studies.
7. Plot the visibility-interacted coefficients.
8. Build compact visibility summary tables.

## Direct Commands

```bash
python code/analysis/00_inspect_revelio_event_study_inputs.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --parent-year-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass_paonly_safe_v3 \
  --parent-occ-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_occupation_year_panel_paonly_safe_v3
```

```bash
python code/analysis/01_build_event_study_sample.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --parent-year-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_year_first_pass_paonly_safe_v3 \
  --parent-occ-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_occupation_year_panel_paonly_safe_v3 \
  --visibility-panel-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_exposure_parent_occ_year_paonly_safe_v3
```

```bash
python code/analysis/02_estimate_event_studies.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --run-base 1 \
  --run-stacked 1
```

```bash
python code/analysis/02_estimate_event_studies.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --visibility-sample-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/event_studies_pa_posting_parent_occ_safe_v3/visibility_sample \
  --run-base 0 \
  --run-stacked 0 \
  --run-visibility 1 \
  --run-visibility-stacked 1
```

```bash
python code/plotting/03_make_event_study_figures.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --mode both
```

```bash
python code/analysis/04_make_event_study_tables.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --mode both
```

## Sbatch Commands

Run the full base + visibility pipeline:

```bash
cd /labs/khanna/predictive_capital/revelio_people_analytics
bash sbatch/run_full_revelio_event_study_pipeline.sh
```

Run the visibility branch only after the sample already exists:

```bash
cd /labs/khanna/predictive_capital/revelio_people_analytics
sbatch sbatch/run_05_estimate_visibility_event_studies.sbatch
sbatch sbatch/run_06_make_visibility_event_study_figures.sbatch
sbatch sbatch/run_07_make_visibility_summary_tables.sbatch
```

## Validate Outputs

Start with:

- `inspection/10_recommended_window.json`
- `sample/01_adoption_cohort_counts.csv`
- `sample/02_event_time_support.csv`
- `results/02_event_study_coefficients.csv`
- `results/03_pretrend_summary.csv`
- `figures/event_study_appendix.pdf`
- `tables/01_estimator_summary.csv`
- `inspection/12_visibility_candidate_columns.csv`
- `visibility_sample/01_visibility_variable_summary.csv`
- `visibility_results/02_visibility_event_study_coefficients.csv`
- `visibility_results/03_visibility_pretrend_summary.csv`
- `visibility_figures/visibility_event_study_appendix.pdf`
- `visibility_tables/01_visibility_estimator_summary.csv`

If the stacked estimator is skipped, check:

- `results/05_model_status.csv`
- `visibility_results/04_visibility_model_status.csv`

for the exact reason, typically missing stacked parquet support or disabled execution.
