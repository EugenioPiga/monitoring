# Empirical Revelio Event Study

This section organizes a first-pass empirical pipeline for firm-level adoption of people analytics or prediction technology in Revelio firm-year data. The starting panel is:

- `/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/firm_year_panel`

The code is written to follow the existing project style:

- standalone CLI scripts
- explicit paths
- Spark-friendly prep and diagnostics
- sbatch wrappers for cluster execution

## Treatment Definitions

The pipeline estimates three adoption definitions.

- Main treatment: `first_people_analytics_firm_year_any_enriched`
  This is the preferred firm-level adoption event because it combines position-side and posting-side signals.
- Position treatment: `first_people_analytics_position_year_any_enriched`
  This is the long-run backbone of the panel and the main robustness definition when posting support is thin.
- Posting treatment: `first_people_analytics_posting_year_any_enriched`
  This is the modern robustness definition when posting coverage is meaningfully present.

For each treatment, the sample builder creates:

- `event_time`
- `ever_treated`
- `never_treated`
- `not_yet_treated`
- `post`
- balanced-treated indicators
- treatment-specific analysis-row flags
- treatment-specific binned event times based on support

## Main Outcomes

Primary outcomes are prioritized around decisions, composition, and organizational change.

- `exit_rate`
- `hire_rate`
- `log_workforce`
- `workforce_growth`
- `avg_seniority`
- `data_analytics_role_share`
- `hr_people_role_share`
- `workers_with_hr_technology_skill_share`
- `workers_with_employee_feedback_tool_skill_share`

Secondary outcomes are still estimated but are not the economic center of the first pass.

- `avg_salary`
- `avg_start_salary`
- `avg_end_salary`
- `posting_count`
- `log_posting_count`
- `avg_posting_salary`
- `people_analytics_positions_any_enriched_share`
- `people_analytics_postings_any_enriched_share`

## Sample Restrictions

The code does not hardcode one fixed sample mechanically. It diagnoses the live panel and writes out the restriction logic before estimation.

Current seeded defaults from the provided diagnostics are:

- Main and position windows: `2010-2022`
- Posting window: `2017-2022`

Those defaults are only fallbacks. The inspection step recomputes year support from the actual panel and updates the recommended windows automatically.

The first-pass restriction logic is:

- drop rows with missing `firm_key` or `year`
- drop invalid year artifacts and tiny tail years
- trim to the common estimation horizon implied by the inspection output
- require treated cohorts to have at least `3` pre-period observations and `2` post-period observations by default
- keep late-treated firms as controls before their own treatment rather than forcing them into treated cohorts
- exclude already-treated cohorts that enter the trimmed window without enough pre-period support
- choose the event-study bin width from `{5, 4, 3, 2}` based on observed support in treated event-time cells
- keep a stronger both-sources restriction for Spec 3 only

The scripts also save:

- a row-level restriction report
- treatment-level cohort eligibility tables
- event-time support tables
- recommended windows and flagged years

## Estimation Design

Baseline estimation uses dynamic TWFE event studies with:

- firm fixed effects
- year fixed effects
- omitted event time `-1`
- firm-level clustered standard errors

Implemented baseline specifications:

- Spec 1: firm FE + year FE
- Spec 2: firm FE + year FE + NAICS2-by-year fixed effects
- Spec 3: Spec 1 on firm-years with both position and posting coverage
- Spec 4a: position-based treatment
- Spec 4b: posting-based treatment

Advanced design:

- Sun-Abraham style event study through `fixest::sunab(...)` for primary outcomes under the main treatment
- this runs when `--run-advanced` is passed and the required R packages are available

## Heterogeneity

The sample builder creates splits for:

- public vs non-public firms
- large vs small firms using baseline workforce
- data-intensive vs less data-intensive firms using pre-adoption skill intensity

Optional heterogeneity regressions can be launched in the estimation step with `--run-heterogeneity`.

## Output Layout

Expected outputs are written under:

- `processed/analysis/diagnostics/input_inspection/`
- `processed/analysis/diagnostics/event_study_sample/`
- `processed/analysis/samples/revelio_event_study_sample.parquet`
- `processed/analysis/event_study/results/`
- `processed/analysis/event_study/quicklook/`
- `processed/analysis/event_study/notes/`
- `processed/analysis/figures/event_study/`
- `processed/analysis/tables/event_study_sample/`
- `processed/analysis/tables/event_study_summary/`

## Run Order

1. Inspect the panel inputs.
2. Build the cleaned event-study sample.
3. Estimate event studies.
4. Build publication-ready figures.
5. Build summary tables.

## Direct Python Commands

Run these from `/labs/khanna/predictive_capital/revelio_people_analytics`.

```bash
python code/analysis/00_inspect_revelio_event_study_inputs.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --dataset-path /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/firm_year_panel
```

```bash
python code/analysis/01_build_event_study_sample.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --dataset-path /labs/khanna/predictive_capital/revelio_people_analytics/processed/final/firm_year_panel \
  --inspection-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/diagnostics/input_inspection \
  --output-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/samples/revelio_event_study_sample.parquet
```

```bash
python code/analysis/02_estimate_event_studies.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --sample-path /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/samples/revelio_event_study_sample.parquet \
  --output-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/event_study \
  --run-advanced
```

```bash
python code/analysis/03_make_event_study_figures.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --results-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/event_study/results \
  --output-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/figures/event_study
```

```bash
python code/analysis/04_make_summary_tables.py \
  --project-root /labs/khanna/predictive_capital/revelio_people_analytics \
  --sample-path /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/samples/revelio_event_study_sample.parquet \
  --output-dir /labs/khanna/predictive_capital/revelio_people_analytics/processed/analysis/tables/event_study_summary
```

## Sbatch Commands

```bash
sbatch sbatch/run_00_inspect_revelio_event_study_inputs.sbatch
sbatch sbatch/run_01_build_event_study_sample.sbatch
sbatch sbatch/run_02_estimate_event_studies.sbatch
sbatch sbatch/run_03_make_event_study_figures.sbatch
sbatch sbatch/run_04_make_summary_tables.sbatch
```

Submit the dependency chain with:

```bash
bash sbatch/run_full_revelio_event_study_pipeline.sh
```

## R Dependencies

The estimation backend requires:

- `fixest`
- `data.table`
- `arrow`
- `dplyr`
- `jsonlite`

If `Rscript` or those packages are missing, the Python estimation entrypoint fails loudly and records the run manifest so the missing dependency is explicit.
