#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/labs/khanna/predictive_capital/revelio_people_analytics"
cd "$PROJECT_ROOT"

source "$HOME/miniconda/bin/activate" revelio
export JAVA_HOME="$HOME/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/code:${PYTHONPATH:-}"

python - <<'PY'
from pyspark.sql import SparkSession, functions as F
PROJECT='/labs/khanna/predictive_capital/revelio_people_analytics'
spark=SparkSession.builder.appName('validate_ai_visibility_safe_v3').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

def check_path(path):
    import os
    if not os.path.exists(path):
        print(f'[MISSING] {path}')
        return False
    print(f'[OK] {path}')
    return True

def show_cols(df, cols, n=10):
    existing=[c for c in cols if c in df.columns]
    missing=[c for c in cols if c not in df.columns]
    if missing:
        print('[WARN] missing columns:', missing)
    if existing:
        df.select(*existing).show(n, truncate=False)

paths={
 'firm_year': f'{PROJECT}/processed/final/firm_year_panel_ai_hr_manager_safe_v3',
 'parent_year': f'{PROJECT}/processed/final/parent_year_first_pass_ai_hr_manager_safe_v3',
 'parent_occ': f'{PROJECT}/processed/final/parent_occupation_year_panel_ai_hr_manager_safe_v3',
 'onet_weights': f'{PROJECT}/processed/external/onet_task_weights_safe_v3',
 'onet_visibility': f'{PROJECT}/processed/external/onet_task_visibility_static_safe_v3',
 'exposure': f'{PROJECT}/processed/final/monitoring_exposure_parent_occ_year_ai_visibility_safe_v3',
 'regressions': f'{PROJECT}/processed/final/regressions_monitoring_exposure_parentocc_fe_ai_visibility_safe_v3',
}
for k,p in paths.items():
    check_path(p)

fy=spark.read.parquet(paths['firm_year'])
print('\n[FIRM-YEAR] rows:', fy.count(), 'firms:', fy.select('firm_key').distinct().count())
show_cols(fy, ['firm_key','parent_rcid','year','firm_name','ai_positions_any_strict_weighted','ai_postings_any_strict','n_hr_positions','n_managers','n_employees','hr_to_employee_ratio','managers_to_employee_ratio'])
fy.groupBy('year').agg(
    F.count('*').alias('rows'),
    F.sum(F.coalesce(F.col('ai_positions_any_strict_weighted'),F.lit(0.0))).alias('ai_pos_strict'),
    F.sum(F.coalesce(F.col('ai_postings_any_strict'),F.lit(0.0))).alias('ai_posts_strict'),
    F.avg('hr_to_employee_ratio').alias('mean_hr_share'),
    F.avg('managers_to_employee_ratio').alias('mean_manager_share')
).orderBy('year').show(200)

po=spark.read.parquet(paths['parent_occ'])
print('\n[PARENT-OCC] rows:', po.count(), 'parents:', po.select('parent_rcid').distinct().count(), 'occ:', po.select('occupation').distinct().count())
show_cols(po, ['parent_rcid','occupation','year','n_workers','n_promotions','n_continuing_workers','promotion_rate','promotion_rate_continuers','hr_to_employee_ratio','managers_to_employee_ratio','ai_posting_log1p'])
po.groupBy('year').agg(F.avg('promotion_rate').alias('mean_promotion_rate'), F.sum('n_promotions').alias('n_promotions')).orderBy('year').show(50)

vis=spark.read.parquet(paths['onet_visibility'])
print('\n[VISIBILITY] rows:', vis.count())
show_cols(vis, ['onet_soc_code','task_id','task_text','task_weight','visibility_internal_static','visibility_external_static','visibility_internal_static_z','visibility_external_static_z'])

exp=spark.read.parquet(paths['exposure'])
print('\n[EXPOSURE] rows:', exp.count())
show_cols(exp, ['parent_rcid','occupation','year','monitoring_exposure_average','pa_visibility_internal_loginside','pa_visibility_external_loginside','ai_visibility_internal_loginside','ai_visibility_external_loginside'])
exp.select([F.mean(c).alias(c) for c in [
    'pa_visibility_internal_old','pa_visibility_external_old','pa_visibility_internal_loginside','pa_visibility_external_loginside','ai_visibility_internal_loginside','ai_visibility_external_loginside'
] if c in exp.columns]).show(truncate=False)

spark.stop()
PY

echo "[REGRESSION OUTPUTS]"
find "$PROJECT_ROOT/processed/final/regressions_monitoring_exposure_parentocc_fe_ai_visibility_safe_v3" -maxdepth 1 -type f -print | sort || true
