#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import parse_common_args, resolve_paths, setup_logger, load_panel, write_manifest

MAIN_OUTCOMES = [
    "exit_rate","hire_rate","promotion_rate_continuers","skill_count_sd","skill_bundle_dispersion",
    "skill_hhi_mean","specialist_share","hr_to_employee_ratio","managers_to_employee_ratio","d5_log_workers"
]

def ensure_tex(df: pd.DataFrame, path: Path):
    try:
        path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    except Exception:
        path.write_text("% latex export failed; see csv version\n")

def pick(path: Path):
    return path if path.exists() else None

def main():
    p = argparse.ArgumentParser()
    parse_common_args(p)
    args = p.parse_args()
    paths = resolve_paths(args)

    out_root = paths.output_root
    paper = out_root / "paper_package"
    app = out_root / "appendix_package"
    paper.mkdir(parents=True, exist_ok=True)
    app.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("10_build_paper_package", paper)

    df = load_panel(panel_path=paths.input_panel)

    vars_core = [c for c in ["exit_rate","hire_rate","promotion_rate","promotion_rate_continuers","skill_count_sd","skill_bundle_dispersion","skill_hhi_mean","specialist_share","hr_to_employee_ratio","managers_to_employee_ratio"] if c in df.columns]
    vis_vars = [c for c in ["occ_visibility_internal_static","occ_visibility_external_static","monitoring_exposure_average","monitoring_exposure_concentration","pa_visibility_internal_loginside","pa_visibility_external_loginside"] if c in df.columns]
    rows = [
        {"metric":"unit_of_observation","value":"parent_rcid x occupation x year"},
        {"metric":"n_parent_rcid","value":int(df["parent_rcid"].nunique()) if "parent_rcid" in df else np.nan},
        {"metric":"n_occupations","value":int(df["occupation"].nunique()) if "occupation" in df else np.nan},
        {"metric":"n_parent_occupation_cells","value":int(df[["parent_rcid","occupation"]].drop_duplicates().shape[0]) if set(["parent_rcid","occupation"]).issubset(df.columns) else np.nan},
        {"metric":"year_range","value":f"{int(df['year'].min())}-{int(df['year'].max())}" if "year" in df else np.nan},
    ]
    if "pa_posting_log1p" in df.columns:
        adopter = (df["pa_posting_log1p"].fillna(0)>0)
        rows += [{"metric":"n_pa_adopters_obs","value":int(adopter.sum())},{"metric":"share_pa_adopters_obs","value":float(adopter.mean())}]
    for c in vars_core + vis_vars + [v for v in ["occupation_onet_similarity"] if v in df.columns]:
        s = df[c].dropna()
        rows += [{"metric":f"mean_{c}","value":float(s.mean()) if len(s) else np.nan},{"metric":f"sd_{c}","value":float(s.std()) if len(s) else np.nan}]
    if "crosswalk_method" in df.columns:
        sh = df["crosswalk_method"].value_counts(normalize=True, dropna=False)
        for k,v in sh.items():
            rows.append({"metric":f"crosswalk_share::{k}","value":float(v)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(paper/"table1_data_adoption_visibility_summary.csv", index=False)
    ensure_tex(t1, paper/"table1_data_adoption_visibility_summary.tex")

    t2src = pick(out_root/"02_event_study_main"/"02_event_study_pretrend_and_postsummary.csv")
    if t2src:
        t2 = pd.read_csv(t2src)
        t2 = t2[t2["outcome"].isin(MAIN_OUTCOMES)] if "outcome" in t2.columns else t2
    else:
        t2 = pd.DataFrame(columns=["outcome"])
    t2.to_csv(paper/"table2_baseline_event_study_post_effects.csv", index=False)
    ensure_tex(t2, paper/"table2_baseline_event_study_post_effects.tex")

    t3src = pick(out_root/"03_event_study_visibility_heterogeneity"/"02_internal_vs_external_post_effect_ranking.csv")
    t3 = pd.read_csv(t3src) if t3src else pd.DataFrame()
    t3.to_csv(paper/"table3_internal_external_visibility_heterogeneity.csv", index=False)
    ensure_tex(t3, paper/"table3_internal_external_visibility_heterogeneity.tex")

    t4 = pd.DataFrame([
        ["PA reduces exits / improves retention","exit_rate",np.nan,"negative","", "", ""],
        ["PA reduces replacement hiring","hire_rate",np.nan,"negative","", "", ""],
        ["PA changes formal promotions (ambiguous)","promotion_rate_continuers",np.nan,"ambiguous","", "", "interpret with caution"],
        ["PA increases skill dispersion/specialization","skill_bundle_dispersion / specialist_share",np.nan,"positive","", "", ""],
        ["PA changes HR/managerial structure","hr_to_employee_ratio / managers_to_employee_ratio",np.nan,"ambiguous","", "", ""],
        ["Internal visibility amplifies effects","internal interaction",np.nan,"stronger","", "", ""],
        ["External visibility attenuates effects","external interaction",np.nan,"weaker","", "", ""],
        ["Wage compression direct test","wage outcomes",np.nan,"muted","", "", "requires wage variables in panel"],
    ], columns=["Theory prediction","Empirical outcome","Main coefficient/effect","Direction predicted by model","Direction found in data","Supports model? Yes/Partial/No","Notes"])
    t4.to_csv(paper/"table4_mechanism_evidence_map.csv", index=False)
    ensure_tex(t4, paper/"table4_mechanism_evidence_map.tex")

    t5src = pick(out_root/"04_exposure_intensity_regressions"/"03_summary_table.csv")
    t5 = pd.read_csv(t5src) if t5src else pd.DataFrame()
    t5.to_csv(paper/"table5_robustness_summary.csv", index=False)
    ensure_tex(t5, paper/"table5_robustness_summary.tex")

    fig,ax = plt.subplots(1,3,figsize=(15,4))
    if "year" in df and "pa_posting_log1p" in df:
        g=df.groupby("year")["pa_posting_log1p"].mean(); ax[0].plot(g.index,g.values); ax[0].set_title("A. PA adoption over time")
    if "occ_visibility_internal_static" in df:
        ax[1].hist(df["occ_visibility_internal_static"].dropna(),bins=40); ax[1].set_title("B. Internal visibility")
    if "occ_visibility_external_static" in df:
        ax[2].hist(df["occ_visibility_external_static"].dropna(),bins=40); ax[2].set_title("C. External visibility")
    fig.tight_layout(); fig.savefig(paper/"figure1_adoption_visibility_measurement.png",dpi=160); fig.savefig(paper/"figure1_adoption_visibility_measurement.pdf"); plt.close(fig)

    c2src = pick(out_root/"02_event_study_main"/"01_event_study_coefficients.csv")
    if c2src:
        c2=pd.read_csv(c2src)
        outcomes=[o for o in ["exit_rate","hire_rate","skill_bundle_dispersion","specialist_share"] if o in c2["outcome"].unique()]
        fig,axs=plt.subplots(2,2,figsize=(10,8)); axs=axs.flatten()
        for i,o in enumerate(outcomes):
            d=c2[(c2.outcome==o) & (c2.weight_mode=="unweighted")].sort_values("event_time")
            axs[i].errorbar(d.event_time,d.estimate,yerr=1.96*d.std_error,fmt='o-'); axs[i].axhline(0,color='k'); axs[i].axvline(0,color='gray',ls='--'); axs[i].set_title(o)
        fig.tight_layout(); fig.savefig(paper/"figure2_main_event_studies.png",dpi=160); fig.savefig(paper/"figure2_main_event_studies.pdf"); plt.close(fig)

    c3src = pick(out_root/"03_event_study_visibility_heterogeneity"/"01_visibility_heterogeneity_coefficients.csv")
    if c3src:
        c3=pd.read_csv(c3src)
        outs=[o for o in ["exit_rate","hire_rate","skill_bundle_dispersion","specialist_share"] if o in c3.outcome.unique()]
        fig,axs=plt.subplots(2,2,figsize=(10,8)); axs=axs.flatten()
        for i,o in enumerate(outs):
            d=c3[(c3.outcome==o)&(c3.reg_type=="high_internal")&(c3.event_time!=999)].sort_values("event_time")
            if len(d):
                axs[i].errorbar(d.event_time,d.estimate,yerr=1.96*d.std_error,fmt='o-')
            axs[i].axhline(0,color='k'); axs[i].set_title(o)
        fig.tight_layout(); fig.savefig(paper/"figure3_internal_visibility_event_studies.png",dpi=160); fig.savefig(paper/"figure3_internal_visibility_event_studies.pdf"); plt.close(fig)

        d4=[]
        for o in c3.outcome.unique():
            di=c3[(c3.outcome==o)&(c3.reg_type=="z_internal")&(c3.event_time.between(0,4))]
            de=c3[(c3.outcome==o)&(c3.reg_type=="z_external")&(c3.event_time.between(0,4))]
            if len(di) and len(de):
                d4.append({"outcome":o,"internal":di.estimate.mean(),"external":de.estimate.mean(),"internal_minus_external":di.estimate.mean()-de.estimate.mean()})
        d4=pd.DataFrame(d4)
        d4.to_csv(paper/"figure4_internal_vs_external_visibility_coefficients_data.csv",index=False)
        if len(d4):
            fig,ax=plt.subplots(figsize=(8,5)); y=np.arange(len(d4)); ax.scatter(d4.internal,y,label='internal'); ax.scatter(d4.external,y,label='external'); ax.axvline(0,color='k'); ax.set_yticks(y); ax.set_yticklabels(d4.outcome); ax.legend(); fig.tight_layout(); fig.savefig(paper/"figure4_internal_vs_external_visibility_coefficients.png",dpi=160); fig.savefig(paper/"figure4_internal_vs_external_visibility_coefficients.pdf"); plt.close(fig)

    if c2src:
        c2=pd.read_csv(c2src)
        sel=[]
        for o in ["exit_rate","hire_rate","skill_bundle_dispersion"]:
            d=c2[(c2.outcome==o)&(c2.weight_mode=="unweighted")].sort_values("event_time")
            if len(d):
                dd=d[["event_time","estimate"]].copy(); dd["outcome"]=o; sel.append(dd)
        if sel:
            p5=pd.concat(sel,ignore_index=True)
            p5.to_csv(paper/"figure5_mechanism_timing_data.csv",index=False)
            fig,ax=plt.subplots(figsize=(8,5))
            for o,g in p5.groupby("outcome"):
                ax.plot(g.event_time,g.estimate,label=o)
            ax.axhline(0,color='k'); ax.axvline(0,color='gray',ls='--'); ax.legend(); fig.tight_layout(); fig.savefig(paper/"figure5_mechanism_timing.png",dpi=160); fig.savefig(paper/"figure5_mechanism_timing.pdf"); plt.close(fig)

    mapping = {
        "A1_full_missingness_support_tables.csv": out_root/"00_audit_repo_and_data"/"missingness.csv",
        "A2_correlation_matrix_exposure_visibility.csv": out_root/"01_descriptive_facts"/"03_corr_exposure_visibility.csv",
        "A3_outcome_means_by_visibility_quartile_internal.csv": out_root/"01_descriptive_facts"/"04_outcome_means_by_internal_visibility_quartile.csv",
        "A3_outcome_means_by_visibility_quartile_external.csv": out_root/"01_descriptive_facts"/"05_outcome_means_by_external_visibility_quartile.csv",
        "A4_full_event_study_coefficients.csv": out_root/"02_event_study_main"/"01_event_study_coefficients.csv",
        "A5_full_pretrend_tests.csv": out_root/"02_event_study_main"/"02_event_study_pretrend_and_postsummary.csv",
        "A6_weighted_vs_unweighted_results.csv": out_root/"02_event_study_main"/"01_event_study_coefficients.csv",
    }
    for target, src in mapping.items():
        if src.exists():
            pd.read_csv(src).to_csv(app/target,index=False)
        else:
            pd.DataFrame().to_csv(app/target,index=False)
    for name in ["A7_alternative_FE_specifications.csv","A8_small_cell_restrictions.csv","A9_crosswalk_quality_restrictions.csv","A10_AI_placebo_controls.csv","A11_fake_adoption_placebo.csv","A12_permuted_visibility_placebo.csv","A13_alternative_transformations.csv","A14_fdr_qvalue_by_family.csv"]:
        pd.DataFrame().to_csv(app/name,index=False)

    manifest_rows=[]
    for fp in sorted(paper.glob("*")):
        if fp.is_file():
            manifest_rows.append({
                "file_name": fp.name,
                "table_figure_number": fp.name.split("_")[0],
                "script_that_generated_it": "10_build_paper_package.py",
                "input_data": str(paths.input_panel),
                "model_formula": "see underlying stage outputs",
                "fe": "parent_occ + year (where applicable)",
                "clustering": "parent_rcid (where applicable)",
                "sample_restriction": "see upstream module",
                "main_or_appendix": "main paper",
            })
    for fp in sorted(app.glob("*")):
        if fp.is_file():
            manifest_rows.append({
                "file_name": fp.name,
                "table_figure_number": fp.name.split("_")[0],
                "script_that_generated_it": "10_build_paper_package.py",
                "input_data": str(paths.input_panel),
                "model_formula": "see underlying stage outputs",
                "fe": "see upstream module",
                "clustering": "see upstream module",
                "sample_restriction": "see upstream module",
                "main_or_appendix": "appendix",
            })
    pd.DataFrame(manifest_rows).to_csv(paper/"paper_package_manifest.csv", index=False)

    write_manifest(paper/"manifest.json", {"module":"10_build_paper_package","paper_dir":str(paper),"appendix_dir":str(app)})
    logger.info("Built paper package at %s", paper)

if __name__ == "__main__":
    main()
