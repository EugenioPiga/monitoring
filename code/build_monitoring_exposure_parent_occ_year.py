#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from monitoring_exposure_utils import create_spark, ensure_dir, save_json, write_parquet

APPLICATION_DESCRIPTIONS = {
    "attrition_retention_prediction": "predict employee attrition turnover retention flight risk and worker churn",
    "performance_dashboards": "build performance dashboards scorecards KPIs and productivity metrics for workers",
    "employee_engagement_listening": "analyze employee engagement listening pulse surveys employee sentiment and workplace experience",
    "workforce_planning_forecasting": "forecast staffing needs workforce planning headcount planning scheduling and labor demand",
    "productivity_monitoring": "monitor employee productivity activity tracking digital activity workflow tracking and time use",
    "performance_review_management": "automate performance reviews performance management feedback calibration and employee evaluation",
    "promotion_succession_talent": "predict promotion potential succession planning talent mobility career path and high potential employees",
    "compliance_quality_monitoring": "monitor compliance quality assurance call quality audits and standardized performance",
    "organizational_network_analysis": "analyze collaboration networks organizational network analysis people networks and communication patterns",
    "learning_skills_analytics": "analyze skills intelligence learning analytics training analytics capabilities and competencies",
}
OUTCOME_COLS = ["d5_log_workers","d5_exit_rate","d5_hire_rate","d5_skill_count_sd","d5_skill_bundle_dispersion","d5_skill_hhi_mean","d5_specialist_share","exit_rate","hire_rate","skill_count_sd","skill_bundle_dispersion","skill_hhi_mean","specialist_share"]

def parse_args():
    p = argparse.ArgumentParser(description="Build parent-occupation-year monitoring exposure average/concentration from O*NET tasks.")
    p.add_argument("--parent-occ-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/parent_occupation_year_panel")
    p.add_argument("--applications-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_applications_parent_year")
    p.add_argument("--onet-task-weights-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_task_weights")
    p.add_argument("--out-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/final/monitoring_exposure_parent_occ_year")
    p.add_argument("--diagnostics-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/diagnostics/monitoring_exposure_parent_occ_year")
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--similarity-threshold-quantile", type=float, default=0.95)
    p.add_argument("--min-similarity", type=float, default=0.05)
    p.add_argument("--shuffle-partitions", type=int, default=1000)
    p.add_argument("--coalesce", type=int, default=250)
    p.add_argument("--tmpdir", default=None)
    return p.parse_args()

def norm(s):
    s = "" if s is None else str(s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()

def cosine_tfidf(a_texts: List[str], b_texts: List[str]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        X = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1).fit_transform(a_texts + b_texts)
        return cosine_similarity(X[:len(a_texts)], X[len(a_texts):])
    except Exception as e:
        print(f"[WARN] sklearn unavailable ({e}); using token Jaccard.", flush=True)
        A = [set(norm(x).split()) for x in a_texts]; B = [set(norm(x).split()) for x in b_texts]
        M = np.zeros((len(A), len(B)))
        for i,a in enumerate(A):
            for j,b in enumerate(B):
                M[i,j] = len(a&b)/len(a|b) if len(a|b) else 0
        return M

def build_app_task(onet_pdf, q, min_sim):
    app_names = list(APPLICATION_DESCRIPTIONS)
    sims = cosine_tfidf([APPLICATION_DESCRIPTIONS[a] for a in app_names], onet_pdf["task_text"].fillna("").tolist())
    tau = max(float(np.quantile(sims.reshape(-1), q)), float(min_sim))
    rows = []
    for i, app in enumerate(app_names):
        idx = np.where(sims[i] >= tau)[0]
        for j in idx:
            rows.append({"application_category": app, "task_id": str(onet_pdf.iloc[j]["task_id"]), "monitoring_task_similarity": float(sims[i,j]), "task_exposed": 1})
    return pd.DataFrame(rows), tau

def build_crosswalk(occ_pdf, onet_occ_pdf):
    occ = occ_pdf[["occupation"]].drop_duplicates().copy()
    onet = onet_occ_pdf[["onet_soc_code","onet_title","onet_description"]].drop_duplicates().copy()
    sims = cosine_tfidf(occ["occupation"].map(norm).tolist(), (onet["onet_title"].fillna("")+" "+onet["onet_description"].fillna("")).map(norm).tolist())
    best = sims.argmax(axis=1)
    out = occ.copy()
    out["onet_soc_code"] = onet.iloc[best]["onet_soc_code"].values
    out["onet_title"] = onet.iloc[best]["onet_title"].values
    out["occupation_onet_similarity"] = sims.max(axis=1)
    return out

def main():
    args = parse_args()
    ensure_dir(args.out_dir); ensure_dir(args.diagnostics_dir)
    spark = create_spark("build_monitoring_exposure_parent_occ_year", args.shuffle_partitions, args.tmpdir)

    parent_occ = spark.read.parquet(args.parent_occ_dir)
    apps = spark.read.parquet(args.applications_dir)
    onet_tasks = spark.read.parquet(args.onet_task_weights_dir)

    keep = ["parent_rcid","occupation","year","n_workers","pa_posting_log1p","event_time_posting"] + [c for c in OUTCOME_COLS if c in parent_occ.columns]
    poc = (parent_occ.where(F.col("occupation_analysis_sample")==1)
           .where((F.col("year")>=args.start_year)&(F.col("year")<=args.end_year))
           .select(*keep).cache())
    _ = poc.count()

    onet_pdf = onet_tasks.select("onet_soc_code","onet_title","onet_description","task_id","task_text","task_weight","task_importance").toPandas()
    onet_pdf["task_id"] = onet_pdf["task_id"].astype(str)
    task_unique = onet_pdf[["task_id","task_text"]].drop_duplicates("task_id")
    sim_pdf, tau = build_app_task(task_unique, args.similarity_threshold_quantile, args.min_similarity)
    if sim_pdf.empty:
        raise RuntimeError("No app-task matches. Lower threshold.")
    sim_sdf = spark.createDataFrame(sim_pdf)
    write_parquet(sim_sdf, os.path.join(args.diagnostics_dir, "02_application_task_similarity"), 1)

    occ_pdf = poc.select("occupation").distinct().toPandas()
    cw_pdf = build_crosswalk(occ_pdf, onet_pdf[["onet_soc_code","onet_title","onet_description"]].drop_duplicates("onet_soc_code"))
    cw_pdf.to_csv(os.path.join(args.diagnostics_dir, "01_revelio_occupation_to_onet_crosswalk.csv"), index=False)
    cw = spark.createDataFrame(cw_pdf)

    app_counts = (apps.where((F.col("year")>=args.start_year)&(F.col("year")<=args.end_year))
                  .groupBy("parent_rcid","year","application_category")
                  .agg(F.sum("application_posting_count").alias("application_posting_count")))
    total = app_counts.groupBy("parent_rcid","year").agg(F.sum("application_posting_count").alias("monitoring_application_count")).withColumn("monitoring_application_log1p", F.log1p("monitoring_application_count"))

    task_py = (app_counts.join(sim_sdf, "application_category", "inner")
               .groupBy("parent_rcid","year","task_id")
               .agg(F.sum(F.col("application_posting_count")*F.col("task_exposed")).alias("task_exposed_weighted_count"),
                    F.sum(F.col("application_posting_count")*F.col("monitoring_task_similarity")).alias("task_similarity_weighted_sum"))
               .join(total, ["parent_rcid","year"], "left")
               .withColumn("xi_task_parent_year", F.col("task_exposed_weighted_count")/F.col("monitoring_application_count"))
               .withColumn("xi_similarity_parent_year", F.col("task_similarity_weighted_sum")/F.col("monitoring_application_count")))

    weights = onet_tasks.select("onet_soc_code", F.col("task_id").cast("string").alias("task_id"), "task_weight")
    po_tasks = poc.join(cw, "occupation", "left").join(weights, "onet_soc_code", "left")
    joined = (po_tasks.join(task_py.select("parent_rcid","year","task_id","xi_task_parent_year","xi_similarity_parent_year","monitoring_application_count","monitoring_application_log1p"),
                            ["parent_rcid","year","task_id"], "left")
              .fillna({"xi_task_parent_year":0.0, "xi_similarity_parent_year":0.0, "monitoring_application_count":0.0, "monitoring_application_log1p":0.0, "task_weight":0.0}))
    agg_expr = [
        F.first("onet_soc_code", ignorenulls=True).alias("onet_soc_code"),
        F.first("onet_title", ignorenulls=True).alias("onet_title"),
        F.first("occupation_onet_similarity", ignorenulls=True).alias("occupation_onet_similarity"),
        F.first("n_workers", ignorenulls=True).alias("n_workers"),
        F.first("pa_posting_log1p", ignorenulls=True).alias("pa_posting_log1p"),
        F.first("event_time_posting", ignorenulls=True).alias("event_time_posting"),
        F.first("monitoring_application_count", ignorenulls=True).alias("monitoring_application_count"),
        F.first("monitoring_application_log1p", ignorenulls=True).alias("monitoring_application_log1p"),
        F.sum(F.col("task_weight")*F.col("xi_task_parent_year")).alias("monitoring_exposure_average_raw"),
        F.sum(F.col("task_weight")*F.col("xi_similarity_parent_year")).alias("monitoring_similarity_average_raw"),
    ] + [F.first(c, ignorenulls=True).alias(c) for c in OUTCOME_COLS if c in poc.columns]
    avg = joined.groupBy("parent_rcid","occupation","year").agg(*agg_expr)
    joined2 = joined.join(avg.select("parent_rcid","occupation","year","monitoring_exposure_average_raw"), ["parent_rcid","occupation","year"], "left") \
                    .withColumn("weighted_sq_dev", F.col("task_weight")*(F.col("xi_task_parent_year")-F.col("monitoring_exposure_average_raw"))**2)
    conc = joined2.groupBy("parent_rcid","occupation","year").agg(F.sum("weighted_sq_dev").alias("monitoring_exposure_concentration_raw"))
    final = (avg.join(conc, ["parent_rcid","occupation","year"], "left")
             .withColumn("monitoring_exposure_average", F.col("monitoring_exposure_average_raw")*F.col("monitoring_application_log1p"))
             .withColumn("monitoring_exposure_concentration", F.col("monitoring_exposure_concentration_raw")*F.col("monitoring_application_log1p"))
             .withColumn("monitoring_similarity_average", F.col("monitoring_similarity_average_raw")*F.col("monitoring_application_log1p"))
             .withColumn("log_n_workers", F.when(F.col("n_workers")>0, F.log("n_workers"))))
    write_parquet(final, args.out_dir, args.coalesce)
    written = spark.read.parquet(args.out_dir)
    meta = {"out_dir":args.out_dir, "rows":written.count(), "parents":written.select("parent_rcid").distinct().count(),
            "occupations":written.select("occupation").distinct().count(), "similarity_threshold":tau,
            "note":"First-pass monitoring exposure: posting application categories mapped to O*NET task statements using TF-IDF cosine similarity."}
    save_json(meta, os.path.join(args.diagnostics_dir, "00_metadata.json"))
    (written.groupBy("year").agg(F.count("*").alias("n_parent_occ_year"), F.countDistinct("parent_rcid").alias("n_parents"),
                                 F.avg("monitoring_exposure_average").alias("mean_monitoring_exposure_average"),
                                 F.avg("monitoring_exposure_concentration").alias("mean_monitoring_exposure_concentration"))
            .orderBy("year").coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(args.diagnostics_dir, "03_yearly_summary_csv")))
    print(meta, flush=True)
    poc.unpersist(); spark.stop()

if __name__ == "__main__":
    main()
