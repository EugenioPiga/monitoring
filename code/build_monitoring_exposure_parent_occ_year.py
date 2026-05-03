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
    p.add_argument("--onet-dir", default="/labs/khanna/predictive_capital/revelio_people_analytics/processed/external/onet_30_2_text")
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

def find_optional_onet_file(onet_dir: str, target_name: str):
    """
    Recursively find an optional O*NET text file, allowing for nested db_* folders.
    """
    root = Path(onet_dir)
    if not root.exists():
        return None

    target_norm = target_name.lower().replace(" ", "").replace("_", "")

    for p in root.rglob("*.txt"):
        name_norm = p.name.lower().replace(" ", "").replace("_", "")
        if name_norm == target_norm:
            return p

    words = target_name.lower().replace(".txt", "").split()
    for p in root.rglob("*.txt"):
        lname = p.name.lower()
        if all(w in lname for w in words):
            return p

    return None


def read_onet_title_file(path: Path):
    """
    Read an O*NET title file robustly. We only need:
      - O*NET-SOC Code
      - any column containing job/alternate/reported title text.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    code_cols = [c for c in df.columns if c.lower().replace(" ", "") in ["o*net-soccode", "onetsoccode"]]
    if not code_cols:
        code_cols = [c for c in df.columns if "soc" in c.lower() and "code" in c.lower()]
    if not code_cols:
        return pd.DataFrame(columns=["onet_soc_code", "candidate_title", "title_source"])

    code_col = code_cols[0]

    title_cols = [c for c in df.columns if "title" in c.lower()]
    if not title_cols:
        return pd.DataFrame(columns=["onet_soc_code", "candidate_title", "title_source"])

    rows = []
    for tc in title_cols:
        tmp = df[[code_col, tc]].copy()
        tmp.columns = ["onet_soc_code", "candidate_title"]
        tmp["candidate_title"] = tmp["candidate_title"].fillna("").astype(str)
        tmp = tmp[tmp["candidate_title"].str.strip().ne("")]
        tmp["title_source"] = path.name
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["onet_soc_code", "candidate_title", "title_source"])

    return pd.concat(rows, ignore_index=True).drop_duplicates()


def build_crosswalk(occ_pdf, onet_occ_pdf, onet_dir=None):
    """
    Improved first-pass crosswalk from Revelio role_k150 labels to O*NET-SOC.

    Uses:
      1. manual overrides for common short Revelio labels;
      2. O*NET official titles;
      3. O*NET Alternate Titles;
      4. O*NET Sample of Reported Titles;
      5. character n-gram TF-IDF over title candidates.
    """

    manual = {
        "accountant": ("13-2011.00", "Accountants and Auditors"),
        "auditor": ("13-2011.00", "Accountants and Auditors"),
        "designer": ("27-1024.00", "Graphic Designers"),
        "ux designer": ("15-1255.01", "Video Game Designers"),
        "writer": ("27-3043.00", "Writers and Authors"),
        "technician": ("49-9099.00", "Installation, Maintenance, and Repair Workers, All Other"),
        "mechanic": ("49-3023.00", "Automotive Service Technicians and Mechanics"),
        "receptionist": ("43-4171.00", "Receptionists and Information Clerks"),
        "cashier": ("41-2011.00", "Cashiers"),
        "recruiter": ("13-1071.00", "Human Resources Specialists"),
        "banker": ("13-2072.00", "Loan Officers"),
        "wealth manager": ("13-2052.00", "Personal Financial Advisors"),
        "realtor": ("41-9022.00", "Real Estate Sales Agents"),
        "economist": ("19-3011.00", "Economists"),
        "geologist": ("19-2042.00", "Geoscientists, Except Hydrologists and Geographers"),
        "pilot": ("53-2012.00", "Commercial Pilots"),
        "scientist": ("19-1099.00", "Life Scientists, All Other"),
        "marketing": ("13-1161.00", "Market Research Analysts and Marketing Specialists"),
        "brand manager": ("11-2021.00", "Marketing Managers"),
        "merchandiser": ("27-1026.00", "Merchandise Displayers and Window Trimmers"),
        "foreman": ("47-1011.00", "First-Line Supervisors of Construction Trades and Extraction Workers"),
        "cleaner": ("37-2011.00", "Janitors and Cleaners, Except Maids and Housekeeping Cleaners"),
        "qa tester": ("15-1253.00", "Software Quality Assurance Analysts and Testers"),
        "sap consultant": ("15-1232.00", "Computer User Support Specialists"),
        "coordinator": ("13-1199.00", "Business Operations Specialists, All Other"),
        "customer success specialist": ("43-4051.00", "Customer Service Representatives"),
        "customer experience specialist": ("43-4051.00", "Customer Service Representatives"),
        "case manager": ("21-1023.00", "Mental Health and Substance Abuse Social Workers"),
        "claims specialist": ("13-1031.00", "Claims Adjusters, Examiners, and Investigators"),
        "content specialist": ("27-3042.00", "Technical Writers"),
        "documentation specialist": ("43-9061.00", "Office Clerks, General"),
        "it specialist": ("15-1232.00", "Computer User Support Specialists"),
        "it analyst": ("15-1211.00", "Computer Systems Analysts"),
        "agent": ("43-4051.00", "Customer Service Representatives"),
        "officer": ("11-9199.00", "Managers, All Other"),
        "md": ("29-1215.00", "Family Medicine Physicians"),
        "am": ("11-1021.00", "General and Operations Managers"),
        "stylist": ("39-5012.00", "Hairdressers, Hairstylists, and Cosmetologists"),
        "genius": ("15-1232.00", "Computer User Support Specialists"),
        "transformation specialist": ("13-1199.00", "Business Operations Specialists, All Other"),

        # Extra mappings motivated by weak-match diagnostics.
        "subject matter expert": ("13-1199.00", "Business Operations Specialists, All Other"),
        "student intern": ("13-1199.00", "Business Operations Specialists, All Other"),
        "contracts specialist": ("13-1023.00", "Purchasing Agents, Except Wholesale, Retail, and Farm Products"),
        "collections specialist": ("43-3011.00", "Bill and Account Collectors"),
        "billing specialist": ("43-3021.00", "Billing and Posting Clerks"),
        "delivery manager": ("11-1021.00", "General and Operations Managers"),
        "fraud analyst": ("13-2099.04", "Fraud Examiners, Investigators and Analysts"),
        "coach": ("27-2022.00", "Coaches and Scouts"),
        "producer": ("27-2012.01", "Producers"),
        "application engineer": ("15-1252.00", "Software Developers"),
        "devops engineer": ("15-1252.00", "Software Developers"),
        "infrastructure engineer": ("15-1244.00", "Network and Computer Systems Administrators"),
        "account manager": ("11-2022.00", "Sales Managers"),
        "project manager": ("13-1082.00", "Project Management Specialists"),
        "project administrator": ("13-1082.00", "Project Management Specialists"),
        "solutions specialist": ("15-1299.08", "Computer Systems Engineers/Architects"),
        "network specialist": ("15-1244.00", "Network and Computer Systems Administrators"),
        "restaurant manager": ("11-9051.00", "Food Service Managers"),
        "payroll specialist": ("43-3051.00", "Payroll and Timekeeping Clerks"),
        "hr business partner": ("13-1071.00", "Human Resources Specialists"),
        "digital marketing specialist": ("13-1161.00", "Market Research Analysts and Marketing Specialists"),
        "corporate trainer": ("13-1151.00", "Training and Development Specialists"),
        "business analyst": ("13-1111.00", "Management Analysts"),
        "product manager": ("11-2021.00", "Marketing Managers"),
        "logistics": ("13-1081.00", "Logisticians"),
        "crew member": ("35-3023.00", "Fast Food and Counter Workers"),
        "driver": ("53-3032.00", "Heavy and Tractor-Trailer Truck Drivers"),
        "pharmacy technician": ("29-2052.00", "Pharmacy Technicians"),
        "corporate strategy": ("13-1111.00", "Management Analysts"),
        "conseiller commercial": ("41-3091.00", "Sales Representatives of Services, Except Advertising, Insurance, Financial Services, and Travel"),
        "business support": ("13-1199.00", "Business Operations Specialists, All Other"),
        "support staff": ("43-9061.00", "Office Clerks, General"),
        "vendor management": ("13-1023.00", "Purchasing Agents, Except Wholesale, Retail, and Farm Products"),
        "customer support": ("43-4051.00", "Customer Service Representatives"),
        "sales support": ("41-2031.00", "Retail Salespersons"),
        "medical rep": ("41-4011.00", "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products"),
        "client services": ("43-4051.00", "Customer Service Representatives"),
        "quality assurance": ("15-1253.00", "Software Quality Assurance Analysts and Testers"),
        "machine operator": ("51-9199.00", "Production Workers, All Other"),
        "planning manager": ("11-1021.00", "General and Operations Managers"),
        "commercial manager": ("11-2022.00", "Sales Managers"),
        "distribution specialist": ("43-5071.00", "Shipping, Receiving, and Inventory Clerks"),
        "web developer": ("15-1254.00", "Web Developers"),
        "quality engineer": ("17-2112.00", "Industrial Engineers"),
        "technology lead": ("15-1211.00", "Computer Systems Analysts"),
        "electrical engineer": ("17-2071.00", "Electrical Engineers"),
        "communications specialist": ("27-3031.00", "Public Relations Specialists"),
        "sustainability specialist": ("13-1199.05", "Sustainability Specialists"),
        "technology analyst": ("15-1211.00", "Computer Systems Analysts"),
        "technical writer": ("27-3042.00", "Technical Writers"),
    }

    occ = occ_pdf[["occupation"]].drop_duplicates().copy()
    occ["occ_clean"] = occ["occupation"].map(norm)

    onet = onet_occ_pdf[["onet_soc_code", "onet_title", "onet_description"]].drop_duplicates().copy()
    onet["candidate_title"] = onet["onet_title"].fillna("")
    onet["title_source"] = "Occupation Data"

    candidates = onet[["onet_soc_code", "onet_title", "candidate_title", "title_source"]].copy()

    # Add O*NET Alternate Titles and Sample of Reported Titles when available.
    if onet_dir is not None:
        alt_path = find_optional_onet_file(onet_dir, "Alternate Titles.txt")
        sample_path = find_optional_onet_file(onet_dir, "Sample of Reported Titles.txt")

        extra = []
        for path in [alt_path, sample_path]:
            if path is None:
                continue
            title_df = read_onet_title_file(path)
            if title_df.empty:
                continue
            title_df = title_df.merge(
                onet[["onet_soc_code", "onet_title"]].drop_duplicates(),
                on="onet_soc_code",
                how="left",
            )
            extra.append(title_df[["onet_soc_code", "onet_title", "candidate_title", "title_source"]])

        if extra:
            candidates = pd.concat([candidates] + extra, ignore_index=True).drop_duplicates()

    candidates["candidate_clean"] = candidates["candidate_title"].map(norm)
    candidates = candidates[candidates["candidate_clean"].ne("")].copy()

    manual_rows = []
    remaining_rows = []

    # Manual exact first.
    for _, r in occ.iterrows():
        key = r["occ_clean"]
        if key in manual:
            code, title = manual[key]
            manual_rows.append({
                "occupation": r["occupation"],
                "onet_soc_code": code,
                "onet_title": title,
                "occupation_onet_similarity": 1.0,
                "crosswalk_method": "manual_exact",
                "matched_candidate_title": title,
                "matched_title_source": "manual",
            })
        else:
            remaining_rows.append(r)

    rem = pd.DataFrame(remaining_rows)
    if rem.empty:
        return pd.DataFrame(manual_rows)

    # Exact match against O*NET title dictionary.
    exact_rows = []
    still_remaining = []
    cand_exact = candidates.drop_duplicates("candidate_clean").set_index("candidate_clean")

    for _, r in rem.iterrows():
        key = r["occ_clean"]
        if key in cand_exact.index:
            m = cand_exact.loc[key]
            exact_rows.append({
                "occupation": r["occupation"],
                "onet_soc_code": m["onet_soc_code"],
                "onet_title": m["onet_title"],
                "occupation_onet_similarity": 0.99,
                "crosswalk_method": "onet_title_exact",
                "matched_candidate_title": m["candidate_title"],
                "matched_title_source": m["title_source"],
            })
        else:
            still_remaining.append(r)

    rem2 = pd.DataFrame(still_remaining)
    if rem2.empty:
        return pd.concat([pd.DataFrame(manual_rows), pd.DataFrame(exact_rows)], ignore_index=True)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = rem2["occ_clean"].tolist() + candidates["candidate_clean"].tolist()
        X = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
        ).fit_transform(corpus)
        sims = cosine_similarity(X[:len(rem2)], X[len(rem2):])

    except Exception as e:
        print(f"[WARN] sklearn char-ngram title crosswalk failed ({e}); using token Jaccard.", flush=True)
        A = [set(x.split()) for x in rem2["occ_clean"]]
        B = [set(x.split()) for x in candidates["candidate_clean"]]
        sims = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                sims[i, j] = len(a & b) / len(a | b) if len(a | b) else 0.0

    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    best_cand = candidates.iloc[best].reset_index(drop=True)

    auto = rem2[["occupation"]].reset_index(drop=True).copy()
    auto["onet_soc_code"] = best_cand["onet_soc_code"].values
    auto["onet_title"] = best_cand["onet_title"].values
    auto["occupation_onet_similarity"] = best_sim
    auto["crosswalk_method"] = np.where(best_sim < 0.18, "auto_weak", "auto_title_char_ngram")
    auto["matched_candidate_title"] = best_cand["candidate_title"].values
    auto["matched_title_source"] = best_cand["title_source"].values

    out = pd.concat(
        [pd.DataFrame(manual_rows), pd.DataFrame(exact_rows), auto],
        ignore_index=True,
    )

    return out[
        [
            "occupation",
            "onet_soc_code",
            "onet_title",
            "occupation_onet_similarity",
            "crosswalk_method",
            "matched_candidate_title",
            "matched_title_source",
        ]
    ]

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
    cw_pdf = build_crosswalk(occ_pdf, onet_pdf[["onet_soc_code","onet_title","onet_description"]].drop_duplicates("onet_soc_code"), args.onet_dir)
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
        F.first("crosswalk_method", ignorenulls=True).alias("crosswalk_method"),
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
