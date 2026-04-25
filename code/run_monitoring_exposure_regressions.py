#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import List
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F

OUTCOMES = ["d5_log_workers","d5_exit_rate","d5_hire_rate","d5_skill_count_sd","d5_skill_bundle_dispersion","d5_skill_hhi_mean","d5_specialist_share"]
EXPOSURES = ["monitoring_exposure_average","monitoring_exposure_concentration"]

def parse_args():
    p=argparse.ArgumentParser(description="Table-5-style regressions using O*NET monitoring exposure.")
    p.add_argument("--panel-dir", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--max-iter", type=int, default=30); p.add_argument("--tol", type=float, default=1e-8)
    return p.parse_args()
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_json(o,p): ensure_dir(str(Path(p).parent)); open(p,"w").write(json.dumps(o,indent=2,sort_keys=True,default=str))
def ncdf(x): return .5*(1+math.erf(x/math.sqrt(2)))
def pval(t): return 2*(1-ncdf(abs(t)))
def resid_twfe(df, cols, fe1, fe2, max_iter, tol):
    out=df[cols].astype(float).copy(); last=None
    for _ in range(max_iter):
        for c in cols:
            out[c]=out[c]-out.groupby(df[fe1])[c].transform("mean")
            out[c]=out[c]-out.groupby(df[fe2])[c].transform("mean")
        norm=float(np.sqrt(np.nanmean(out[cols].to_numpy()**2)))
        if last is not None and abs(last-norm)<tol: break
        last=norm
    return out
def fit(y,X,cl):
    y=np.asarray(y,float); X=np.asarray(X,float); cl=np.asarray(cl)
    ok=np.isfinite(y)&np.all(np.isfinite(X),axis=1); y,X,cl=y[ok],X[ok],cl[ok]
    n,k=X.shape
    if n<=k: return None
    inv=np.linalg.pinv(X.T@X); b=inv@(X.T@y); u=y-X@b
    meat=np.zeros((k,k)); groups=pd.unique(cl)
    for g in groups:
        idx=np.where(cl==g)[0]; s=X[idx].T@u[idx]; meat+=np.outer(s,s)
    G=len(groups); corr=(G/(G-1))*((n-1)/(n-k)) if G>1 and n>k else 1
    V=corr*inv@meat@inv; se=np.sqrt(np.maximum(np.diag(V),0)); t=b/se; p=np.array([pval(x) for x in t])
    r2=1-np.sum(u**2)/np.sum((y-y.mean())**2)
    return b,se,t,p,n,G,r2

def main():
    args=parse_args(); ensure_dir(args.out_dir)
    spark=SparkSession.builder.appName("monitoring_exposure_table5_regressions").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    df0=spark.read.parquet(args.panel_dir)
    rows=[]; diags=[]
    for y in [c for c in OUTCOMES if c in df0.columns]:
        cols=["parent_rcid","occupation","year",y,"log_n_workers","monitoring_exposure_average","monitoring_exposure_concentration","occupation_onet_similarity"]
        cols=[c for c in cols if c in df0.columns]
        pdf=(df0.where(F.col(y).isNotNull()).where(F.col("log_n_workers").isNotNull())
             .where(F.col("monitoring_exposure_average").isNotNull()).where(F.col("monitoring_exposure_concentration").isNotNull())
             .select(*cols).toPandas().replace([np.inf,-np.inf],np.nan))
        if pdf.empty: continue
        pdf["parent_year_fe"]=pdf["parent_rcid"].astype(str)+"_y"+pdf["year"].astype(int).astype(str)
        pdf["occupation_year_fe"]=pdf["occupation"].astype(str)+"_y"+pdf["year"].astype(int).astype(str)
        use=pdf.dropna(subset=[y,"log_n_workers"]+EXPOSURES+["parent_year_fe","occupation_year_fe"]).copy()
        if use.empty: continue
        for c in EXPOSURES:
            sd=use[c].std(); use["std_"+c]=(use[c]-use[c].mean())/sd if pd.notna(sd) and sd>0 else 0.0
        xcols=["std_monitoring_exposure_average","std_monitoring_exposure_concentration","log_n_workers"]
        r=resid_twfe(use, [y]+xcols, "parent_year_fe", "occupation_year_fe", args.max_iter, args.tol)
        out=fit(r[y], r[xcols], use["parent_rcid"])
        if out is None: continue
        b,se,t,p,n,G,r2=out
        for i,term in enumerate(xcols):
            rows.append({"outcome":y,"term":term,"coef":float(b[i]),"std_err":float(se[i]),"t_stat":float(t[i]),"p_value":float(p[i]),"nobs":int(n),"n_clusters":int(G),"r2_resid":float(r2),"fixed_effects":"parent_year + occupation_year"})
        diags.append({"outcome":y,"nobs":int(len(use)),"n_clusters":int(use.parent_rcid.nunique()),"y_mean":float(use[y].mean()),"y_sd":float(use[y].std()),"mean_onet_match_similarity":float(use.occupation_onet_similarity.mean()) if "occupation_onet_similarity" in use else np.nan})
    res=pd.DataFrame(rows); diag=pd.DataFrame(diags)
    res.to_csv(os.path.join(args.out_dir,"10_monitoring_exposure_table5_results.csv"),index=False)
    diag.to_csv(os.path.join(args.out_dir,"11_monitoring_exposure_diagnostics.csv"),index=False)
    save_json({"panel_dir":args.panel_dir,"out_dir":args.out_dir,"outcomes":[c for c in OUTCOMES if c in df0.columns]}, os.path.join(args.out_dir,"00_metadata.json"))
    spark.stop()
if __name__=="__main__": main()
