#!/usr/bin/env python
import argparse
from pathlib import Path
import joblib,numpy as np,pandas as pd,shap
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-dir",required=True);ap.add_argument("--features",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--max-samples",type=int,default=3000);a=ap.parse_args();run=Path(a.run_dir);bundle=joblib.load(run/"model.joblib")
    if bundle.kind!="lightgbm":raise RuntimeError("SHAP script supports LightGBM.")
    cols=(run/"feature_columns.txt").read_text(encoding="utf-8").splitlines();f=pd.read_parquet(a.features);sp=pd.read_csv(a.splits)[["file_id","split","generator"]]
    if "generator" in f:f=f.drop(columns=["generator"])
    if "split" in f:f=f.drop(columns=["split"])
    df=sp.merge(f,on="file_id",how="inner");test=df[df["split"]=="test"].copy()
    if len(test)>a.max_samples:test=test.sample(a.max_samples,random_state=42)
    X=test[cols].replace([np.inf,-np.inf],np.nan);sv=shap.TreeExplainer(bundle.estimator).shap_values(X);out=run/"shap";out.mkdir(exist_ok=True)
    if isinstance(sv,list):arr=np.stack(sv,axis=1)
    else:
        arr=np.asarray(sv)
        if arr.ndim==3 and arr.shape[1]==len(cols):arr=np.transpose(arr,(0,2,1))
    if arr.ndim==3:
        glob=np.mean(np.abs(arr),axis=(0,1));pd.DataFrame({"feature":cols,"mean_abs_shap":glob}).sort_values("mean_abs_shap",ascending=False).to_csv(out/"global_importance.csv",index=False)
        pd.DataFrame(np.mean(np.abs(arr),axis=0),index=bundle.classes_,columns=cols).to_csv(out/"classwise_mean_abs_shap.csv")
    else:raise RuntimeError(f"Unexpected SHAP shape {arr.shape}")
    print(out)
if __name__=="__main__":main()
