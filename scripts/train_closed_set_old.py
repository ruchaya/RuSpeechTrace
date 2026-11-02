#!/usr/bin/env python
import argparse
from pathlib import Path
import joblib,numpy as np,pandas as pd
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.leakage import assert_no_leakage
from source_attribution.metrics import closed_set_metrics,per_class_metrics
from source_attribution.models import build_and_train
from source_attribution.utils import json_dump,set_seed

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--splits",required=True); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=load_config(a.config); seed=int(cfg["experiment"].get("seed",42)); set_seed(seed)
    feat=pd.read_parquet(a.features); sp=pd.read_csv(a.splits); assert_no_leakage(sp)
    if "generator" in feat: feat=feat.drop(columns=["generator"])
    if "split" in feat: feat=feat.drop(columns=["split"])
    meta=[c for c in ["file_id","split","generator","speaker","voice","augmentation_type","source_dataset"] if c in sp.columns]
    df=sp[meta].merge(feat,on="file_id",how="inner")
    fs=cfg.get("feature_selection",{}); cols=feature_columns(df,fs.get("include_prefixes"),fs.get("exclude_columns"),bool(fs.get("exclude_duration",False)))
    X=df[cols].replace([np.inf,-np.inf],np.nan); tr=df["split"]=="train"; va=df["split"]=="validation"; te=df["split"]=="test"; mc=cfg["model"]
    model=build_and_train(X.loc[tr],df.loc[tr,"generator"],X.loc[va],df.loc[va,"generator"],mc["type"],mc.get("params",{}),int(cfg.get("training",{}).get("early_stopping_rounds",80)),seed)
    proba=model.predict_proba(X.loc[te]); pred=model.predict(X.loc[te]); y=df.loc[te,"generator"].astype(str).to_numpy(); met=closed_set_metrics(y,pred,proba,model.classes_)
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"],seed); record_run_context(run,cfg); json_dump(met,run/"metrics.json")
    pd.DataFrame(per_class_metrics(y,pred)).to_csv(run/"class_metrics.csv",index=False)
    out=df.loc[te,[c for c in ["file_id","generator","speaker","voice","augmentation_type","source_dataset"] if c in df.columns]].copy().rename(columns={"generator":"generator_true"}); out["generator_pred"]=pred; out.to_parquet(run/"predictions.parquet",index=False)
    np.save(run/"probabilities.npy",proba); (run/"classes.txt").write_text("\n".join(map(str,model.classes_)),encoding="utf-8"); (run/"feature_columns.txt").write_text("\n".join(cols),encoding="utf-8"); joblib.dump(model,run/"model.joblib")
    print(met); print(run)
if __name__=="__main__": main()
