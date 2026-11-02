#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.utils import json_dump,set_seed

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--features",required=True);ap.add_argument("--inventory",required=True);ap.add_argument("--config",required=True);ap.add_argument("--voice-column",default="voice");a=ap.parse_args()
    cfg=load_config(a.config);seed=int(cfg["experiment"].get("seed",42));set_seed(seed);f=pd.read_parquet(a.features);inv=pd.read_csv(a.inventory)[["file_id","generator",a.voice_column]].copy()
    if "generator" in f:f=f.drop(columns=["generator"])
    df=inv.merge(f,on="file_id",how="inner");df=df[df[a.voice_column].notna()].copy();counts=df.groupby("generator")[a.voice_column].nunique();df=df[df["generator"].isin(counts[counts>=2].index)].copy()
    if df["generator"].nunique()<2:raise RuntimeError("Not enough generators with >=2 voices.")
    groups=(df["generator"].astype(str)+"::"+df[a.voice_column].astype(str)).to_numpy();chosen=None
    for tr,te in GroupShuffleSplit(n_splits=20,test_size=.2,random_state=seed).split(df,groups=groups):
        if set(df.iloc[te]["generator"])<=set(df.iloc[tr]["generator"]):chosen=(tr,te);break
    if chosen is None:raise RuntimeError("Could not form valid voice-held-out split.")
    tr,te=chosen;train_df=df.iloc[tr].reset_index(drop=True);g2=(train_df["generator"].astype(str)+"::"+train_df[a.voice_column].astype(str)).to_numpy();tr2,va2=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+1).split(train_df,groups=g2))
    fs=cfg.get("feature_selection",{});cols=feature_columns(df,fs.get("include_prefixes"),fs.get("exclude_columns"),bool(fs.get("exclude_duration",False)));mc=cfg["model"];model=build_and_train(train_df.iloc[tr2][cols],train_df.iloc[tr2]["generator"],train_df.iloc[va2][cols],train_df.iloc[va2]["generator"],mc["type"],mc.get("params",{}),80,seed)
    test=df.iloc[te];p=model.predict(test[cols]);pr=model.predict_proba(test[cols]);y=test["generator"].astype(str).to_numpy();met=closed_set_metrics(y,p,pr,model.classes_);run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"]+"_voice_held_out",seed);record_run_context(run,cfg);json_dump(met,run/"metrics.json");o=test[["file_id","generator",a.voice_column]].copy();o["generator_pred"]=p;o.to_parquet(run/"predictions.parquet",index=False);print(met);print(run)
if __name__=="__main__": main()
