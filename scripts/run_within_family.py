#!/usr/bin/env python
import argparse
import pandas as pd,numpy as np
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.utils import json_dump,set_seed
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--features",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--ontology",required=True);ap.add_argument("--config",required=True);a=ap.parse_args();cfg=load_config(a.config);seed=int(cfg["experiment"].get("seed",42));set_seed(seed);f=pd.read_parquet(a.features);sp=pd.read_csv(a.splits);ont=pd.read_csv(a.ontology)
    verified=ont[ont["base_model_family"].fillna("").astype(str).str.strip().ne("")&ont["evidence"].fillna("").astype(str).str.strip().ne("")];mapf=dict(zip(verified["ruasd_label"].astype(str),verified["base_model_family"].astype(str)))
    if "generator" in f:f=f.drop(columns=["generator"])
    if "split" in f:f=f.drop(columns=["split"])
    df=sp[["file_id","split","generator"]].merge(f,on="file_id",how="inner");df["family"]=df["generator"].astype(str).map(mapf);fs=cfg.get("feature_selection",{});cols=feature_columns(df,fs.get("include_prefixes"));mc=cfg["model"];res={}
    for fam,g in df[df["family"].notna()].groupby("family"):
        if g["generator"].nunique()<2:continue
        tr=g["split"]=="train";va=g["split"]=="validation";te=g["split"]=="test"
        if min(tr.sum(),va.sum(),te.sum())==0:continue
        model=build_and_train(g.loc[tr,cols],g.loc[tr,"generator"],g.loc[va,cols],g.loc[va,"generator"],mc["type"],mc.get("params",{}),80,seed);p=model.predict(g.loc[te,cols]);pr=model.predict_proba(g.loc[te,cols]);y=g.loc[te,"generator"].astype(str).to_numpy();res[str(fam)]={"n_generators":int(g["generator"].nunique()),**closed_set_metrics(y,p,pr,model.classes_)}
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"]+"_within_family",seed);record_run_context(run,cfg);json_dump(res,run/"metrics.json");print(res);print(run)
if __name__=="__main__":main()
