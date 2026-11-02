#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.ood import msp_id_score,energy_id_score,sme_id_score,fit_known_quantile_threshold
from source_attribution.metrics import open_set_metrics
from source_attribution.utils import json_dump,set_seed

def score(method,model,X,cfg):
    if method=="msp":return msp_id_score(model.predict_proba(X))
    if method=="energy":return energy_id_score(model.decision_scores(X),cfg["ood"].get("temperature_energy",1.0))
    if method=="sme":return sme_id_score(model.decision_scores(X),cfg["ood"].get("temperature_sme",4.0))
    raise ValueError("External OOD supports msp/energy/sme; prototype is evaluated in run_open_set.py.")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--features",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--config",required=True);ap.add_argument("--unknown-generator",required=True);a=ap.parse_args()
    cfg=load_config(a.config);seed=int(cfg["experiment"].get("seed",42));set_seed(seed);f=pd.read_parquet(a.features);sp=pd.read_csv(a.splits)
    if "generator" in f:f=f.drop(columns=["generator"])
    if "split" in f:f=f.drop(columns=["split"])
    df=sp.merge(f,on="file_id",how="inner");u=str(a.unknown_generator);g=df["generator"].astype(str);tr=(df["split"]=="train")&(g!=u);va=(df["split"]=="validation")&(g!=u);kt=(df["split"]=="test")&(g!=u);ut=(df["split"]=="test")&(g==u)
    if ut.sum()==0:raise RuntimeError(f"No test samples for external OOD generator {u}")
    fs=cfg.get("feature_selection",{});cols=feature_columns(df,fs.get("include_prefixes"),fs.get("exclude_columns"),bool(fs.get("exclude_duration",False)));X=df[cols].replace([np.inf,-np.inf],np.nan);mc=cfg["model"];model=build_and_train(X.loc[tr],df.loc[tr,"generator"],X.loc[va],df.loc[va,"generator"],mc["type"],mc.get("params",{}),80,seed);kp=model.predict(X.loc[kt]);y=df.loc[kt,"generator"].astype(str).to_numpy();res={}
    for method in [m for m in cfg["ood"]["methods"] if m!="prototype"]:
        vs=score(method,model,X.loc[va],cfg);ks=score(method,model,X.loc[kt],cfg);us=score(method,model,X.loc[ut],cfg);th=fit_known_quantile_threshold(vs,cfg["ood"].get("known_reject_quantile",.05));met=open_set_metrics(y,kp,ks,us);met["unknown_reject_rate"]=float(np.mean(us<th));res[method]=met
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"]+"_external_"+u.replace("/","_"),seed);record_run_context(run,cfg);json_dump({"unknown_generator":u,"metrics":res},run/"metrics.json");print(res);print(run)
if __name__=="__main__":main()
