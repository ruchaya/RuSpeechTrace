#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from source_attribution.config import load_config
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.experiment import create_run_dir,record_run_context
from source_attribution.leakage import assert_no_leakage
from source_attribution.utils import json_dump,set_seed

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--inventory",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--config",required=True);a=ap.parse_args()
    cfg=load_config(a.config);seed=int(cfg["experiment"].get("seed",42));set_seed(seed);inv=pd.read_csv(a.inventory);sp=pd.read_csv(a.splits);assert_no_leakage(sp)
    cols=[c for c in ["duration","sample_rate","channels"] if c in inv.columns]
    if not cols:raise RuntimeError("No numeric metadata fields available.")
    df=sp[["file_id","split","generator"]].merge(inv[["file_id"]+cols].drop_duplicates("file_id"),on="file_id",how="left");X=df[cols].replace([np.inf,-np.inf],np.nan);tr=df["split"]=="train";va=df["split"]=="validation";te=df["split"]=="test";mc=cfg["model"]
    model=build_and_train(X.loc[tr],df.loc[tr,"generator"],X.loc[va],df.loc[va,"generator"],mc["type"],mc.get("params",{}),80,seed);p=model.predict(X.loc[te]);pr=model.predict_proba(X.loc[te]);y=df.loc[te,"generator"].astype(str).to_numpy();met=closed_set_metrics(y,p,pr,model.classes_)
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"]+"_metadata_only",seed);record_run_context(run,cfg);json_dump({"features":cols,"metrics":met},run/"metrics.json");o=df.loc[te,["file_id","generator"]].copy();o["generator_pred"]=p;o.to_parquet(run/"predictions.parquet",index=False);print(met);print(run)
if __name__=="__main__":main()
