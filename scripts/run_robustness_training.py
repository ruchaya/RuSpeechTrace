#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.utils import json_dump,set_seed

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--features",required=True,help="Feature store containing clean and augmented derivatives.");ap.add_argument("--inventory",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--config",required=True);a=ap.parse_args()
    cfg=load_config(a.config);seed=int(cfg["experiment"].get("seed",42));set_seed(seed);f=pd.read_parquet(a.features);inv=pd.read_csv(a.inventory);sp=pd.read_csv(a.splits)[["file_id","split","generator"]];smap=sp.set_index("file_id")["split"].to_dict();inv["split"]=inv["file_id"].map(smap)
    if "source_audio" in inv:
        miss=inv["split"].isna()&inv["source_audio"].notna();inv.loc[miss,"split"]=inv.loc[miss,"source_audio"].astype(str).map(smap)
    meta=inv[[c for c in ["file_id","generator","augmentation_type","raw_or_augmented","split"] if c in inv.columns]].drop_duplicates("file_id")
    for c in ["generator","augmentation_type","raw_or_augmented","split"]:
        if c in f:f=f.drop(columns=[c])
    df=meta.merge(f,on="file_id",how="inner");fs=cfg.get("feature_selection",{});cols=feature_columns(df,fs.get("include_prefixes"));X=df[cols].replace([np.inf,-np.inf],np.nan);clean_labels=set(str(x).lower() for x in cfg["robustness"].get("clean_labels",["raw","clean","none",""]));aug=df.get("augmentation_type",pd.Series("",index=df.index)).fillna("").astype(str).str.lower();raw=df.get("raw_or_augmented",pd.Series("",index=df.index)).fillna("").astype(str).str.lower();clean=aug.isin(clean_labels)&~raw.str.contains("aug");train=df["split"]=="train";val=clean&(df["split"]=="validation");mc=cfg["model"]
    # Model A: clean-only.
    m_clean=build_and_train(X.loc[train&clean],df.loc[train&clean,"generator"],X.loc[val],df.loc[val,"generator"],mc["type"],mc.get("params",{}),80,seed)
    # Model B: clean + all available training augmentations.
    m_aug=build_and_train(X.loc[train],df.loc[train,"generator"],X.loc[val],df.loc[val,"generator"],mc["type"],mc.get("params",{}),80,seed)
    conditions={"clean":clean&(df["split"]=="test")}
    for c in sorted(set(aug[~clean].dropna())):conditions[c]=(df["split"]=="test")&(aug==c)
    results={}
    for model_name,model in [("clean_trained",m_clean),("augmented_trained",m_aug)]:
        results[model_name]={}
        for cond,mask in conditions.items():
            if not mask.sum():continue
            p=model.predict(X.loc[mask]);pr=model.predict_proba(X.loc[mask]);y=df.loc[mask,"generator"].astype(str).to_numpy();results[model_name][cond]=closed_set_metrics(y,p,pr,model.classes_)
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"]+"_training_comparison",seed);record_run_context(run,cfg);json_dump(results,run/"metrics.json");print(run)
if __name__=="__main__":main()
