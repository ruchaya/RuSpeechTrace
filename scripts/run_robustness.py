#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.utils import json_dump,set_seed

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--inventory",required=True); ap.add_argument("--splits",required=True); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=load_config(a.config); seed=int(cfg["experiment"].get("seed",42)); set_seed(seed); feat=pd.read_parquet(a.features); inv=pd.read_csv(a.inventory); sp=pd.read_csv(a.splits)[["file_id","split","generator"]]
    split_map=sp.set_index("file_id")["split"].to_dict(); inv["split"]=inv["file_id"].map(split_map)
    if "source_audio" in inv:
        miss=inv["split"].isna()&inv["source_audio"].notna(); inv.loc[miss,"split"]=inv.loc[miss,"source_audio"].astype(str).map(split_map)
    meta=inv[[c for c in ["file_id","generator","augmentation_type","raw_or_augmented","source_audio","split"] if c in inv.columns]].drop_duplicates("file_id")
    for c in ["generator","split","augmentation_type","raw_or_augmented","source_audio"]:
        if c in feat: feat=feat.drop(columns=[c])
    df=meta.merge(feat,on="file_id",how="inner"); fs=cfg.get("feature_selection",{}); cols=feature_columns(df,fs.get("include_prefixes")); X=df[cols].replace([np.inf,-np.inf],np.nan)
    clean_labels=set(str(x).lower() for x in cfg["robustness"].get("clean_labels",["raw","clean","none",""])); aug=df.get("augmentation_type",pd.Series("",index=df.index)).fillna("").astype(str).str.lower(); raw=df.get("raw_or_augmented",pd.Series("",index=df.index)).fillna("").astype(str).str.lower(); clean=aug.isin(clean_labels)&~raw.str.contains("aug")
    tr=clean&(df["split"]=="train"); va=clean&(df["split"]=="validation"); ct=clean&(df["split"]=="test"); mc=cfg["model"]; model=build_and_train(X.loc[tr],df.loc[tr,"generator"],X.loc[va],df.loc[va,"generator"],mc["type"],mc.get("params",{}),80,seed)
    conditions={"clean":ct}; [conditions.setdefault(c,(df["split"]=="test")&(aug==c)) for c in sorted(set(aug[~clean].dropna()))]
    res={}; preds=[]; clean_f1=None
    for name,mask in conditions.items():
        if mask.sum()==0: continue
        p=model.predict(X.loc[mask]); pr=model.predict_proba(X.loc[mask]); y=df.loc[mask,"generator"].astype(str).to_numpy(); m=closed_set_metrics(y,p,pr,model.classes_)
        if name=="clean": clean_f1=m["macro_f1"]
        if clean_f1 is not None: m["delta_macro_f1_vs_clean"]=m["macro_f1"]-clean_f1
        res[name]=m; tmp=df.loc[mask,["file_id","generator"]].copy(); tmp["condition"]=name; tmp["generator_pred"]=p; preds.append(tmp)
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"],seed); record_run_context(run,cfg); json_dump(res,run/"metrics.json"); pd.concat(preds,ignore_index=True).to_parquet(run/"predictions.parquet",index=False); print(run)
if __name__=="__main__": main()
