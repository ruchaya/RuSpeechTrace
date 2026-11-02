#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.utils import json_dump,set_seed

def verified_mapping(path):
    m=pd.read_csv(path); req={"family_label","ruasd_model","mlaad_model","evidence","allowed_attribution_level"}; missing=req-set(m.columns)
    if missing: raise ValueError(f"Mapping missing {sorted(missing)}")
    m=m[m["evidence"].fillna("").astype(str).str.strip().ne("")&m["family_label"].fillna("").astype(str).str.strip().ne("")].copy()
    if len(m)==0: raise RuntimeError("No evidence-backed family mappings. Fill data_inventory/family_mapping_ruasd_mlaad.csv manually.")
    return m

def add_family(df,mapping,dataset):
    d=dict(zip(mapping["ruasd_model" if dataset=="ruasd" else "mlaad_model"].astype(str),mapping["family_label"].astype(str))); x=df.copy(); x["family_target"]=x["generator"].astype(str).map(d); return x[x["family_target"].notna()].copy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args(); cfg=load_config(a.config); seed=int(cfg["experiment"].get("seed",42)); set_seed(seed); inp=cfg["inputs"]; mapping=verified_mapping(inp["mapping"])
    r=pd.read_parquet(inp["ruasd_features"]); rs=pd.read_csv(inp["ruasd_splits"])[["file_id","split","generator"]]
    if "generator" in r:r=r.drop(columns=["generator"])
    if "split" in r:r=r.drop(columns=["split"])
    r=rs.merge(r,on="file_id",how="inner"); m=pd.read_parquet(inp["mlaad_features"])
    if "generator" not in m: raise ValueError("MLAAD features must include generator.")
    r=add_family(r,mapping,"ruasd"); m=add_family(m,mapping,"mlaad"); common=sorted(set(r["family_target"])&set(m["family_target"]))
    if len(common)<2: raise RuntimeError("Need >=2 verified common families.")
    r=r[r["family_target"].isin(common)];m=m[m["family_target"].isin(common)]; fs=cfg.get("feature_selection",{}); cols=[c for c in feature_columns(r,fs.get("include_prefixes")) if c in m.columns and pd.api.types.is_numeric_dtype(m[c])]
    if not cols: raise RuntimeError("No common numeric features.")
    tr=r["split"]=="train";va=r["split"]=="validation";te=r["split"]=="test";mc=cfg["model"]; model=build_and_train(r.loc[tr,cols],r.loc[tr,"family_target"],r.loc[va,cols],r.loc[va,"family_target"],mc["type"],mc.get("params",{}),80,seed)
    res={};preds=[]
    for name,df,mask in [("ruasd_to_ruasd",r,te),("ruasd_to_mlaad",m,pd.Series(True,index=m.index))]:
        p=model.predict(df.loc[mask,cols]);pr=model.predict_proba(df.loc[mask,cols]);y=df.loc[mask,"family_target"].astype(str).to_numpy();res[name]=closed_set_metrics(y,p,pr,model.classes_);tmp=df.loc[mask,[c for c in ["file_id","generator","family_target"] if c in df.columns]].copy();tmp["experiment"]=name;tmp["family_pred"]=p;preds.append(tmp)
    idx=np.arange(len(m)); trm,vam=train_test_split(idx,test_size=.2,random_state=seed,stratify=m["family_target"]); rev=build_and_train(m.iloc[trm][cols],m.iloc[trm]["family_target"],m.iloc[vam][cols],m.iloc[vam]["family_target"],mc["type"],mc.get("params",{}),80,seed)
    p=rev.predict(r.loc[te,cols]);pr=rev.predict_proba(r.loc[te,cols]);y=r.loc[te,"family_target"].astype(str).to_numpy();res["mlaad_to_ruasd"]=closed_set_metrics(y,p,pr,rev.classes_);tmp=r.loc[te,[c for c in ["file_id","generator","family_target"] if c in r.columns]].copy();tmp["experiment"]="mlaad_to_ruasd";tmp["family_pred"]=p;preds.append(tmp)
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"],seed);record_run_context(run,cfg);json_dump({"common_families":common,"metrics":res},run/"metrics.json");pd.concat(preds,ignore_index=True).to_parquet(run/"predictions.parquet",index=False);print(run)
if __name__=="__main__": main()
