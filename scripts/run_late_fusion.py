#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics
from source_attribution.utils import json_dump,set_seed

def prep(path,sp,prefixes=None):
    f=pd.read_parquet(path)
    if "generator" in f:f=f.drop(columns=["generator"])
    if "split" in f:f=f.drop(columns=["split"])
    d=sp[["file_id","split","generator"]].merge(f,on="file_id",how="inner");cols=feature_columns(d,prefixes);return d,cols

def align_probs(model,proba,target_classes):
    idx={str(c):i for i,c in enumerate(model.classes_)};out=np.zeros((len(proba),len(target_classes)))
    for j,c in enumerate(target_classes):out[:,j]=proba[:,idx[str(c)]]
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--features-a",required=True);ap.add_argument("--features-b",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--config-a",required=True);ap.add_argument("--config-b",required=True);ap.add_argument("--output-root",default="results/raw");ap.add_argument("--seed",type=int,default=42);a=ap.parse_args();set_seed(a.seed);sp=pd.read_csv(a.splits);ca=load_config(a.config_a);cb=load_config(a.config_b);da,cola=prep(a.features_a,sp,ca.get("feature_selection",{}).get("include_prefixes"));db,colb=prep(a.features_b,sp,cb.get("feature_selection",{}).get("include_prefixes"));common=sorted(set(da["file_id"])&set(db["file_id"]));da=da[da["file_id"].isin(common)].sort_values("file_id").reset_index(drop=True);db=db[db["file_id"].isin(common)].sort_values("file_id").reset_index(drop=True)
    if not np.array_equal(da["file_id"],db["file_id"]):raise RuntimeError("Feature stores cannot be aligned.")
    tr=da["split"]=="train";va=da["split"]=="validation";te=da["split"]=="test";ma=build_and_train(da.loc[tr,cola],da.loc[tr,"generator"],da.loc[va,cola],da.loc[va,"generator"],ca["model"]["type"],ca["model"].get("params",{}),80,a.seed);mb=build_and_train(db.loc[tr,colb],db.loc[tr,"generator"],db.loc[va,colb],db.loc[va,"generator"],cb["model"]["type"],cb["model"].get("params",{}),80,a.seed)
    classes=np.array(sorted(set(ma.classes_)&set(mb.classes_)));pav=align_probs(ma,ma.predict_proba(da.loc[va,cola]),classes);pbv=align_probs(mb,mb.predict_proba(db.loc[va,colb]),classes);yva=da.loc[va,"generator"].astype(str).to_numpy();grid=np.linspace(0,1,21);best=(-1,None)
    for alpha in grid:
        p=alpha*pav+(1-alpha)*pbv;pred=classes[np.argmax(p,1)];score=closed_set_metrics(yva,pred,p,classes)["macro_f1"]
        if score>best[0]:best=(score,float(alpha))
    alpha=best[1];pat=align_probs(ma,ma.predict_proba(da.loc[te,cola]),classes);pbt=align_probs(mb,mb.predict_proba(db.loc[te,colb]),classes);pt=alpha*pat+(1-alpha)*pbt;pred=classes[np.argmax(pt,1)];yt=da.loc[te,"generator"].astype(str).to_numpy();met=closed_set_metrics(yt,pred,pt,classes);run=create_run_dir(a.output_root,"late_fusion",a.seed);record_run_context(run,{"alpha":alpha,"config_a":ca,"config_b":cb});json_dump({"selected_alpha":alpha,"validation_macro_f1":best[0],"test":met},run/"metrics.json");o=da.loc[te,["file_id","generator"]].copy().rename(columns={"generator":"generator_true"});o["generator_pred"]=pred;o.to_parquet(run/"predictions.parquet",index=False);print(met);print(run)
if __name__=="__main__":main()
