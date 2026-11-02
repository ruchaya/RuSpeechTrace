#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns,create_run_dir,record_run_context
from source_attribution.leakage import assert_no_leakage,assert_unknown_absent
from source_attribution.models import build_and_train
from source_attribution.ood import msp_id_score,energy_id_score,sme_id_score,fit_known_quantile_threshold,class_prototypes,prototype_scores
from source_attribution.metrics import open_set_metrics
from source_attribution.utils import json_load,json_dump,set_seed
from sklearn.metrics import f1_score

def classifier_score(method,model,X,cfg):
    p=model.predict_proba(X); l=model.decision_scores(X)
    if method=="msp": return msp_id_score(p)
    if method=="energy": return energy_id_score(l,float(cfg["ood"].get("temperature_energy",1)))
    if method=="sme": return sme_id_score(l,float(cfg["ood"].get("temperature_sme",4)))
    raise ValueError(method)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--splits",required=True); ap.add_argument("--folds",required=True); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=load_config(a.config); seed=int(cfg["experiment"].get("seed",42)); set_seed(seed); feat=pd.read_parquet(a.features); sp=pd.read_csv(a.splits); assert_no_leakage(sp); folds=json_load(a.folds)
    if "generator" in feat: feat=feat.drop(columns=["generator"])
    if "split" in feat: feat=feat.drop(columns=["split"])
    df=sp.merge(feat,on="file_id",how="inner"); fs=cfg.get("feature_selection",{}); cols=feature_columns(df,fs.get("include_prefixes"),fs.get("exclude_columns"),bool(fs.get("exclude_duration",False))); X=df[cols].replace([np.inf,-np.inf],np.nan)
    results={}; predframes=[]
    for fid,fd in folds["folds"].items():
        known=set(fd["known_generators"]); unknown=set(fd["unknown_generators"]); g=df["generator"].astype(str)
        tr=(df["split"]=="train")&g.isin(known); va=(df["split"]=="validation")&g.isin(known); kt=(df["split"]=="test")&g.isin(known); ut=(df["split"]=="test")&g.isin(unknown)
        assert_unknown_absent(df[tr],df[va],unknown); mc=cfg["model"]; model=build_and_train(X.loc[tr],df.loc[tr,"generator"],X.loc[va],df.loc[va,"generator"],mc["type"],mc.get("params",{}),80,seed)
        kpred_model=model.predict(X.loc[kt]); ktrue=df.loc[kt,"generator"].astype(str).to_numpy(); fres={}
        # Prototype preprocessing fitted on training only.
        imp=SimpleImputer(strategy="median"); sc=StandardScaler()
        Xtrp=sc.fit_transform(imp.fit_transform(X.loc[tr])); Xvap=sc.transform(imp.transform(X.loc[va])); Xktp=sc.transform(imp.transform(X.loc[kt])); Xutp=sc.transform(imp.transform(X.loc[ut]))
        proto_labels,centers=class_prototypes(Xtrp,df.loc[tr,"generator"].astype(str).to_numpy())
        for method in cfg["ood"]["methods"]:
            if method=="prototype":
                vidx,vs,_=prototype_scores(Xvap,centers); kidx,ks,_=prototype_scores(Xktp,centers); uidx,us,_=prototype_scores(Xutp,centers)
                kpred=proto_labels[kidx]; upred=proto_labels[uidx]
            else:
                vs=classifier_score(method,model,X.loc[va],cfg); ks=classifier_score(method,model,X.loc[kt],cfg); us=classifier_score(method,model,X.loc[ut],cfg); kpred=kpred_model; upred=model.predict(X.loc[ut])
            th=fit_known_quantile_threshold(vs,float(cfg["ood"].get("known_reject_quantile",.05))); m=open_set_metrics(ktrue,kpred,ks,us)
            open_true=np.r_[ktrue,np.array(["unknown"]*ut.sum(),dtype=object)]
            open_pred=np.r_[np.where(ks>=th,kpred,"unknown"),np.where(us>=th,upred,"unknown")]
            m.update({"threshold":th,"known_accept_rate":float(np.mean(ks>=th)),"unknown_reject_rate":float(np.mean(us<th)),"macro_f1_with_unknown":float(f1_score(open_true.astype(str),open_pred.astype(str),average="macro"))}); fres[method]=m
            a1=df.loc[kt,["file_id","generator"]].copy(); a1["fold"]=fid;a1["method"]=method;a1["is_known_true"]=True;a1["generator_pred"]=kpred;a1["id_score"]=ks;a1["accepted_known"]=ks>=th
            a2=df.loc[ut,["file_id","generator"]].copy(); a2["fold"]=fid;a2["method"]=method;a2["is_known_true"]=False;a2["generator_pred"]=upred;a2["id_score"]=us;a2["accepted_known"]=us>=th; predframes += [a1,a2]
        results[fid]={"known_generators":sorted(known),"unknown_generators":sorted(unknown),"methods":fres}
    run=create_run_dir(cfg["experiment"].get("output_root","results/raw"),cfg["experiment"]["name"],seed); record_run_context(run,cfg); json_dump(results,run/"metrics.json"); pd.concat(predframes,ignore_index=True).to_parquet(run/"predictions.parquet",index=False); print(run)
if __name__=="__main__": main()
