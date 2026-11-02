from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .leakage import leakage_report

def is_raw_mask(df):
    vals=df.get("raw_or_augmented",pd.Series("",index=df.index)).fillna("").astype(str).str.lower().str.strip()
    aug=df.get("augmentation_type",pd.Series("",index=df.index)).fillna("").astype(str).str.lower().str.strip()
    return ~(vals.str.contains("aug")|vals.str.contains("distort")|~aug.isin(["","none","raw","clean","nan"]))

def select_balanced_subset(inv,max_per_generator=2000,min_per_generator=1,raw_only=True,seed=42):
    df=inv.copy()
    if "split_eligibility" in df: df=df[df["split_eligibility"].fillna(False)]
    if raw_only: df=df[is_raw_mask(df)]
    df=df[df["generator"].notna()].copy()
    counts=df["generator"].value_counts()
    df=df[df["generator"].isin(counts[counts>=min_per_generator].index)]
    parts=[]
    for _,g in df.groupby("generator",sort=True):
        parts.append(g.sample(n=min(max_per_generator,len(g)),random_state=seed))
    if not parts: raise RuntimeError("No generator satisfies selection criteria.")
    return pd.concat(parts,ignore_index=True)

def _groups(df):
    group=df["text_hash"].astype("string")
    empty=group.isna()|(group.str.len()==0)
    source=df.get("source_audio",pd.Series(pd.NA,index=df.index)).astype("string")
    fallback=source.where(source.notna(),df["file_id"].astype("string"))
    return group.where(~empty,fallback).astype(str)

def make_global_splits(df,seed=42):
    work=df.copy().reset_index(drop=True)
    y=work["generator"].astype(str).to_numpy(); groups=_groups(work).to_numpy()
    sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    tv_idx,test_idx=next(sg.split(work,y=y,groups=groups))
    tv=work.iloc[tv_idx].copy()
    sg2=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed+1)
    tr_rel,val_rel=next(sg2.split(tv,y=tv["generator"].astype(str),groups=_groups(tv)))
    split=np.empty(len(work),dtype=object)
    split[tv_idx[tr_rel]]="train"; split[tv_idx[val_rel]]="validation"; split[test_idx]="test"
    work["split"]=split
    rep=leakage_report(work)
    if not rep["passed"]: raise RuntimeError(rep)
    return work

def make_open_set_folds(split_df,n_folds=5,seed=42):
    generators=sorted(split_df["generator"].dropna().astype(str).unique())
    rng=np.random.default_rng(seed); arr=np.array(generators,dtype=object); rng.shuffle(arr)
    chunks=np.array_split(arr,n_folds); allg=set(generators); folds={}
    for i,ch in enumerate(chunks,1):
        unknown=sorted(map(str,ch.tolist()))
        folds[str(i)]={"known_generators":sorted(allg-set(unknown)),"unknown_generators":unknown,
                       "rule":"Unknown evaluation uses globally held-out test rows only."}
    return {"n_folds":n_folds,"seed":seed,"folds":folds}
