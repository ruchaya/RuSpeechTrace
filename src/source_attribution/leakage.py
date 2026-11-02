from __future__ import annotations
import pandas as pd

def _non_null_set(s): return set(s.dropna().astype(str))

def leakage_report(df, split_col="split"):
    required={split_col,"file_id","text_hash","source_audio","generator"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing leakage columns: {sorted(missing)}")
    rep={"passed":True,"errors":[]}
    splits=[s for s in ["train","validation","test"] if s in set(df[split_col].astype(str))]
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            ov=_non_null_set(df[df[split_col]==a]["text_hash"]) & _non_null_set(df[df[split_col]==b]["text_hash"])
            if ov:
                rep["passed"]=False; rep["errors"].append({"type":"text_hash_overlap","splits":[a,b],"count":len(ov)})
    if df["file_id"].duplicated().any():
        rep["passed"]=False; rep["errors"].append({"type":"duplicate_file_id","count":int(df["file_id"].duplicated(keep=False).sum())})
    src=df.dropna(subset=["source_audio"])
    if len(src):
        bad=src.groupby("source_audio")[split_col].nunique()
        if (bad>1).any():
            rep["passed"]=False; rep["errors"].append({"type":"source_derivative_overlap","count":int((bad>1).sum())})
    return rep

def assert_no_leakage(df, split_col="split"):
    rep=leakage_report(df,split_col)
    if not rep["passed"]: raise RuntimeError(f"Leakage checks failed: {rep['errors']}")

def assert_unknown_absent(train, validation, unknown_generators):
    unknown=set(map(str,unknown_generators))
    for name,part in [("train",train),("validation",validation)]:
        overlap=set(part["generator"].astype(str))&unknown
        if overlap: raise RuntimeError(f"Unknown generators leaked into {name}: {sorted(overlap)}")
