#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from source_attribution.splits import select_balanced_subset,make_global_splits,make_open_set_folds
from source_attribution.leakage import leakage_report
from source_attribution.utils import json_dump

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--inventory",required=True); ap.add_argument("--output",default="splits/closed_set.csv"); ap.add_argument("--fold-output",default="splits/open_set_folds.json")
    ap.add_argument("--max-per-generator",type=int,default=2000); ap.add_argument("--min-per-generator",type=int,default=50); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--include-augmented",action="store_true"); a=ap.parse_args()
    inv=pd.read_csv(a.inventory)
    if inv["file_id"].duplicated().any(): raise RuntimeError("Inventory has duplicate file IDs.")
    subset=select_balanced_subset(inv,a.max_per_generator,a.min_per_generator,not a.include_augmented,a.seed)
    df=make_global_splits(subset,a.seed); Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.output,index=False)
    rep=leakage_report(df); json_dump(rep,Path(a.output).with_name("leakage_report.json"))
    if not rep["passed"]: raise RuntimeError(rep)
    json_dump(make_open_set_folds(df,5,a.seed),a.fold_output)
    print(df["split"].value_counts()); print(df.groupby("split")["generator"].nunique()); print(a.output,a.fold_output)

if __name__=="__main__": main()
