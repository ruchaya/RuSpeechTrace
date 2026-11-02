#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from source_attribution.plotting import save_confusion,save_score_distribution
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results-root",default="results/raw");ap.add_argument("--output-dir",default="figures");a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for p in Path(a.results_root).rglob("predictions.parquet"):
        try:df=pd.read_parquet(p)
        except Exception:continue
        tag=f"{p.parent.parent.name}_{p.parent.name}"
        if {"generator_true","generator_pred"}<=set(df.columns):save_confusion(df["generator_true"].astype(str),df["generator_pred"].astype(str),out/f"{tag}_confusion.png")
        if {"is_known_true","id_score"}<=set(df.columns):
            k=df[df["is_known_true"].astype(bool)]["id_score"].to_numpy();u=df[~df["is_known_true"].astype(bool)]["id_score"].to_numpy()
            if len(k) and len(u):save_score_distribution(k,u,out/f"{tag}_ood_scores.png")
    print(out)
if __name__=="__main__":main()
