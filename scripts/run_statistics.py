#!/usr/bin/env python
import argparse,json
from pathlib import Path
import numpy as np
from source_attribution.statistics import bootstrap_ci
from source_attribution.utils import json_dump
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results-root",default="results/raw");ap.add_argument("--output",default="results/statistical_tests/summary.json");a=ap.parse_args();exps={}
    for p in Path(a.results_root).rglob("metrics.json"):
        try:d=json.loads(p.read_text(encoding="utf-8"))
        except Exception:continue
        simple={k:v for k,v in d.items() if isinstance(v,(int,float))};exp=p.parent.parent.name
        if simple:exps.setdefault(exp,[]).append(simple)
    out={}
    for exp,runs in exps.items():
        keys=sorted(set.intersection(*(set(r) for r in runs))) if runs else [];out[exp]={}
        for k in keys:
            vals=[r[k] for r in runs if np.isfinite(r[k])]
            if vals:
                mean,lo,hi=bootstrap_ci(vals,5000,.95,42);out[exp][k]={"n_runs":len(vals),"mean":mean,"sd":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,"bootstrap_ci95":[lo,hi]}
    json_dump(out,a.output);print(a.output)
if __name__=="__main__":main()
