#!/usr/bin/env python
import argparse,json
from pathlib import Path
import pandas as pd
def flat(d,p=""):
    out={}
    for k,v in d.items():
        key=f"{p}.{k}" if p else str(k)
        if isinstance(v,dict):out.update(flat(v,key))
        elif isinstance(v,(int,float,str,bool)) or v is None:out[key]=v
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results-root",default="results/raw");ap.add_argument("--output-dir",default="results/tables");a=ap.parse_args();rows=[]
    for p in Path(a.results_root).rglob("metrics.json"):
        try:d=json.loads(p.read_text(encoding="utf-8"))
        except Exception:continue
        r={"experiment":p.parent.parent.name,"run_id":p.parent.name};r.update(flat(d));rows.append(r)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if rows:
        df=pd.DataFrame(rows);df.to_csv(out/"all_metrics.csv",index=False);df.to_excel(out/"all_metrics.xlsx",index=False)
    print(out)
if __name__=="__main__":main()
