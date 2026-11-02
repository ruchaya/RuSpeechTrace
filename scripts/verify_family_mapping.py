#!/usr/bin/env python
import argparse,pandas as pd
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mapping",default="data_inventory/family_mapping_ruasd_mlaad.csv");a=ap.parse_args();m=pd.read_csv(a.mapping);req=["family_label","ruasd_model","mlaad_model","evidence","allowed_attribution_level"];missing=[c for c in req if c not in m]
    if missing:raise RuntimeError(f"Missing columns: {missing}")
    bad=m[(m["family_label"].fillna("").astype(str).str.strip()=="")|(m["ruasd_model"].fillna("").astype(str).str.strip()=="")|(m["mlaad_model"].fillna("").astype(str).str.strip()=="")|(m["evidence"].fillna("").astype(str).str.strip()=="")]
    if len(bad):raise RuntimeError(f"{len(bad)} mapping rows are incomplete or lack evidence.")
    print(f"Verified structural completeness of {len(m)} mapping rows. This script does not independently validate external evidence.")
if __name__=="__main__":main()
