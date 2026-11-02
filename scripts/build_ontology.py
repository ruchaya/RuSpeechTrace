#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--inventory",required=True); ap.add_argument("--output",default="data_inventory/generator_ontology.csv"); a=ap.parse_args()
    inv=pd.read_csv(a.inventory); gens=sorted(inv["generator"].dropna().astype(str).unique())
    rows=[{"ruasd_label":g,"generator_name":g,"base_model_family":"","checkpoint_or_voice":"","task_type":"unknown","vocoder":"unknown","open_source_or_commercial":"unknown","family_label_for_transfer":"","evidence":"","notes":"VERIFY MANUALLY; do not infer architecture from label."} for g in gens]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False)
    print(f"Ontology skeleton: {len(rows)} generators. Manual evidence verification required.")
if __name__=="__main__": main()
