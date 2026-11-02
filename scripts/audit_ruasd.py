#!/usr/bin/env python
import argparse
from pathlib import Path
from source_attribution.inventory import audit_dataset,inventory_summary
from source_attribution.io import write_table
from source_attribution.utils import json_dump

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",required=True); ap.add_argument("--output",default="data_inventory/ruasd_inventory.csv"); a=ap.parse_args()
    inv=audit_dataset(a.dataset_root); write_table(inv,a.output); s=inventory_summary(inv); out=Path(a.output); json_dump(s,out.with_name("inventory_summary.json"))
    if s["duplicate_file_ids"]:
        inv[inv["file_id"].duplicated(keep=False)].sort_values("file_id").to_csv(out.with_name("duplicate_file_ids.csv"),index=False)
        raise RuntimeError(f"Duplicate file IDs detected: {s['duplicate_file_ids']} rows.")
    print(f"Rows={s['rows']}; eligible={s['eligible_rows']}; generators={inv['generator'].nunique(dropna=True)}")
    print(a.output)

if __name__=="__main__": main()
