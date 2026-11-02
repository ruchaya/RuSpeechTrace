#!/usr/bin/env python
import argparse
from pathlib import Path
from source_attribution.inventory import audit_dataset,inventory_summary
from source_attribution.io import write_table
from source_attribution.utils import json_dump
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",required=True); ap.add_argument("--output",default="data_inventory/mlaad_inventory.csv"); a=ap.parse_args()
    inv=audit_dataset(a.dataset_root); write_table(inv,a.output); json_dump(inventory_summary(inv),Path(a.output).with_name("mlaad_inventory_summary.json")); print(a.output)
if __name__=="__main__": main()
