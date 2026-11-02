#!/usr/bin/env python
import subprocess,sys
if __name__=="__main__":
    raise SystemExit(subprocess.call([sys.executable,"scripts/extract_acoustic_features.py","--inventory","data_inventory/mlaad_inventory.csv","--output","features/mlaad_acoustic.parquet"]))
