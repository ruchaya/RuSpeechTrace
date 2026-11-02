#!/usr/bin/env python
import argparse,subprocess,sys,tempfile,yaml
from copy import deepcopy
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--features",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--config",required=True);a=ap.parse_args();base=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for name,exclude in [("with_duration",False),("without_duration",True)]:
        cfg=deepcopy(base);cfg["experiment"]["name"]=base["experiment"]["name"]+"_ablation_"+name;cfg.setdefault("feature_selection",{})["exclude_duration"]=exclude
        with tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False,encoding="utf-8") as f:yaml.safe_dump(cfg,f,sort_keys=False);tmp=f.name
        try:subprocess.check_call([sys.executable,"scripts/train_closed_set.py","--features",a.features,"--splits",a.splits,"--config",tmp])
        finally:Path(tmp).unlink(missing_ok=True)
    print("Duration ablation completed. For peak-normalization, re-extract features with --peak-normalize and --no-peak-normalize.")
if __name__=="__main__":main()
