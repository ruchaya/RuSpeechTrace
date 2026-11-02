#!/usr/bin/env python
import argparse,subprocess,sys,tempfile,yaml
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["closed","open"],required=True);ap.add_argument("--config",required=True);ap.add_argument("--features",required=True);ap.add_argument("--splits",required=True);ap.add_argument("--folds",default="splits/open_set_folds.json");ap.add_argument("--seeds",default="42,123,456,789,2026");a=ap.parse_args();base=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for seed in [int(x) for x in a.seeds.split(",")]:
        cfg=dict(base);cfg["experiment"]=dict(base["experiment"]);cfg["experiment"]["seed"]=seed
        with tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False,encoding="utf-8") as f:yaml.safe_dump(cfg,f,sort_keys=False);tmp=f.name
        try:
            cmd=[sys.executable,"scripts/train_closed_set.py","--features",a.features,"--splits",a.splits,"--config",tmp] if a.mode=="closed" else [sys.executable,"scripts/run_open_set.py","--features",a.features,"--splits",a.splits,"--folds",a.folds,"--config",tmp]
            subprocess.check_call(cmd)
        finally:Path(tmp).unlink(missing_ok=True)
if __name__=="__main__":main()
