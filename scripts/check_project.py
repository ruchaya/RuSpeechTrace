#!/usr/bin/env python
from pathlib import Path
import subprocess,sys
def main():
    required=["README.md","CODEX_TASK.md","configs/closed_set_acoustic.yaml","configs/open_set.yaml","src/source_attribution","scripts","tests"]
    missing=[p for p in required if not Path(p).exists()]
    if missing:raise SystemExit(f"Missing required project items: {missing}")
    rc=subprocess.call([sys.executable,"-m","pytest","-q"])
    raise SystemExit(rc)
if __name__=="__main__":main()
