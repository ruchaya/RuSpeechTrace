#!/usr/bin/env python
import argparse,subprocess
from pathlib import Path
OFFICIAL="https://github.com/clovaai/aasist.git"
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--destination",default="third_party/aasist");ap.add_argument("--commit",default=None);a=ap.parse_args();d=Path(a.destination)
    if not d.exists():subprocess.check_call(["git","clone",OFFICIAL,str(d)])
    if a.commit:subprocess.check_call(["git","checkout",a.commit],cwd=d)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=d,text=True).strip();print(f"{d} commit={sha}")
if __name__=="__main__":main()
