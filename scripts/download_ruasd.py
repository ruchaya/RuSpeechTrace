#!/usr/bin/env python
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-id",default="lab260/RuASD"); ap.add_argument("--output-dir",required=True); ap.add_argument("--revision",default=None)
    a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    print(snapshot_download(repo_id=a.repo_id,repo_type="dataset",revision=a.revision,local_dir=a.output_dir))

if __name__=="__main__": main()
