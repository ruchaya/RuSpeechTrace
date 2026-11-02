#!/usr/bin/env python
from pathlib import Path
import json
from source_attribution.utils import environment_info
def main():
    exps={}
    for p in Path("results/raw").rglob("metrics.json"):
        exps[p.parent.parent.name]=exps.get(p.parent.parent.name,0)+1
    lines=["# Run report","","## Environment","```json",json.dumps(environment_info(),ensure_ascii=False,indent=2),"```","","## Experiments actually executed"]+[f"- {k}: {v} run(s)" for k,v in sorted(exps.items())] if exps else ["# Run report","","## Environment","```json",json.dumps(environment_info(),ensure_ascii=False,indent=2),"```","","## Experiments actually executed","- None"]
    lines += ["","## Verified numerical results","Read each experiment's `metrics.json`; no values are fabricated here.","","## Blockers","Review missing-input exceptions and logs.","","## Exact commands to continue","See README.md and Makefile."]
    Path("RUN_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print("RUN_REPORT.md")
if __name__=="__main__":main()
