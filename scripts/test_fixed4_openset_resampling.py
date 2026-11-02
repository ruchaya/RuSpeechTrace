import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fixed4_openset_resampled import stratum_sample, short_hash

def run():
    tr = pd.DataFrame({"file_id":[str(i) for i in range(100)], "generator": ["b"]*50+["a"]*50, "split":["train"]*100})
    t42 = stratum_sample(tr, .8, 42)
    assert len(t42) == 80
    assert sum(x in t42 for x in tr[tr.generator == "a"].file_id) == 40
    assert sum(x in t42 for x in tr[tr.generator == "b"].file_id) == 40
    assert t42 == stratum_sample(tr.sample(frac=1, random_state=7), .8, 42)
    assert t42 != stratum_sample(tr, .8, 123)
    assert short_hash(t42) == short_hash(sorted(t42))
    print("PASS: deterministic generator-stratified sampling, 80%, paired train IDs")
if __name__ == "__main__":
    run()
