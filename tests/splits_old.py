import pandas as pd
from source_attribution.splits import make_global_splits,make_open_set_folds

def toy():
    return pd.DataFrame([{"file_id":f"g{g}_{i}","generator":f"g{g}","text_hash":f"t{g}_{i}","source_audio":None} for g in range(5) for i in range(50)])

def test_deterministic():
    a=make_global_splits(toy(),42);b=make_global_splits(toy(),42)
    assert a[["file_id","split"]].equals(b[["file_id","split"]])

def test_each_generator_unknown_once():
    d=make_global_splits(toy(),42);f=make_open_set_folds(d,5,42);u=[g for x in f["folds"].values() for g in x["unknown_generators"]]
    assert sorted(u)==sorted(d["generator"].unique())
