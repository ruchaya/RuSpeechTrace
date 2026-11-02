import pandas as pd
from source_attribution.inventory import normalize_text,inventory_summary

def test_normalize_text():
    assert normalize_text("  Привет,   МИР! ")=="привет мир"

def test_duplicate_count():
    df=pd.DataFrame({"file_id":["a","a"],"split_eligibility":[True,True],"generator":["g","g"],"path":["x","x"],"duration":[1.,2.],"text_hash":["t1","t2"],"speaker":[None,None],"voice":[None,None]})
    assert inventory_summary(df)["duplicate_file_ids"]==2
