import pandas as pd,pytest
from source_attribution.leakage import leakage_report,assert_unknown_absent

def test_text_overlap():
    d=pd.DataFrame({"file_id":["a","b"],"text_hash":["x","x"],"source_audio":[None,None],"generator":["g","g"],"split":["train","test"]})
    assert not leakage_report(d)["passed"]

def test_unknown_leak():
    with pytest.raises(RuntimeError):
        assert_unknown_absent(pd.DataFrame({"generator":["a","b"]}),pd.DataFrame({"generator":["a"]}),["b"])
