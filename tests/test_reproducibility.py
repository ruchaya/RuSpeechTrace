from source_attribution.utils import stable_hash_text
def test_hash():
    assert stable_hash_text("abc")==stable_hash_text("abc")
    assert stable_hash_text("abc")!=stable_hash_text("abd")
