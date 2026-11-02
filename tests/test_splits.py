import pandas as pd

from source_attribution.splits import (
    make_global_splits,
    make_open_set_folds,
    one_row_per_generator_text,
    select_primary_pool,
    select_split_quotas,
)


def toy(n_generators=5, n_texts=100):
    rows = []
    for i in range(n_texts):
        # The same text_hash is deliberately shared across generators.
        for g in range(n_generators):
            rows.append(
                {
                    "file_id": f"g{g}_{i}",
                    "generator": f"g{g}",
                    "text_hash": f"t{i}",
                    "source_audio": None,
                }
            )
    return pd.DataFrame(rows)


def test_deterministic():
    d = toy()
    a = make_global_splits(d, 42)
    b = make_global_splits(d, 42)
    assert a[["file_id", "split"]].equals(b[["file_id", "split"]])


def test_shared_text_never_crosses_splits():
    d = make_global_splits(toy(), 42)
    assert d.groupby("text_hash")["split"].nunique().max() == 1


def test_each_generator_unknown_once():
    d = make_global_splits(toy(), 42)
    f = make_open_set_folds(d, 5, 42)
    u = [
        g
        for x in f["folds"].values()
        for g in x["unknown_generators"]
    ]
    assert sorted(u) == sorted(d["generator"].unique())


def test_string_split_eligibility_is_supported():
    d = toy(n_generators=2, n_texts=20)
    d["label"] = "fake"
    d["group"] = "raw"
    d["raw_or_augmented"] = "raw"
    d["augmentation_type"] = pd.NA
    d["split_eligibility"] = "primary_closed_set_candidate"

    out = select_primary_pool(d, ["g0", "g1"])
    assert len(out) == len(d)


def test_exact_split_quotas():
    d = toy(n_generators=2, n_texts=200)
    d = one_row_per_generator_text(d, seed=42)
    d = make_global_splits(d, 42)

    # Small quotas that are guaranteed by this toy set.
    final = select_split_quotas(
        d,
        quotas={"train": 20, "validation": 5, "test": 5},
        seed=42,
    )
    counts = final.groupby(["split", "generator"]).size()

    for split_name, quota in {
        "train": 20,
        "validation": 5,
        "test": 5,
    }.items():
        for generator in ["g0", "g1"]:
            assert counts.loc[(split_name, generator)] == quota

    assert final.groupby("text_hash")["split"].nunique().max() == 1
