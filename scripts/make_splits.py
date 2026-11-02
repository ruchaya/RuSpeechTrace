#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from source_attribution.leakage import leakage_report
from source_attribution.splits import (
    make_global_splits,
    make_open_set_folds,
    one_row_per_generator_text,
    select_primary_pool,
    select_split_quotas,
)
from source_attribution.utils import json_dump


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inventory",
        default="data_inventory/ruasd_inventory.csv",
    )
    ap.add_argument(
        "--config",
        default="configs/data.yaml",
    )
    ap.add_argument(
        "--output",
        default="splits/closed_set.csv",
    )
    ap.add_argument(
        "--fold-output",
        default="splits/open_set_folds.json",
    )
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    with open(a.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    selection = cfg["selection"]
    split_cfg = cfg["split"]

    primary_generators = selection["primary_generators"]
    quotas = selection["per_split_quota"]
    seed = int(split_cfg["seed"])
    n_folds = int(split_cfg.get("open_set_folds", 5))

    inv = pd.read_csv(a.inventory, low_memory=False)

    if inv["file_id"].duplicated().any():
        raise RuntimeError("Inventory has duplicate file IDs.")

    pool = select_primary_pool(
        inv,
        primary_generators=primary_generators,
        raw_only=bool(selection.get("raw_only", True)),
    )
    unique_pool = one_row_per_generator_text(pool, seed=seed)

    assigned = make_global_splits(unique_pool, seed=seed)
    final = select_split_quotas(assigned, quotas=quotas, seed=seed)

    output_path = Path(a.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False, encoding="utf-8-sig")

    rep = leakage_report(final)
    rep.update(
        {
            "primary_generators": sorted(primary_generators),
            "n_primary_generators": len(primary_generators),
            "candidate_rows_before_generator_text_dedup": int(len(pool)),
            "candidate_rows_after_generator_text_dedup": int(len(unique_pool)),
            "final_rows": int(len(final)),
            "per_split_quota": {k: int(v) for k, v in quotas.items()},
            "split_counts": {
                str(k): int(v)
                for k, v in final["split"].value_counts().items()
            },
        }
    )
    json_dump(rep, output_path.with_name("leakage_report.json"))

    if not rep["passed"]:
        raise RuntimeError(rep)

    folds = make_open_set_folds(
        final,
        n_folds=n_folds,
        seed=seed,
    )
    json_dump(folds, a.fold_output)

    print("\nFinal split counts:")
    print(final["split"].value_counts().to_string())

    print("\nPer-generator counts by split:")
    print(
        final.groupby(["generator", "split"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nUnique text hashes by split:")
    print(final.groupby("split")["text_hash"].nunique().to_string())

    print("\nLeakage passed:", rep["passed"])
    print("Closed set:", a.output)
    print("Leakage:  ", output_path.with_name("leakage_report.json"))
    print("Open set: ", a.fold_output)


if __name__ == "__main__":
    main()
