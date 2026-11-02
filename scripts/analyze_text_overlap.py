#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze cross-generator text reuse for the strict RuASD primary pool."
    )
    p.add_argument(
        "--inventory",
        default="data_inventory/ruasd_inventory.csv",
        help="Final RuASD inventory CSV.",
    )
    p.add_argument(
        "--generator-counts",
        default="data_inventory/ruasd_generator_counts.csv",
        help="Per-generator audit table.",
    )
    p.add_argument(
        "--summary",
        default="data_inventory/text_overlap_summary.json",
        help="Summary JSON output.",
    )
    p.add_argument(
        "--generator-summary",
        default="data_inventory/text_overlap_by_generator.csv",
        help="Per-generator overlap CSV output.",
    )
    p.add_argument(
        "--hash-summary",
        default="data_inventory/text_overlap_hashes.csv",
        help="Shared-text-hash CSV output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    inv = pd.read_csv(args.inventory, low_memory=False)
    counts = pd.read_csv(args.generator_counts, low_memory=False)

    # Strict primary eligibility: at least 2000 available text hashes AND
    # at least 2000 distinct texts. This is derived from the audit, not hard-coded.
    eligible = counts[
        (counts["n_with_text"] >= 2000)
        & (counts["n_unique_texts"] >= 2000)
    ]["subset"].astype(str).tolist()

    x = inv[
        inv["label"].astype(str).str.lower().eq("fake")
        & inv["group"].astype(str).str.lower().eq("raw")
        & inv["generator"].astype(str).isin(eligible)
        & inv["text_hash"].notna()
    ][["file_id", "generator", "text_hash"]].copy()

    # Per hash: how many rows and how many generators use the same normalized text?
    h = (
        x.groupby("text_hash")
        .agg(
            n_rows=("file_id", "size"),
            n_generators=("generator", "nunique"),
            generators=("generator", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )

    shared = h[h["n_generators"] > 1].copy()
    shared = shared.sort_values(["n_generators", "n_rows"], ascending=False)

    shared_hashes = set(shared["text_hash"])
    x["shared_across_generators"] = x["text_hash"].isin(shared_hashes)

    by_gen = (
        x.groupby("generator")
        .agg(
            n_rows=("file_id", "size"),
            n_unique_texts=("text_hash", "nunique"),
            n_rows_on_shared_texts=("shared_across_generators", "sum"),
            n_shared_unique_texts=(
                "text_hash",
                lambda s: s[s.isin(shared_hashes)].nunique(),
            ),
        )
        .reset_index()
    )
    by_gen["shared_rows_pct"] = (
        100.0 * by_gen["n_rows_on_shared_texts"] / by_gen["n_rows"]
    ).round(2)
    by_gen = by_gen.sort_values("generator")

    distribution = (
        h["n_generators"].value_counts().sort_index().to_dict()
    )

    summary = {
        "primary_generator_count": int(len(eligible)),
        "primary_generators": sorted(eligible),
        "rows_with_text_in_primary_pool": int(len(x)),
        "unique_text_hashes_in_primary_pool": int(h["text_hash"].nunique()),
        "shared_text_hashes_across_generators": int(len(shared)),
        "rows_belonging_to_shared_text_hashes": int(
            x["shared_across_generators"].sum()
        ),
        "rows_belonging_to_shared_text_hashes_pct": round(
            100.0 * x["shared_across_generators"].mean(), 2
        ),
        "max_generators_sharing_one_text": int(h["n_generators"].max()),
        "text_hash_count_by_number_of_generators": {
            str(int(k)): int(v) for k, v in distribution.items()
        },
    }

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    by_gen.to_csv(args.generator_summary, index=False, encoding="utf-8-sig")
    shared.to_csv(args.hash_summary, index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPer-generator: {args.generator_summary}")
    print(f"Shared hashes:  {args.hash_summary}")
    print(f"Summary:        {args.summary}")


if __name__ == "__main__":
    main()
