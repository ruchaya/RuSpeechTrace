#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a conservative RuASD generator ontology skeleton from the final inventory."
    )
    p.add_argument(
        "--inventory",
        default="data_inventory/ruasd_inventory.csv",
        help="Final RuASD inventory CSV.",
    )
    p.add_argument(
        "--output",
        default="data_inventory/generator_ontology.csv",
        help="Ontology CSV to create/update.",
    )
    return p.parse_args()


def compact_top_values(s: pd.Series, limit: int = 5) -> str:
    vals = (
        s.astype("string")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(limit)
    )
    if vals.empty:
        return ""
    return json.dumps(
        [{"value": str(k), "count": int(v)} for k, v in vals.items()],
        ensure_ascii=False,
    )


def main() -> None:
    args = parse_args()
    inventory_path = Path(args.inventory)
    output_path = Path(args.output)

    x = pd.read_csv(inventory_path, low_memory=False)

    required = {
        "file_id", "label", "group", "generator", "duration",
        "sample_rate", "channels", "text_hash", "model_raw"
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    rf = x[
        x["label"].astype(str).str.lower().eq("fake")
        & x["group"].astype(str).str.lower().eq("raw")
    ].copy()

    if rf["generator"].isna().any():
        raise SystemExit("Raw fake rows with missing generator found.")

    rf["duration"] = pd.to_numeric(rf["duration"], errors="coerce")

    rows = []
    for generator, g in rf.groupby("generator", sort=True):
        model_clean = (
            g["model_raw"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )

        rows.append({
            "ruasd_label": str(generator),
            "generator_name": str(generator),
            "base_model_family": "unknown",
            "checkpoint_or_voice": "unknown",
            "task_type": "unknown",
            "vocoder": "unknown",
            "open_source_or_commercial": "unknown",
            "family_label_for_transfer": "unknown",
            "evidence": "",
            "notes": "",
            "n_raw": int(len(g)),
            "total_hours": round(float(g["duration"].sum() / 3600.0), 6),
            "n_with_text_hash": int(g["text_hash"].notna().sum()),
            "n_unique_texts": int(g["text_hash"].nunique(dropna=True)),
            "sample_rates_hz": ",".join(
                map(str, sorted(g["sample_rate"].dropna().astype(int).unique().tolist()))
            ),
            "channels": ",".join(
                map(str, sorted(g["channels"].dropna().astype(int).unique().tolist()))
            ),
            "observed_model_nonempty": int(model_clean.shape[0]),
            "observed_model_nunique": int(model_clean.nunique()),
            "observed_model_top_values": compact_top_values(g["model_raw"]),
            "ontology_status": "needs_evidence",
        })

    out = pd.DataFrame(rows).sort_values("ruasd_label").reset_index(drop=True)

    if len(out) != 37:
        print(f"WARNING: expected 37 raw-fake generator labels, found {len(out)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Created: {output_path}")
    print(f"Rows: {len(out)}")
    print("\nGenerators:")
    print(out[[
        "ruasd_label", "n_raw", "total_hours", "n_with_text_hash",
        "sample_rates_hz", "observed_model_nunique"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
