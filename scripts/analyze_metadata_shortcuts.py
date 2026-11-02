#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose metadata shortcuts in the frozen RuASD closed-set subset."
    )
    p.add_argument(
        "--features",
        default="features/ruasd_closed_set_acoustic.parquet",
    )
    p.add_argument(
        "--splits",
        default="splits/closed_set.csv",
    )
    p.add_argument(
        "--out-dir",
        default="results/summaries/metadata_shortcut_diagnostics",
    )
    return p.parse_args()


def dominant_value_and_share(s: pd.Series):
    vc = s.value_counts(dropna=False)
    if vc.empty:
        return np.nan, np.nan
    return vc.index[0], float(vc.iloc[0] / vc.sum())


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features)
    split_path = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(
        feature_path,
        columns=["file_id", "duration", "sample_rate", "channels"],
    )
    sp = pd.read_csv(
        split_path,
        low_memory=False,
        usecols=["file_id", "generator", "split"],
    )

    df = sp.merge(meta, on="file_id", how="left", validate="one_to_one")
    if len(df) != 68000:
        raise RuntimeError(f"Expected 68000 rows, got {len(df)}")

    rows = []
    for gen, g in df.groupby("generator", sort=True):
        dom_sr, dom_sr_share = dominant_value_and_share(g["sample_rate"])
        dom_ch, dom_ch_share = dominant_value_and_share(g["channels"])
        rows.append(
            {
                "generator": gen,
                "n": len(g),
                "n_sample_rates": int(g["sample_rate"].nunique(dropna=True)),
                "dominant_sample_rate": dom_sr,
                "dominant_sample_rate_share": dom_sr_share,
                "n_channels_values": int(g["channels"].nunique(dropna=True)),
                "dominant_channels": dom_ch,
                "dominant_channels_share": dom_ch_share,
                "duration_min": float(g["duration"].min()),
                "duration_q25": float(g["duration"].quantile(0.25)),
                "duration_median": float(g["duration"].median()),
                "duration_q75": float(g["duration"].quantile(0.75)),
                "duration_max": float(g["duration"].max()),
                "duration_mean": float(g["duration"].mean()),
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(
        out_dir / "generator_metadata_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sr_counts = pd.crosstab(df["generator"], df["sample_rate"])
    sr_counts.to_csv(
        out_dir / "generator_x_sample_rate_counts.csv",
        encoding="utf-8-sig",
    )
    sr_props = sr_counts.div(sr_counts.sum(axis=1), axis=0)
    sr_props.to_csv(
        out_dir / "generator_x_sample_rate_row_proportions.csv",
        encoding="utf-8-sig",
    )

    sr_summary_rows = []
    for sr, g in df.groupby("sample_rate", sort=True):
        split_counts = (
            g.groupby(["generator", "split"])
            .size()
            .unstack(fill_value=0)
        )
        for required_col in ["train", "validation", "test"]:
            if required_col not in split_counts.columns:
                split_counts[required_col] = 0

        full_quota_mask = (
            (split_counts["train"] >= 1300)
            & (split_counts["validation"] >= 300)
            & (split_counts["test"] >= 400)
        )
        full_quota_generators = sorted(
            split_counts.index[full_quota_mask].astype(str).tolist()
        )

        sr_summary_rows.append(
            {
                "sample_rate": sr,
                "n_rows": int(len(g)),
                "n_generators_any": int(g["generator"].nunique()),
                "n_generators_full_1300_300_400": len(full_quota_generators),
                "generators_full_1300_300_400": ";".join(full_quota_generators),
            }
        )

    sr_summary = pd.DataFrame(sr_summary_rows).sort_values(
        ["n_generators_full_1300_300_400", "n_generators_any", "n_rows"],
        ascending=False,
    )
    sr_summary.to_csv(
        out_dir / "sample_rate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quota = (
        df.groupby(["sample_rate", "generator", "split"])
        .size()
        .rename("n")
        .reset_index()
    )
    quota.to_csv(
        out_dir / "sample_rate_generator_split_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    chance_top1 = 1.0 / df["generator"].nunique()

    print("\n=== Metadata shortcut diagnostics ===")
    print(f"Rows: {len(df)}")
    print(f"Generators: {df['generator'].nunique()}")
    print(f"Chance Top-1 / balanced-accuracy reference: {chance_top1:.6f}")
    print(f"Unique native sample rates: {df['sample_rate'].nunique()}")
    print(f"Unique channel values: {df['channels'].nunique()}")

    single_sr = int((profile["n_sample_rates"] == 1).sum())
    print(
        f"Generators using exactly one native sample rate: "
        f"{single_sr}/{len(profile)}"
    )

    print("\nSample-rate candidates for a strict same-SR control:")
    cols = [
        "sample_rate",
        "n_rows",
        "n_generators_any",
        "n_generators_full_1300_300_400",
    ]
    print(sr_summary[cols].to_string(index=False))

    print("\nGenerators with the strongest single-SR concentration:")
    show = profile[
        [
            "generator",
            "dominant_sample_rate",
            "dominant_sample_rate_share",
            "duration_median",
        ]
    ].sort_values(
        ["dominant_sample_rate_share", "generator"],
        ascending=[False, True],
    )
    print(show.to_string(index=False))

    print(f"\nSaved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
