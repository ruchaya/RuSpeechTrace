from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from source_attribution.leakage import assert_no_leakage
from source_attribution.utils import file_sha256


FEATURES = Path("features/ruasd_closed_set_acoustic.parquet")
SPLITS = Path("splits/closed_set.csv")
OUT = Path("results/summaries/duration_overlap_diagnostics")

EXPECTED_FEATURE_SHA = (
    "a21d537c5133a7e6daa852138f96ee7f8bf1491036cbf940da6451f688adaad8"
)
EXPECTED_SPLIT_SHA = (
    "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0"
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    feature_sha = file_sha256(FEATURES)
    split_sha = file_sha256(SPLITS)

    if feature_sha != EXPECTED_FEATURE_SHA:
        raise RuntimeError(f"Feature store changed: {feature_sha}")

    if split_sha != EXPECTED_SPLIT_SHA:
        raise RuntimeError(f"Frozen split changed: {split_sha}")

    splits = pd.read_csv(
    SPLITS,
    low_memory=False,
    usecols=[
        "file_id",
        "generator",
        "split",
        "text_hash",
        "source_audio",
    ],
)

    assert_no_leakage(splits)

    features = pd.read_parquet(
        FEATURES,
        columns=[
            "file_id",
            "prosody_duration_sec",
        ],
    )

    df = splits.merge(
        features,
        on="file_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != 68000:
        raise RuntimeError(
            f"Expected 68000 rows, got {len(df)}"
        )

    if df["prosody_duration_sec"].isna().any():
        raise RuntimeError(
            "Missing prosody_duration_sec"
        )

    train = df[
        df["split"].eq("train")
    ].copy()

    validation = df[
        df["split"].eq("validation")
    ].copy()

    generators = sorted(
        df["generator"].unique()
    )

    quantiles = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.25,
        0.50,
        0.75,
        0.80,
        0.90,
        0.95,
        0.99,
    ]

    profile_rows = []

    for generator, g in train.groupby(
        "generator",
        sort=True,
    ):
        duration = g[
            "prosody_duration_sec"
        ].astype(float)

        row = {
            "generator": generator,
            "n_train": len(g),
            "min": duration.min(),
            "mean": duration.mean(),
            "max": duration.max(),
        }

        for q in quantiles:
            row[
                f"q{int(q * 100):02d}"
            ] = duration.quantile(q)

        profile_rows.append(row)

    profile = pd.DataFrame(
        profile_rows
    )

    profile.to_csv(
        OUT / "duration_profile_train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidates = [
        ("q01_q99", 0.01, 0.99),
        ("q05_q95", 0.05, 0.95),
        ("q10_q90", 0.10, 0.90),
        ("q20_q80", 0.20, 0.80),
    ]

    summary_rows = []
    count_rows = []

    for name, q_low, q_high in candidates:

        low_column = (
            f"q{int(q_low * 100):02d}"
        )

        high_column = (
            f"q{int(q_high * 100):02d}"
        )

        # Общая нижняя граница:
        # самый высокий lower quantile
        # среди всех generators.
        lower = float(
            profile[low_column].max()
        )

        # Общая верхняя граница:
        # самый низкий upper quantile
        # среди всех generators.
        upper = float(
            profile[high_column].min()
        )

        valid = lower <= upper

        result = {
            "candidate": name,
            "defined_from": "train_only",
            "lower_sec": lower,
            "upper_sec": upper,
            "width_sec": max(
                0.0,
                upper - lower,
            ),
            "valid_interval": valid,
        }

        for split_name, split_df in [
            ("train", train),
            ("validation", validation),
        ]:

            if valid:
                kept = split_df[
                    split_df[
                        "prosody_duration_sec"
                    ].between(
                        lower,
                        upper,
                        inclusive="both",
                    )
                ]

                counts = (
                    kept
                    .groupby("generator")
                    .size()
                    .reindex(
                        generators,
                        fill_value=0,
                    )
                )

            else:
                counts = pd.Series(
                    0,
                    index=generators,
                    dtype=int,
                )

            result[
                f"min_per_generator_{split_name}"
            ] = int(counts.min())

            result[
                f"median_per_generator_{split_name}"
            ] = float(counts.median())

            result[
                f"max_per_generator_{split_name}"
            ] = int(counts.max())

            result[
                f"balanced_total_{split_name}"
            ] = int(
                counts.min()
                * len(generators)
            )

            for generator, n in counts.items():
                count_rows.append(
                    {
                        "candidate": name,
                        "split": split_name,
                        "generator": generator,
                        "lower_sec": lower,
                        "upper_sec": upper,
                        "n": int(n),
                    }
                )

        summary_rows.append(result)

    summary = pd.DataFrame(
        summary_rows
    )

    counts = pd.DataFrame(
        count_rows
    )

    summary.to_csv(
        OUT / "duration_overlap_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    counts.to_csv(
        OUT / "duration_overlap_counts_by_generator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "feature_sha256": feature_sha,
        "split_sha256": split_sha,
        "rows": len(df),
        "generators": len(generators),
        "duration_field": (
            "prosody_duration_sec"
        ),
        "important": (
            "Duration interval boundaries "
            "were derived from TRAIN only. "
            "TEST was not inspected."
        ),
        "candidates": summary_rows,
    }

    (
        OUT
        / "duration_overlap_summary.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Duration-overlap diagnostic ==="
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Generators: {len(generators)}"
    )

    print(
        "IMPORTANT: interval boundaries "
        "derived from TRAIN only."
    )

    print(
        "TEST was not used."
    )

    columns = [
        "candidate",
        "lower_sec",
        "upper_sec",
        "width_sec",
        "valid_interval",
        "min_per_generator_train",
        "median_per_generator_train",
        "min_per_generator_validation",
        "median_per_generator_validation",
        "balanced_total_train",
        "balanced_total_validation",
    ]

    print(
        "\nCandidate intervals:"
    )

    print(
        summary[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved to:",
        OUT,
    )


if __name__ == "__main__":
    main()