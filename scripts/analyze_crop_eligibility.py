from pathlib import Path

import pandas as pd

from source_attribution.leakage import assert_no_leakage
from source_attribution.utils import file_sha256


FEATURES = Path("features/ruasd_closed_set_acoustic.parquet")
SPLITS = Path("splits/closed_set.csv")

EXPECTED_FEATURE_SHA = (
    "a21d537c5133a7e6daa852138f96ee7f8bf1491036cbf940da6451f688adaad8"
)
EXPECTED_SPLIT_SHA = (
    "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0"
)


def main():
    if file_sha256(FEATURES) != EXPECTED_FEATURE_SHA:
        raise RuntimeError("Frozen feature store changed")

    if file_sha256(SPLITS) != EXPECTED_SPLIT_SHA:
        raise RuntimeError("Frozen split changed")

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

    thresholds = [2.0, 4.0, 8.0, 10.0]

    rows = []

    for seconds in thresholds:
        for split_name in [
            "train",
            "validation",
            "test",
        ]:
            part = df[
                df["split"].eq(split_name)
            ]

            counts = (
                part.assign(
                    eligible=(
                        part["prosody_duration_sec"]
                        >= seconds
                    )
                )
                .groupby("generator")["eligible"]
                .agg(["sum", "count"])
            )

            counts["short"] = (
                counts["count"]
                - counts["sum"]
            )

            rows.append(
                {
                    "crop_sec": seconds,
                    "split": split_name,
                    "total": int(
                        counts["count"].sum()
                    ),
                    "eligible": int(
                        counts["sum"].sum()
                    ),
                    "short": int(
                        counts["short"].sum()
                    ),
                    "min_eligible_per_generator": int(
                        counts["sum"].min()
                    ),
                    "median_eligible_per_generator": float(
                        counts["sum"].median()
                    ),
                    "max_short_per_generator": int(
                        counts["short"].max()
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    out = Path(
        "results/summaries/"
        "crop_eligibility.csv"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        out,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== Fixed-crop eligibility ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:",
        out,
    )


if __name__ == "__main__":
    main()