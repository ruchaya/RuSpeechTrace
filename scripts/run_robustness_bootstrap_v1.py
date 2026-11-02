
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


MAIN = Path("results/raw/robustness_main_v1")
SAFE = Path("results/raw/robustness_safe_codec_control_v1")
OUT = Path("results/raw/robustness_statistics_v1")

N_BOOTSTRAP = 2000
RANDOM_SEED = 20261017

MAIN_CONDITIONS = [
    "clean",
    "noise_30db",
    "noise_20db",
    "noise_10db",
    "synthetic_echo",
    "mp3_64kbps",
]

COMPARISONS = [
    ("noise_30db", "clean"),
    ("noise_20db", "clean"),
    ("noise_10db", "clean"),
    ("synthetic_echo", "clean"),
    ("mp3_64kbps", "clean"),
    ("pcm16_original", "clean"),
    ("pcm16_safe", "clean"),
    ("mp3_safe", "clean"),
    ("pcm16_safe", "pcm16_original"),
    ("mp3_safe", "mp3_64kbps"),
    ("mp3_64kbps", "pcm16_original"),
    ("mp3_safe", "pcm16_safe"),
]


def checked_column(frame, column):
    if frame[column].isna().any():
        raise RuntimeError("Missing values: " + column)

    return frame[column].to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(
        MAIN / "selected_test_records.csv"
    )

    if len(selected) != 680:
        raise RuntimeError("Expected 680 records")

    if selected.file_id.duplicated().any():
        raise RuntimeError("Duplicate file IDs")

    selected = selected.sort_values(
        ["generator", "file_id"]
    ).reset_index(drop=True)

    file_ids = selected.file_id
    actual = selected.generator.to_numpy()
    labels = sorted(selected.generator.unique())

    if len(labels) != 34:
        raise RuntimeError("Expected 34 generators")

    group_indices = [
        np.flatnonzero(actual == name)
        for name in labels
    ]

    if any(len(indices) != 20 for indices in group_indices):
        raise RuntimeError("Expected 20 records per generator")

    main_predictions = pd.read_parquet(
        MAIN / "predictions.parquet"
    )

    if main_predictions.duplicated(
        ["file_id", "condition"]
    ).any():
        raise RuntimeError("Duplicate main predictions")

    predictions = {}

    for condition in MAIN_CONDITIONS:
        subset = main_predictions.loc[
            main_predictions.condition.eq(condition)
        ].set_index("file_id")

        if len(subset) != 680:
            raise RuntimeError(
                "Unexpected record count: " + condition
            )

        subset = subset.reindex(file_ids)

        true_labels = checked_column(
            subset, "generator"
        )

        if not np.array_equal(actual, true_labels):
            raise RuntimeError(
                "Labels mismatch: " + condition
            )

        predictions[condition] = checked_column(
            subset, "predicted_generator"
        )

    safe = pd.read_csv(
        SAFE / "predictions_comparison.csv"
    )

    if safe.file_id.duplicated().any():
        raise RuntimeError("Duplicate safe predictions")

    safe = safe.set_index("file_id").reindex(file_ids)

    if not np.array_equal(
        actual,
        checked_column(safe, "generator"),
    ):
        raise RuntimeError("Safe labels mismatch")

    if not np.array_equal(
        predictions["clean"],
        checked_column(safe, "clean_prediction"),
    ):
        raise RuntimeError("Clean predictions mismatch")

    if not np.array_equal(
        predictions["mp3_64kbps"],
        checked_column(safe, "mp3_original_prediction"),
    ):
        raise RuntimeError("MP3 predictions mismatch")

    predictions["pcm16_original"] = checked_column(
        safe, "pcm16_original_prediction"
    )

    predictions["pcm16_safe"] = checked_column(
        safe, "pcm16_safe_prediction"
    )

    predictions["mp3_safe"] = checked_column(
        safe, "mp3_safe_prediction"
    )

    conditions = list(predictions)

    known_labels = set(labels)

    for condition, pred in predictions.items():
        if not set(pred).issubset(known_labels):
            raise RuntimeError(
                "Unknown predicted labels: " + condition
            )

    print("STRATIFIED PAIRED BOOTSTRAP", flush=True)
    print("RECORDS:", len(actual), flush=True)
    print("GENERATORS:", len(labels), flush=True)
    print("RECORDS PER GENERATOR: 20", flush=True)
    print("BOOTSTRAP ITERATIONS:", N_BOOTSTRAP, flush=True)
    print("SEED:", RANDOM_SEED, flush=True)

    observed = {}

    for condition in conditions:
        observed[condition] = f1_score(
            actual,
            predictions[condition],
            labels=labels,
            average="macro",
            zero_division=0,
        )

    rng = np.random.default_rng(RANDOM_SEED)

    bootstrap = np.empty(
        (N_BOOTSTRAP, len(conditions)),
        dtype=np.float64,
    )

    for iteration in range(N_BOOTSTRAP):
        indices = np.concatenate([
            rng.choice(
                group,
                size=len(group),
                replace=True,
            )
            for group in group_indices
        ])

        y = actual[indices]

        for j, condition in enumerate(conditions):
            bootstrap[iteration, j] = f1_score(
                y,
                predictions[condition][indices],
                labels=labels,
                average="macro",
                zero_division=0,
            )

        if (iteration + 1) % 250 == 0:
            print(
                "BOOTSTRAP",
                iteration + 1,
                "/",
                N_BOOTSTRAP,
                flush=True,
            )

    position = {
        name: j
        for j, name in enumerate(conditions)
    }

    metric_rows = []

    print("\nMACRO-F1 AND 95% BOOTSTRAP CI")

    for condition in conditions:
        j = position[condition]
        low, high = np.percentile(
            bootstrap[:, j],
            [2.5, 97.5],
        )

        score = observed[condition]

        metric_rows.append({
            "condition": condition,
            "macro_f1": score,
            "ci95_low": low,
            "ci95_high": high,
            "accuracy": np.mean(
                predictions[condition] == actual
            ),
        })

        print(
            condition,
            "F1:", round(score, 6),
            "CI95:",
            round(float(low), 6),
            round(float(high), 6),
            flush=True,
        )

    difference_rows = []

    print("\nPAIRED MACRO-F1 DIFFERENCES AND 95% CI")

    for condition_a, condition_b in COMPARISONS:
        a = position[condition_a]
        b = position[condition_b]

        differences = (
            bootstrap[:, a] - bootstrap[:, b]
        )

        low, high = np.percentile(
            differences,
            [2.5, 97.5],
        )

        observed_difference = (
            observed[condition_a]
            - observed[condition_b]
        )

        difference_rows.append({
            "condition_a": condition_a,
            "condition_b": condition_b,
            "delta_macro_f1": observed_difference,
            "ci95_low": low,
            "ci95_high": high,
        })

        print(
            condition_a,
            "MINUS",
            condition_b,
            "DELTA:", round(observed_difference, 6),
            "CI95:",
            round(float(low), 6),
            round(float(high), 6),
            flush=True,
        )

    pd.DataFrame(metric_rows).to_csv(
        OUT / "macro_f1_bootstrap_ci.csv",
        index=False,
    )

    pd.DataFrame(difference_rows).to_csv(
        OUT / "paired_macro_f1_differences.csv",
        index=False,
    )

    print(
        "\nROBUSTNESS BOOTSTRAP PASSED",
        flush=True,
    )
    print("RESULTS SAVED:", OUT, flush=True)


if __name__ == "__main__":
    main()
