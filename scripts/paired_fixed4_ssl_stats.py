
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path("results/raw/fixed4_acoustic_vs_wavlm_lr")
OUTPUT = ROOT / "paired_statistics"

MODEL_PATHS = {
    "Acoustic_LR": ROOT / "acoustic_fixed4/predictions.parquet",
    "WavLM_LR": ROOT / "wavlm_fixed4/predictions.parquet",
    "Acoustic_LightGBM": Path(
        "results/raw/closed_set_acoustic_fixed4_lightgbm/"
        "20261003T104455Z_seed42/predictions.parquet"
    ),
}

PAIRS = [
    ("Acoustic_LR", "WavLM_LR"),
    ("Acoustic_LR", "Acoustic_LightGBM"),
    ("Acoustic_LightGBM", "WavLM_LR"),
]

N_BOOTSTRAP = 2000
SEED = 42


def metrics(y, pred, n_classes):
    cm = np.bincount(
    	y.astype(np.int64) * n_classes + pred.astype(np.int64),
    	minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    denominator = support + predicted

    class_f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros(n_classes, dtype=float),
        where=denominator > 0,
    )

    return np.array([
        np.trace(cm) / cm.sum(),
        class_f1.mean(),
    ])


def interval(values):
    return np.percentile(values, [2.5, 97.5])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("LOADING THREE SETS OF TEST PREDICTIONS", flush=True)

    frames = {}

    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_parquet(
            path,
            columns=[
                "file_id",
                "generator_true",
                "generator_pred",
            ],
        )

        if len(frame) != 13600:
            raise RuntimeError(
                f"{name}: unexpected test size {len(frame)}"
            )

        if frame["file_id"].duplicated().any():
            raise RuntimeError(
                f"{name}: duplicate file IDs"
            )

        for col in ["file_id", "generator_true", "generator_pred"]:
            frame[col] = frame[col].astype(str)

        frames[name] = frame.set_index(
            "file_id"
        ).sort_index()

    names = list(MODEL_PATHS)
    reference = frames[names[0]]

    for name in names[1:]:
        other = frames[name]

        if not reference.index.equals(other.index):
            raise RuntimeError(
                f"{name}: test file IDs differ"
            )

        if not np.array_equal(
            reference["generator_true"].to_numpy(),
            other["generator_true"].to_numpy(),
        ):
            raise RuntimeError(
                f"{name}: true labels differ"
            )

    print("TEST ALIGNMENT VERIFIED: 13600 RECORDS", flush=True)

    labels = sorted(
        reference["generator_true"].unique()
    )
    n_classes = len(labels)

    if n_classes != 34:
        raise RuntimeError(
            f"Expected 34 classes, got {n_classes}"
        )

    y = pd.Categorical(
        reference["generator_true"],
        categories=labels,
    ).codes

    if (y < 0).any():
        raise RuntimeError("Unknown true label")

    class_indices = [
        np.flatnonzero(y == k)
        for k in range(n_classes)
    ]

    if not all(
        len(idx) == 400 for idx in class_indices
    ):
        raise RuntimeError(
            "Expected exactly 400 test records per class"
        )

    predictions = {}
    correct = {}

    for name in names:
        pred = pd.Categorical(
            frames[name]["generator_pred"],
            categories=labels,
        ).codes

        if (pred < 0).any():
            raise RuntimeError(
                f"{name}: unknown predicted class"
            )

        predictions[name] = pred
        correct[name] = pred == y

    print("PREDICTIONS VERIFIED", flush=True)

    observed = {
        name: metrics(y, predictions[name], n_classes)
        for name in names
    }

    boot = np.zeros(
        (N_BOOTSTRAP, len(names), 2),
        dtype=np.float64,
    )

    rng = np.random.default_rng(SEED)

    print(
        f"RUNNING {N_BOOTSTRAP} PAIRED "
        "CLASS-STRATIFIED BOOTSTRAP REPLICATES",
        flush=True,
    )

    for replicate in range(N_BOOTSTRAP):
        # Use the same resampled records for all models.
        sampled = np.concatenate([
            idx[rng.integers(
                0, len(idx), size=len(idx)
            )]
            for idx in class_indices
        ])

        ys = y[sampled]

        for model_index, name in enumerate(names):
            boot[
                replicate, model_index
            ] = metrics(
                ys,
                predictions[name][sampled],
                n_classes,
            )

        if (replicate + 1) % 500 == 0:
            print(
                f"BOOTSTRAP: {replicate + 1}/{N_BOOTSTRAP}",
                flush=True,
            )

    summary_rows = []

    for i, name in enumerate(names):
        acc_ci = interval(boot[:, i, 0])
        f1_ci = interval(boot[:, i, 1])

        summary_rows.append({
            "model": name,
            "accuracy": observed[name][0],
            "accuracy_ci_low": acc_ci[0],
            "accuracy_ci_high": acc_ci[1],
            "macro_f1": observed[name][1],
            "macro_f1_ci_low": f1_ci[0],
            "macro_f1_ci_high": f1_ci[1],
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT / "model_metrics_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMODEL METRICS WITH 95% CI", flush=True)
    print(
        summary.round(6).to_string(index=False),
        flush=True,
    )

    pair_rows = []

    for first, second in PAIRS:
        i = names.index(first)
        j = names.index(second)

        delta_accuracy = (
            boot[:, i, 0] - boot[:, j, 0]
        )
        delta_f1 = (
            boot[:, i, 1] - boot[:, j, 1]
        )

        accuracy_ci = interval(delta_accuracy)
        f1_ci = interval(delta_f1)

        # Discordant correctness outcomes.
        first_only = int(np.sum(
            correct[first] & ~correct[second]
        ))
        second_only = int(np.sum(
            ~correct[first] & correct[second]
        ))

        discordant = first_only + second_only

        if discordant:
            p = binomtest(
                min(first_only, second_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        else:
            p = 1.0

        pair_rows.append({
            "model_a": first,
            "model_b": second,
            "accuracy_difference": (
                observed[first][0]
                - observed[second][0]
            ),
            "accuracy_diff_ci_low": accuracy_ci[0],
            "accuracy_diff_ci_high": accuracy_ci[1],
            "macro_f1_difference": (
                observed[first][1]
                - observed[second][1]
            ),
            "macro_f1_diff_ci_low": f1_ci[0],
            "macro_f1_diff_ci_high": f1_ci[1],
            "a_correct_b_wrong": first_only,
            "a_wrong_b_correct": second_only,
            "mcnemar_exact_p": p,
        })

    comparison = pd.DataFrame(pair_rows)

    # Holm correction across three McNemar comparisons.
    raw_p = comparison[
        "mcnemar_exact_p"
    ].to_numpy()

    order = np.argsort(raw_p)
    adjusted = np.zeros(len(raw_p))
    running_max = 0.0

    for position, original_index in enumerate(order):
        candidate = min(
            1.0,
            (len(raw_p) - position)
            * raw_p[original_index],
        )
        running_max = max(
            running_max, candidate
        )
        adjusted[original_index] = running_max

    comparison["mcnemar_holm_p"] = adjusted

    comparison.to_csv(
        OUTPUT / "paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPAIRED COMPARISONS", flush=True)
    print(
        comparison.round(6).to_string(index=False),
        flush=True,
    )

    class_rows = []

    for k, generator in enumerate(labels):
        mask = y == k

        row = {
            "generator": generator,
            "support": int(mask.sum()),
        }

        for name in names:
            row[f"{name}_recall"] = float(
                correct[name][mask].mean()
            )

        row["acoustic_minus_wavlm_recall"] = (
            row["Acoustic_LR_recall"]
            - row["WavLM_LR_recall"]
        )

        class_rows.append(row)

    class_table = pd.DataFrame(
        class_rows
    ).sort_values(
        "acoustic_minus_wavlm_recall",
        ascending=False,
    )

    class_table.to_csv(
        OUTPUT / "per_generator_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nTOP 10 GENERATORS: "
        "ACOUSTIC ADVANTAGE OVER WAVLM",
        flush=True,
    )
    print(
        class_table[
            [
                "generator",
                "Acoustic_LR_recall",
                "WavLM_LR_recall",
                "acoustic_minus_wavlm_recall",
            ]
        ].head(10).round(5).to_string(index=False),
        flush=True,
    )

    manifest = {
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "method": (
            "Paired stratified bootstrap; "
            "400 test recordings resampled with "
            "replacement within each true generator"
        ),
        "confidence_interval": (
            "Percentile 95% interval"
        ),
        "hypothesis_test": (
            "Exact McNemar test for paired accuracy"
        ),
        "multiple_testing": (
            "Holm adjustment for three comparisons"
        ),
        "classifiers": {
            name: str(path)
            for name, path in MODEL_PATHS.items()
        },
        "interpretation_limit": (
            "Intervals quantify recording-level "
            "test uncertainty with fixed fitted models "
            "and fixed class balance. They do not "
            "quantify model-seed or cross-corpus variability."
        ),
    }

    (
        OUTPUT / "analysis_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nPAIRED STATISTICAL ANALYSIS COMPLETED",
        flush=True,
    )
    print("RESULTS:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
