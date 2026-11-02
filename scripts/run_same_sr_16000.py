from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from source_attribution.experiment import create_run_dir, record_run_context
from source_attribution.leakage import assert_no_leakage
from source_attribution.metrics import closed_set_metrics, per_class_metrics
from source_attribution.models import build_and_train
from source_attribution.utils import file_sha256, json_dump, set_seed


FEATURE_PATH = Path("features/ruasd_closed_set_acoustic.parquet")
SPLIT_PATH = Path("splits/closed_set.csv")
CONFIG_PATH = Path("configs/same_sr_16000.yaml")

EXPECTED_FEATURE_SHA = (
    "a21d537c5133a7e6daa852138f96ee7f8bf1491036cbf940da6451f688adaad8"
)
EXPECTED_SPLIT_SHA = (
    "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0"
)

EXPECTED_QUOTA = {
    "train": 1300,
    "validation": 300,
    "test": 400,
}


def run_experiment(
    df,
    feature_columns,
    experiment_name,
    cfg,
    feature_sha,
    split_sha,
):
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    tr = df["split"].eq("train")
    va = df["split"].eq("validation")
    te = df["split"].eq("test")

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    model_cfg = cfg["model"]

    model = build_and_train(
        X.loc[tr],
        df.loc[tr, "generator"],
        X.loc[va],
        df.loc[va, "generator"],
        model_cfg["type"],
        model_cfg["params"],
        int(cfg["training"]["early_stopping_rounds"]),
        seed,
    )

    val_proba = model.predict_proba(X.loc[va])
    val_pred = model.predict(X.loc[va])
    val_true = df.loc[va, "generator"].astype(str).to_numpy()

    test_proba = model.predict_proba(X.loc[te])
    test_pred = model.predict(X.loc[te])
    test_true = df.loc[te, "generator"].astype(str).to_numpy()

    val_metrics = closed_set_metrics(
        val_true,
        val_pred,
        val_proba,
        model.classes_,
    )

    test_metrics = closed_set_metrics(
        test_true,
        test_pred,
        test_proba,
        model.classes_,
    )

    run_cfg = {
        "experiment": {
            "name": experiment_name,
            "seed": seed,
            "output_root": cfg["experiment"]["output_root"],
            "status": cfg["experiment"]["status"],
        },
        "control": cfg["control"],
        "features": feature_columns,
        "model": cfg["model"],
        "training": cfg["training"],
    }

    run_dir = create_run_dir(
        cfg["experiment"]["output_root"],
        experiment_name,
        seed,
    )

    record_run_context(run_dir, run_cfg)

    metrics = {
        "experiment": experiment_name,
        "status": "posthoc_shortcut_diagnostic",
        "native_sample_rate": 16000,
        "n_classes": int(len(model.classes_)),
        "chance_top1": 1.0 / len(model.classes_),
        "n_features": len(feature_columns),
        "rows": {
            "train": int(tr.sum()),
            "validation": int(va.sum()),
            "test": int(te.sum()),
        },
        "validation": val_metrics,
        "test": test_metrics,
    }

    json_dump(metrics, run_dir / "metrics.json")

    pd.DataFrame(
        per_class_metrics(test_true, test_pred)
    ).to_csv(
        run_dir / "class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(
        test_true,
        test_pred,
        labels=model.classes_,
    )

    pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_,
    ).to_csv(
        run_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    predictions = df.loc[
        te,
        [
            "file_id",
            "generator",
            "text_hash",
            "sample_rate",
            "channels",
        ],
    ].copy().reset_index(drop=True)

    predictions = predictions.rename(
        columns={"generator": "generator_true"}
    )

    predictions["generator_pred"] = test_pred
    predictions["dataset"] = "RuASD"
    predictions["distortion"] = "clean"
    predictions["seed"] = seed
    predictions["probabilities_or_logits_reference"] = (
        "probabilities.parquet"
    )

    predictions.to_parquet(
        run_dir / "predictions.parquet",
        index=False,
    )

    probabilities = pd.DataFrame(
        {"file_id": predictions["file_id"].astype(str)}
    )

    for i, cls in enumerate(model.classes_):
        probabilities[f"p::{cls}"] = test_proba[:, i].astype(
            np.float32
        )

    probabilities.to_parquet(
        run_dir / "probabilities.parquet",
        index=False,
    )

    joblib.dump(
        model,
        run_dir / "model.joblib",
    )

    (run_dir / "classes.txt").write_text(
        "\n".join(map(str, model.classes_)),
        encoding="utf-8",
    )

    (run_dir / "feature_columns.txt").write_text(
        "\n".join(feature_columns),
        encoding="utf-8",
    )

    json_dump(
        {
            "features_sha256": feature_sha,
            "splits_sha256": split_sha,
            "native_sample_rate": 16000,
            "selected_columns": feature_columns,
        },
        run_dir / "artifact_hashes.json",
    )

    estimator = model.estimator

    json_dump(
        {
            "kind": model.kind,
            "seed": seed,
            "n_classes": int(len(model.classes_)),
            "n_features": len(feature_columns),
            "best_iteration": (
                int(estimator.best_iteration_)
                if getattr(estimator, "best_iteration_", None)
                is not None
                else None
            ),
        },
        run_dir / "model_metadata.json",
    )

    return {
        "experiment": experiment_name,
        "n_features": len(feature_columns),
        "validation_macro_f1": val_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics[
            "balanced_accuracy"
        ],
        "test_top1_accuracy": test_metrics[
            "top1_accuracy"
        ],
        "test_top3_accuracy": test_metrics[
            "top3_accuracy"
        ],
        "run_dir": str(run_dir),
    }


def main():
    cfg = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    feature_sha = file_sha256(FEATURE_PATH)
    split_sha = file_sha256(SPLIT_PATH)

    if feature_sha != EXPECTED_FEATURE_SHA:
        raise RuntimeError(
            "Feature store SHA changed:\n"
            f"{feature_sha}"
        )

    if split_sha != EXPECTED_SPLIT_SHA:
        raise RuntimeError(
            "Frozen split SHA changed:\n"
            f"{split_sha}"
        )

    splits = pd.read_csv(
        SPLIT_PATH,
        low_memory=False,
    )

    assert_no_leakage(splits)

    features = pd.read_parquet(
        FEATURE_PATH
    )

    acoustic_columns = [
        c
        for c in features.columns
        if c.startswith("spectral_")
        or c.startswith("prosody_")
    ]

    if len(acoustic_columns) != 790:
        raise RuntimeError(
            f"Expected 790 acoustic columns, "
            f"got {len(acoustic_columns)}"
        )

    acoustic_no_duration = [
        c
        for c in acoustic_columns
        if c != "prosody_duration_sec"
    ]

    if len(acoustic_no_duration) != 789:
        raise RuntimeError(
            f"Expected 789 features without duration, "
            f"got {len(acoustic_no_duration)}"
        )

    keep_columns = list(
        dict.fromkeys(
            [
                "file_id",
                "sample_rate",
                "channels",
            ]
            + acoustic_columns
        )
    )

    df = splits[
        [
            "file_id",
            "generator",
            "split",
            "text_hash",
        ]
    ].merge(
        features[keep_columns],
        on="file_id",
        how="left",
        validate="one_to_one",
    )

    generators = cfg["control"]["generators"]

    subset = df[
        df["generator"].isin(generators)
    ].copy()

    found_generators = sorted(
        subset["generator"].unique()
    )

    if found_generators != sorted(generators):
        raise RuntimeError(
            "Generator list mismatch"
        )

    if not subset["sample_rate"].eq(16000).all():
        print(
            subset.loc[
                ~subset["sample_rate"].eq(16000),
                ["generator", "sample_rate"],
            ]
            .value_counts()
        )
        raise RuntimeError(
            "Not all selected samples are native 16 kHz"
        )

    if subset["channels"].nunique() != 1:
        raise RuntimeError(
            "Channels are not constant"
        )

    quota = (
        subset
        .groupby(["generator", "split"])
        .size()
        .unstack(fill_value=0)
    )

    for generator in generators:
        for split_name, expected in EXPECTED_QUOTA.items():
            got = int(
                quota.loc[generator, split_name]
            )

            if got != expected:
                raise RuntimeError(
                    f"{generator}/{split_name}: "
                    f"{got} != {expected}"
                )

    if len(subset) != 10000:
        raise RuntimeError(
            f"Expected 10000 rows, got {len(subset)}"
        )

    print(
        "\n=== Same-native-sample-rate "
        "16 kHz control ==="
    )

    print("Rows:", len(subset))
    print("Classes:", len(generators))

    print(
        "Train:",
        int(
            subset["split"]
            .eq("train")
            .sum()
        ),
    )

    print(
        "Validation:",
        int(
            subset["split"]
            .eq("validation")
            .sum()
        ),
    )

    print(
        "Test:",
        int(
            subset["split"]
            .eq("test")
            .sum()
        ),
    )

    print(
        "Native sample rates:",
        sorted(
            subset["sample_rate"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Channels:",
        sorted(
            subset["channels"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Chance Top-1:",
        1 / len(generators),
    )

    results = []

    print(
        "\n=== acoustic_no_duration ==="
    )

    results.append(
        run_experiment(
            subset,
            acoustic_no_duration,
            "same_sr_16000_acoustic_no_duration",
            cfg,
            feature_sha,
            split_sha,
        )
    )

    print(
        json.dumps(
            results[-1],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\n=== duration_only ==="
    )

    results.append(
        run_experiment(
            subset,
            ["prosody_duration_sec"],
            "same_sr_16000_duration_only",
            cfg,
            feature_sha,
            split_sha,
        )
    )

    print(
        json.dumps(
            results[-1],
            ensure_ascii=False,
            indent=2,
        )
    )

    summary = pd.DataFrame(
        results
    )

    output_dir = Path(
        cfg["experiment"]["summary_root"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "same_sr_16000_seed42.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== Summary ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSummary CSV:",
        output_path,
    )


if __name__ == "__main__":
    main()
