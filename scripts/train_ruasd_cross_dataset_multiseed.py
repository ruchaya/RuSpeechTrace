
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import f1_score

from source_attribution.leakage import assert_no_leakage
from source_attribution.models import build_and_train
from source_attribution.utils import file_sha256, set_seed


SEEDS = [123, 456, 789, 2026]

FEATURE_FILE = Path("features/ruasd_closed_set_acoustic.parquet")
SPLIT_FILE = Path("splits/closed_set.csv")

REFERENCE_DIR = Path(
    "results/raw/shortcut_acoustic_no_duration/"
    "20260928T091631Z_seed42"
)

OUTPUT_ROOT = Path("results/raw")

EXPECTED_FEATURE_SHA = (
    "a21d537c5133a7e6daa852138f96ee7f8bf1491036cbf940da6451f688adaad8"
)
EXPECTED_SPLIT_SHA = (
    "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0"
)


def verify_frozen_inputs():
    print("VERIFYING FROZEN INPUTS", flush=True)

    actual_feature_sha = file_sha256(FEATURE_FILE)
    actual_split_sha = file_sha256(SPLIT_FILE)

    if actual_feature_sha != EXPECTED_FEATURE_SHA:
        raise RuntimeError("RuASD feature store SHA-256 mismatch")

    if actual_split_sha != EXPECTED_SPLIT_SHA:
        raise RuntimeError("RuASD split SHA-256 mismatch")

    cfg = yaml.safe_load(
        (REFERENCE_DIR / "config_resolved.yaml").read_text(
            encoding="utf-8"
        )
    )

    columns = [
        line.strip()
        for line in (REFERENCE_DIR / "feature_columns.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(columns) == 789
    assert len(set(columns)) == 789
    assert cfg["selected_columns"] == columns
    assert cfg["experiment"]["seed"] == 42
    assert cfg["model"]["type"] == "lightgbm"
    assert cfg["training"]["early_stopping_rounds"] == 80

    assert cfg["frozen_artifacts"]["features_sha256"] == actual_feature_sha
    assert cfg["frozen_artifacts"]["splits_sha256"] == actual_split_sha

    reference = joblib.load(REFERENCE_DIR / "model.joblib")

    assert list(reference.estimator.booster_.feature_name()) == columns
    assert len(reference.classes_) == 34

    print("REFERENCE MODEL VERIFIED", flush=True)
    print("FEATURES:", len(columns), flush=True)
    print("CLASSES:", len(reference.classes_), flush=True)

    return cfg, columns, set(map(str, reference.classes_))


def load_frozen_data(columns, reference_classes):
    print("\nLOADING FROZEN RUASD DATA", flush=True)

    splits = pd.read_csv(SPLIT_FILE, low_memory=False)
    assert_no_leakage(splits)

    features = pd.read_parquet(
        FEATURE_FILE,
        columns=["file_id"] + columns,
    )

    assert splits["file_id"].is_unique
    assert features["file_id"].is_unique

    df = splits[
        ["file_id", "generator", "split"]
    ].merge(
        features,
        on="file_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != 68000:
        raise RuntimeError(f"Expected 68000 rows; got {len(df)}")

    expected_counts = {
        "train": 44200,
        "validation": 10200,
        "test": 13600,
    }

    actual_counts = df["split"].value_counts().to_dict()

    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected split counts: {actual_counts}"
        )

    if set(df["generator"].astype(str)) != reference_classes:
        raise RuntimeError("Generator classes do not match reference")

    X = df[columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().all(axis=1).any():
        raise RuntimeError("At least one sample has all features missing")

    train_mask = df["split"].eq("train")
    val_mask = df["split"].eq("validation")
    test_mask = df["split"].eq("test")

    train_classes = set(
        df.loc[train_mask, "generator"].astype(str)
    )
    val_classes = set(
        df.loc[val_mask, "generator"].astype(str)
    )

    assert train_classes == reference_classes
    assert val_classes == reference_classes

    print("TRAIN:", int(train_mask.sum()), flush=True)
    print("VALIDATION:", int(val_mask.sum()), flush=True)
    print("TEST:", int(test_mask.sum()), flush=True)

    return df, X, train_mask, val_mask, test_mask


def train_one(
    seed,
    cfg,
    columns,
    reference_classes,
    df,
    X,
    train_mask,
    val_mask,
    test_mask,
):
    output_dir = OUTPUT_ROOT / (
        f"ruasd_34class_no_duration_seed{seed}"
    )

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: {output_dir}. "
            "Refusing to overwrite it."
        )

    print(f"\n========== SEED {seed} ==========", flush=True)
    print("TRAINING LIGHTGBM...", flush=True)

    set_seed(seed)

    model_cfg = cfg["model"]

    model = build_and_train(
        X.loc[train_mask],
        df.loc[train_mask, "generator"],
        X.loc[val_mask],
        df.loc[val_mask, "generator"],
        model_type=model_cfg["type"],
        params=model_cfg["params"],
        early_stopping_rounds=int(
            cfg["training"]["early_stopping_rounds"]
        ),
        seed=seed,
    )

    actual_columns = list(
        model.estimator.booster_.feature_name()
    )

    if actual_columns != columns:
        raise RuntimeError("Trained model feature order mismatch")

    if set(map(str, model.classes_)) != reference_classes:
        raise RuntimeError("Trained model class set mismatch")

    val_true = df.loc[
        val_mask, "generator"
    ].astype(str).to_numpy()

    test_true = df.loc[
        test_mask, "generator"
    ].astype(str).to_numpy()

    val_predictions = model.predict(X.loc[val_mask])
    test_predictions = model.predict(X.loc[test_mask])

    val_f1 = float(
        f1_score(val_true, val_predictions, average="macro")
    )
    test_f1 = float(
        f1_score(test_true, test_predictions, average="macro")
    )

    best_iteration = int(model.estimator.best_iteration_)

    print("BEST ITERATION:", best_iteration, flush=True)
    print("VALIDATION MACRO-F1:", round(val_f1, 6), flush=True)
    print("TEST MACRO-F1:", round(test_f1, 6), flush=True)

    if test_f1 < 0.95:
        raise RuntimeError(
            f"Unexpectedly low RuASD test Macro-F1: {test_f1}"
        )

    metadata = {
        "experiment": "ruasd_34class_no_duration_multiseed",
        "seed": seed,
        "source_reference_run": str(REFERENCE_DIR),
        "feature_store": str(FEATURE_FILE),
        "split_file": str(SPLIT_FILE),
        "feature_sha256": EXPECTED_FEATURE_SHA,
        "split_sha256": EXPECTED_SPLIT_SHA,
        "n_features": len(columns),
        "n_classes": len(model.classes_),
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_iteration": best_iteration,
        "validation_macro_f1": val_f1,
        "test_macro_f1": test_f1,
        "model_type": model_cfg["type"],
        "model_params": model_cfg["params"],
        "early_stopping_rounds": int(
            cfg["training"]["early_stopping_rounds"]
        ),
        "duration_excluded": True,
    }

    output_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(
        model,
        output_dir / "model.joblib",
    )

    (output_dir / "feature_columns.txt").write_text(
        "\n".join(columns),
        encoding="utf-8",
    )

    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("SAVED:", output_dir, flush=True)
    return metadata


def main():
    # Refuse to start if any intended output already exists.
    for seed in SEEDS:
        target_dir = OUTPUT_ROOT / (
            f"ruasd_34class_no_duration_seed{seed}"
        )
        if target_dir.exists():
            raise RuntimeError(
                f"Existing output detected: {target_dir}"
            )

    cfg, columns, reference_classes = verify_frozen_inputs()

    df, X, train_mask, val_mask, test_mask = load_frozen_data(
        columns,
        reference_classes,
    )

    results = []

    for seed in SEEDS:
        result = train_one(
            seed,
            cfg,
            columns,
            reference_classes,
            df,
            X,
            train_mask,
            val_mask,
            test_mask,
        )
        results.append(result)

    print("\n========== ALL TRAININGS COMPLETED ==========")

    summary = pd.DataFrame([
        {
            "seed": r["seed"],
            "best_iteration": r["best_iteration"],
            "validation_macro_f1": r["validation_macro_f1"],
            "test_macro_f1": r["test_macro_f1"],
        }
        for r in results
    ])

    print(summary.to_string(index=False))
    print(
        "\nRuASD models for seeds 123, 456, 789, 2026 "
        "are ready for cross-dataset evaluation.",
        flush=True,
    )


if __name__ == "__main__":
    main()
