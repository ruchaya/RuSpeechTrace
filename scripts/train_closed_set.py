#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from source_attribution.config import load_config
from source_attribution.experiment import (
    create_run_dir,
    feature_columns,
    record_run_context,
)
from source_attribution.leakage import assert_no_leakage
from source_attribution.metrics import closed_set_metrics, per_class_metrics
from source_attribution.models import build_and_train
from source_attribution.utils import file_sha256, json_dump, set_seed


PACKAGE_DISTS = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "librosa",
    "soundfile",
    "lightgbm",
    "torch",
    "torchaudio",
    "transformers",
    "datasets",
    "huggingface_hub",
    "shap",
    "matplotlib",
    "tqdm",
    "PyYAML",
    "joblib",
    "pyarrow",
    "resampy",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the frozen RuASD clean closed-set Acoustic + LightGBM baseline."
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
        "--config",
        default="configs/closed_set_acoustic.yaml",
    )
    p.add_argument(
        "--inventory",
        default="data_inventory/ruasd_inventory.csv",
    )
    p.add_argument(
        "--feature-meta",
        default="features/ruasd_closed_set_acoustic.meta.json",
    )
    p.add_argument(
        "--feature-validation",
        default="features/ruasd_closed_set_acoustic_validation.json",
    )
    p.add_argument(
        "--split-validation",
        default="splits/frozen_split_validation.json",
    )
    return p.parse_args()


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGE_DISTS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def expected_hash_from_validation(
    validation_path: Path,
    artifact_path: Path,
) -> str | None:
    if not validation_path.exists():
        return None
    data = json.loads(validation_path.read_text(encoding="utf-8"))
    if not data.get("passed", False):
        raise RuntimeError(f"Validation did not pass: {validation_path}")

    target_name = artifact_path.name
    for key, value in (data.get("sha256") or {}).items():
        normalized_name = str(key).replace("\\", "/").split("/")[-1]
        if normalized_name == target_name:
            return str(value)
    return None


def assert_frozen_artifacts(
    feature_path: Path,
    split_path: Path,
    feature_meta_path: Path,
    feature_validation_path: Path,
    split_validation_path: Path,
) -> dict[str, str]:
    current_feature_sha = file_sha256(feature_path)
    current_split_sha = file_sha256(split_path)

    expected_feature_sha = expected_hash_from_validation(
        feature_validation_path,
        feature_path,
    )
    expected_split_sha = expected_hash_from_validation(
        split_validation_path,
        split_path,
    )

    if expected_feature_sha is None:
        raise RuntimeError(
            f"Could not find feature SHA-256 in {feature_validation_path}"
        )
    if expected_split_sha is None:
        raise RuntimeError(
            f"Could not find split SHA-256 in {split_validation_path}"
        )

    if current_feature_sha != expected_feature_sha:
        raise RuntimeError(
            "Feature store changed after validation: "
            f"{current_feature_sha} != {expected_feature_sha}"
        )
    if current_split_sha != expected_split_sha:
        raise RuntimeError(
            "Frozen split changed after validation: "
            f"{current_split_sha} != {expected_split_sha}"
        )

    if not feature_meta_path.exists():
        raise RuntimeError(f"Missing feature metadata: {feature_meta_path}")

    feature_meta = json.loads(
        feature_meta_path.read_text(encoding="utf-8")
    )
    if feature_meta.get("split_sha256") != current_split_sha:
        raise RuntimeError(
            "Feature store was extracted from a different split SHA-256."
        )
    if int(feature_meta.get("written_rows", -1)) != 68000:
        raise RuntimeError(
            f"Expected 68000 feature rows, got "
            f"{feature_meta.get('written_rows')}"
        )
    if int(feature_meta.get("feature_columns", -1)) != 790:
        raise RuntimeError(
            f"Expected 790 acoustic features, got "
            f"{feature_meta.get('feature_columns')}"
        )

    return {
        "features_sha256": current_feature_sha,
        "splits_sha256": current_split_sha,
        "feature_meta_sha256": file_sha256(feature_meta_path),
        "feature_validation_sha256": file_sha256(feature_validation_path),
        "split_validation_sha256": file_sha256(split_validation_path),
    }


def setup_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"closed_set_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        run_dir / "run.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def probability_frame(
    file_ids: pd.Series,
    proba: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    data: dict[str, object] = {
        "file_id": file_ids.astype(str).to_numpy()
    }
    for idx, cls in enumerate(classes):
        data[f"p::{cls}"] = proba[:, idx].astype(np.float32)
    return pd.DataFrame(data)


def main() -> None:
    args = parse_args()

    feature_path = Path(args.features)
    split_path = Path(args.splits)
    config_path = Path(args.config)
    inventory_path = Path(args.inventory)
    feature_meta_path = Path(args.feature_meta)
    feature_validation_path = Path(args.feature_validation)
    split_validation_path = Path(args.split_validation)

    cfg = load_config(config_path)
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    artifact_hashes = assert_frozen_artifacts(
        feature_path,
        split_path,
        feature_meta_path,
        feature_validation_path,
        split_validation_path,
    )
    artifact_hashes["config_sha256"] = file_sha256(config_path)
    if inventory_path.exists():
        artifact_hashes["inventory_sha256"] = file_sha256(inventory_path)
    else:
        artifact_hashes["inventory_sha256"] = "MISSING"

    sp = pd.read_csv(split_path, low_memory=False)
    assert_no_leakage(sp)

    feat = pd.read_parquet(feature_path)

    fs = cfg.get("feature_selection", {})
    cols = feature_columns(
        feat,
        fs.get("include_prefixes"),
        fs.get("exclude_columns"),
        bool(fs.get("exclude_duration", False)),
    )

    # Metadata come only from the frozen split.  This avoids duplicate
    # speaker/voice/source columns from the feature parquet after merge.
    meta_candidates = [
        "file_id",
        "sample_id",
        "split",
        "generator",
        "speaker",
        "voice",
        "text_hash",
        "augmentation_type",
        "source_dataset",
        "raw_or_augmented",
    ]
    meta = [c for c in meta_candidates if c in sp.columns]

    feature_table = feat[["file_id"] + cols].copy()
    df = sp[meta].merge(
        feature_table,
        on="file_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(sp):
        raise RuntimeError(f"Merge changed row count: {len(df)} != {len(sp)}")

    if df[cols].isna().all(axis=1).any():
        n = int(df[cols].isna().all(axis=1).sum())
        raise RuntimeError(f"Rows with no acoustic features after merge: {n}")

    X = df[cols].replace([np.inf, -np.inf], np.nan)
    tr = df["split"].eq("train")
    va = df["split"].eq("validation")
    te = df["split"].eq("test")

    expected_counts = {
        "train": 44200,
        "validation": 10200,
        "test": 13600,
    }
    actual_counts = {
        "train": int(tr.sum()),
        "validation": int(va.sum()),
        "test": int(te.sum()),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected split counts: {actual_counts} != {expected_counts}"
        )

    run = create_run_dir(
        cfg["experiment"].get("output_root", "results/raw"),
        cfg["experiment"]["name"],
        seed,
    )
    logger = setup_run_logger(run)

    record_run_context(run, cfg)
    json_dump(package_versions(), run / "package_versions.json")
    json_dump(artifact_hashes, run / "artifact_hashes.json")

    logger.info("Run directory: %s", run)
    logger.info("Seed: %d", seed)
    logger.info("Features: %d", len(cols))
    logger.info("Classes: %d", df["generator"].nunique())
    logger.info("Split counts: %s", actual_counts)
    logger.info("Frozen split SHA-256: %s", artifact_hashes["splits_sha256"])
    logger.info(
        "Feature store SHA-256: %s",
        artifact_hashes["features_sha256"],
    )

    mc = cfg["model"]
    model = build_and_train(
        X.loc[tr],
        df.loc[tr, "generator"],
        X.loc[va],
        df.loc[va, "generator"],
        mc["type"],
        mc.get("params", {}),
        int(cfg.get("training", {}).get("early_stopping_rounds", 80)),
        seed,
    )

    # Validation is used for model selection/early stopping; test is evaluated
    # only after the fitted model is fixed.
    val_proba = model.predict_proba(X.loc[va])
    val_pred = model.predict(X.loc[va])
    val_true = df.loc[va, "generator"].astype(str).to_numpy()
    val_metrics = closed_set_metrics(
        val_true,
        val_pred,
        val_proba,
        model.classes_,
    )

    test_proba = model.predict_proba(X.loc[te])
    test_pred = model.predict(X.loc[te])
    test_true = df.loc[te, "generator"].astype(str).to_numpy()
    test_metrics = closed_set_metrics(
        test_true,
        test_pred,
        test_proba,
        model.classes_,
    )

    metrics = {
        "experiment": cfg["experiment"]["name"],
        "seed": seed,
        "n_classes": int(len(model.classes_)),
        "n_features": int(len(cols)),
        "rows": actual_counts,
        "validation": val_metrics,
        "test": test_metrics,
    }
    json_dump(metrics, run / "metrics.json")

    pd.DataFrame(
        per_class_metrics(test_true, test_pred)
    ).to_csv(
        run / "class_metrics.csv",
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
        run / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    pred_cols = [
        c for c in [
            "file_id",
            "sample_id",
            "speaker",
            "voice",
            "text_hash",
            "augmentation_type",
            "source_dataset",
        ]
        if c in df.columns
    ]
    predictions = (
        df.loc[te, pred_cols]
        .copy()
        .reset_index(drop=True)
    )
    predictions["generator_true"] = test_true
    predictions["generator_pred"] = test_pred
    predictions["dataset"] = "RuASD"
    predictions["distortion"] = "clean"
    predictions["seed"] = seed
    predictions["probabilities_or_logits_reference"] = (
        "probabilities.parquet"
    )
    predictions.to_parquet(
        run / "predictions.parquet",
        index=False,
    )

    probability_frame(
        df.loc[te, "file_id"].reset_index(drop=True),
        test_proba,
        model.classes_,
    ).to_parquet(
        run / "probabilities.parquet",
        index=False,
    )

    np.save(run / "probabilities.npy", test_proba)
    (run / "classes.txt").write_text(
        "\n".join(map(str, model.classes_)),
        encoding="utf-8",
    )
    (run / "feature_columns.txt").write_text(
        "\n".join(cols),
        encoding="utf-8",
    )
    joblib.dump(model, run / "model.joblib")

    estimator = model.estimator
    model_metadata = {
        "kind": model.kind,
        "classes": list(map(str, model.classes_)),
        "n_classes": int(len(model.classes_)),
        "n_features": int(len(cols)),
        "seed": seed,
        "best_iteration": (
            int(estimator.best_iteration_)
            if getattr(estimator, "best_iteration_", None) is not None
            else None
        ),
        "estimator_params": estimator.get_params()
        if hasattr(estimator, "get_params")
        else {},
    }
    json_dump(model_metadata, run / "model_metadata.json")

    if hasattr(estimator, "feature_importances_"):
        pd.DataFrame(
            {
                "feature": cols,
                "importance": np.asarray(
                    estimator.feature_importances_,
                    dtype=float,
                ),
            }
        ).sort_values(
            "importance",
            ascending=False,
        ).to_csv(
            run / "feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    logger.info(
        "Validation metrics: %s",
        json.dumps(val_metrics, ensure_ascii=False),
    )
    logger.info(
        "Test metrics: %s",
        json.dumps(test_metrics, ensure_ascii=False),
    )
    logger.info("Saved canonical closed-set baseline artifacts.")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(run)


if __name__ == "__main__":
    main()
