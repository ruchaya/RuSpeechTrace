
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from source_attribution.config import load_config
from source_attribution.experiment import feature_columns
from source_attribution.models import build_and_train
from source_attribution.utils import file_sha256, set_seed


FEATURES = Path("features/mlaad_23class_acoustic.parquet")
FEATURE_META = Path("features/mlaad_23class_acoustic.meta.json")
SPLIT = Path("splits/mlaad_ru_23class_split_v1.csv")
CONFIG = Path("configs/closed_set_acoustic.yaml")


def save_json(data, path):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluate(model, X, y):
    pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(y, pred, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y, pred, average="weighted")
        ),
    }
    return metrics, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = load_config(CONFIG)

    meta = json.loads(
        FEATURE_META.read_text(encoding="utf-8")
    )

    split_sha = file_sha256(SPLIT)
    feature_sha = file_sha256(FEATURES)

    assert meta["split_sha256"] == split_sha
    assert meta["rows"] == 23000
    assert meta["generators"] == 23
    assert meta["feature_columns"] == 790

    d = pd.read_parquet(FEATURES)

    assert len(d) == 23000
    assert d["path"].is_unique
    assert d["generator"].nunique() == 23
    assert set(d["split"]) == {
        "train", "validation", "test"
    }

    original = pd.read_csv(
        SPLIT,
        usecols=["path", "generator", "split"],
        encoding="utf-8-sig",
    )

    left = d[
        ["path", "generator", "split"]
    ].sort_values("path").reset_index(drop=True)

    right = original.sort_values(
        "path"
    ).reset_index(drop=True)

    assert left.equals(right)

    for key in (
        "component",
        "text_key",
        "original_file",
    ):
        assert d.groupby(key)["split"].nunique().max() == 1

    balance = pd.crosstab(
        d["generator"], d["split"]
    )

    assert (balance["train"] == 650).all()
    assert (balance["validation"] == 150).all()
    assert (balance["test"] == 200).all()

    fs = cfg.get("feature_selection", {})

    cols = feature_columns(
        d,
        fs.get("include_prefixes"),
        fs.get("exclude_columns"),
        True,
    )

    assert len(cols) == 789
    assert "prosody_duration_sec" not in cols

    X = d[cols].replace(
        [np.inf, -np.inf], np.nan
    )

    assert all(
        pd.api.types.is_numeric_dtype(X[c])
        for c in cols
    )

    assert not X.isna().all(axis=1).any()
    assert not X.isna().all(axis=0).any()

    tr = d["split"].eq("train")
    va = d["split"].eq("validation")
    te = d["split"].eq("test")

    assert (int(tr.sum()), int(va.sum()), int(te.sum())) == (
        14950, 3450, 4600
    )

    print("MLAAD 23-CLASS AUDIT PASSED")
    print("TOTAL ROWS:", len(d))
    print("GENERATORS:", d["generator"].nunique())
    print("FEATURES:", len(cols))
    print("TRAIN:", int(tr.sum()))
    print("VALIDATION:", int(va.sum()))
    print("TEST:", int(te.sum()))
    print("MISSING FEATURE CELLS:", int(X.isna().sum().sum()))
    print("ROWS WITH MISSING FEATURES:", int(X.isna().any(axis=1).sum()))
    print("LEAKAGE CHECK: PASSED")
    print("SPLIT SHA256:", split_sha)
    print("FEATURE SHA256:", feature_sha)

    if args.dry_run:
        print("DRY RUN PASSED - NO TRAINING")
        return

    output = Path(
        f"results/raw/mlaad_23class_closedset_seed{args.seed}"
    )

    if output.exists():
        raise RuntimeError(
            f"Output directory already exists: {output}"
        )

    output.mkdir(parents=True)

    save_json(
        {
            "dataset": "MLAAD",
            "task": "23_class_closed_set",
            "seed": args.seed,
            "classes": sorted(
                d["generator"].unique().tolist()
            ),
            "feature_count": len(cols),
            "duration_excluded": True,
            "feature_sha256": feature_sha,
            "split_sha256": split_sha,
            "config_sha256": file_sha256(CONFIG),
            "missing_values": "native_lightgbm",
            "reference_speaker_disjoint": False,
        },
        output / "experiment_context.json",
    )

    (output / "feature_columns.txt").write_text(
        "\n".join(cols),
        encoding="utf-8",
    )

    mc = cfg["model"]

    print("TRAINING LIGHTGBM...")

    model = build_and_train(
        X.loc[tr],
        d.loc[tr, "generator"],
        X.loc[va],
        d.loc[va, "generator"],
        mc["type"],
        mc.get("params", {}),
        int(cfg["training"]["early_stopping_rounds"]),
        args.seed,
    )

    val_metrics, _ = evaluate(
        model,
        X.loc[va],
        d.loc[va, "generator"],
    )

    test_metrics, pred = evaluate(
        model,
        X.loc[te],
        d.loc[te, "generator"],
    )

    test_truth = d.loc[
        te, "generator"
    ].astype(str).to_numpy()

    save_json(
        {
            "seed": args.seed,
            "n_classes": 23,
            "n_features": len(cols),
            "validation": val_metrics,
            "test": test_metrics,
        },
        output / "metrics.json",
    )

    report = classification_report(
        test_truth,
        pred,
        labels=model.classes_,
        output_dict=True,
        zero_division=0,
    )

    save_json(
        report,
        output / "classification_report.json",
    )

    matrix = confusion_matrix(
        test_truth,
        pred,
        labels=model.classes_,
    )

    pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_,
    ).to_csv(
        output / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    predictions = d.loc[
        te,
        ["path", "generator", "split"],
    ].copy()

    predictions["predicted_generator"] = pred

    predictions.to_parquet(
        output / "predictions.parquet",
        index=False,
    )

    probabilities = model.predict_proba(X.loc[te])

    prob_df = pd.DataFrame(
        probabilities,
        columns=[
            "p::" + str(c) for c in model.classes_
        ],
    )

    prob_df.insert(
        0, "path",
        d.loc[te, "path"].to_numpy(),
    )

    prob_df.to_parquet(
        output / "probabilities.parquet",
        index=False,
    )

    joblib.dump(
        model,
        output / "model.joblib",
    )

    print("VALIDATION:", val_metrics)
    print("TEST:", test_metrics)
    print("SAVED:", output)
    print("TRAINING PASSED")


if __name__ == "__main__":
    main()
