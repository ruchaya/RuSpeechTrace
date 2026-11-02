
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from source_attribution.metrics import (
    closed_set_metrics,
    per_class_metrics,
)
from source_attribution.models import build_and_train


SPLITS = Path("splits/closed_set.csv")
ACOUSTIC = Path(
    "features/ruasd_closed_set_acoustic_fixed4.parquet"
)
WAVLM = Path(
    "embeddings/wavlm_base_ruasd_4s/chunks"
)
OUT = Path("results/raw/fixed4_acoustic_vs_wavlm_lr")

C_VALUES = [0.001, 0.01, 0.03, 0.1, 1.0, 10.0]
SEED = 42
MAX_ITER = 600
TOL = 1e-3


def save_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_representation(name):
    print(f"\nLOADING {name}", flush=True)

    if name == "acoustic_fixed4":
        schema = pd.read_parquet(ACOUSTIC).columns
        columns = [
            c for c in schema
            if c.startswith(("spectral_", "prosody_"))
        ]
        assert len(columns) == 789
        frame = pd.read_parquet(
            ACOUSTIC,
            columns=["file_id"] + columns,
        )

    elif name == "wavlm_fixed4":
        paths = sorted(WAVLM.glob("*.parquet"))
        assert len(paths) == 120, (
            f"Expected 120 chunks, found {len(paths)}"
        )

        frame = pd.concat(
            [pd.read_parquet(p) for p in paths],
            ignore_index=True,
        )

        columns = [
            f"emb_{i:04d}" for i in range(768)
        ]
        assert set(columns).issubset(frame.columns)

        frame = frame[["file_id"] + columns]

    else:
        raise ValueError(name)

    if len(frame) != 68000:
        raise RuntimeError(
            f"{name}: expected 68000 records"
        )

    if frame["file_id"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate file IDs"
        )

    return frame, columns


def train_one_representation(name, split):
    started = perf_counter()
    run = OUT / name
    run.mkdir(parents=True, exist_ok=True)

    features, columns = load_representation(name)

    df = split.merge(
        features,
        on="file_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(df) != 68000:
        raise RuntimeError(
            "Unexpected merged row count"
        )

    if not df["_merge"].eq("both").all():
        raise RuntimeError(
            f"{name}: missing features"
        )

    df = df.drop(columns="_merge")

    train = df["split"].eq("train")
    val = df["split"].eq("validation")
    test = df["split"].eq("test")

    assert int(train.sum()) == 44200
    assert int(val.sum()) == 10200
    assert int(test.sum()) == 13600

    # Same feature preprocessing in both experiments.
    X_train = df.loc[train, columns].replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    X_val = df.loc[val, columns].replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    X_test = df.loc[test, columns].replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    y_train = df.loc[
        train, "generator"
    ].astype(str).to_numpy()

    y_val = df.loc[
        val, "generator"
    ].astype(str).to_numpy()

    y_test = df.loc[
        test, "generator"
    ].astype(str).to_numpy()

    print(
        f"SHAPES: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}",
        flush=True,
    )

    best_model = None
    best_c = None
    best_key = (-1.0, -1.0)
    validation_rows = []

    for c in C_VALUES:
        t0 = perf_counter()

        print(
            f"{name}: TRAINING C={c}",
            flush=True,
        )

        model = build_and_train(
            X_train,
            y_train,
            model_type="logistic",
            params={
                "C": c,
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": MAX_ITER,
                "tol": TOL,
                "class_weight": "balanced",
            },
            seed=SEED,
        )

        val_pred = model.predict(X_val)
        val_metrics = closed_set_metrics(
            y_val,
            val_pred,
        )

        iterations = int(
            np.max(
                model.estimator.named_steps[
                    "classifier"
                ].n_iter_
            )
        )

        elapsed = perf_counter() - t0

        row = {
            "representation": name,
            "C": c,
            "validation_accuracy": val_metrics[
                "accuracy"
            ],
            "validation_macro_f1": val_metrics[
                "macro_f1"
            ],
            "iterations": iterations,
            "reached_iteration_limit": (
                iterations >= MAX_ITER
            ),
            "training_seconds": round(
                elapsed, 2
            ),
        }

        validation_rows.append(row)

        print(
            f"  VAL ACC={row['validation_accuracy']:.6f}"
            f" F1={row['validation_macro_f1']:.6f}"
            f" ITER={iterations}"
            f" TIME={elapsed:.1f}s",
            flush=True,
        )

        pd.DataFrame(
            validation_rows
        ).to_csv(
            run / "validation_grid.csv",
            index=False,
        )

        # The test split is not used for C selection.
        key = (
            val_metrics["macro_f1"],
            val_metrics["accuracy"],
        )

        if key > best_key:
            best_key = key
            best_c = c
            best_model = model

    if best_model is None:
        raise RuntimeError(
            "No Logistic Regression model selected"
        )

    print(
        f"{name}: SELECTED C={best_c}",
        flush=True,
    )

    # Evaluate the frozen selected model only once.
    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(
        X_test
    )

    test_metrics = closed_set_metrics(
        y_test,
        test_pred,
        test_proba,
        best_model.classes_,
    )

    predictions = pd.DataFrame({
        "file_id": df.loc[
            test, "file_id"
        ].astype(str).to_numpy(),
        "generator_true": y_test,
        "generator_pred": test_pred,
        "correct": y_test == test_pred,
    })

    predictions.to_parquet(
        run / "predictions.parquet",
        index=False,
    )

    pd.DataFrame(
        per_class_metrics(
            y_test,
            test_pred,
        )
    ).to_csv(
        run / "class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=best_model.classes_,
    )

    pd.DataFrame(
        cm,
        index=best_model.classes_,
        columns=best_model.classes_,
    ).to_csv(
        run / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    np.save(
        run / "test_probabilities.npy",
        test_proba.astype(np.float32),
    )

    (run / "feature_columns.txt").write_text(
        "\n".join(columns),
        encoding="utf-8",
    )

    joblib.dump(
        best_model,
        run / "model.joblib",
    )

    result = {
        "representation": name,
        "model": "LogisticRegression",
        "seed": SEED,
        "selected_C": best_c,
        "selection_metric": "validation_macro_f1",
        "validation_candidates": validation_rows,
        "test": test_metrics,
        "n_features": len(columns),
        "rows": {
            "train": 44200,
            "validation": 10200,
            "test": 13600,
        },
        "total_seconds": round(
            perf_counter() - started, 2
        ),
    }

    save_json(
        run / "metrics.json",
        result,
    )

    print(
        f"\n{name}: TEST RESULTS",
        flush=True,
    )
    print(
        json.dumps(
            test_metrics,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print(
        f"RESULTS SAVED: {run}",
        flush=True,
    )

    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(
        SPLITS,
        usecols=[
            "file_id",
            "generator",
            "split",
        ],
    )

    if (
        len(split) != 68000
        or split["file_id"].duplicated().any()
        or split["generator"].nunique() != 34
    ):
        raise RuntimeError(
            "Frozen split validation failed"
        )

    results = []

    for name in [
        "acoustic_fixed4",
        "wavlm_fixed4",
    ]:
        results.append(
            train_one_representation(
                name, split
            )
        )

    summary = pd.DataFrame([
        {
            "representation": r["representation"],
            "classifier": r["model"],
            "selected_C": r["selected_C"],
            "accuracy": r["test"]["accuracy"],
            "macro_f1": r["test"]["macro_f1"],
            "balanced_accuracy": r[
                "test"
            ]["balanced_accuracy"],
            "top3_accuracy": r[
                "test"
            ]["top3_accuracy"],
        }
        for r in results
    ])

    summary.to_csv(
        OUT / "comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== FINAL COMPARISON ==========",
        flush=True,
    )
    print(
        summary.to_string(index=False),
        flush=True,
    )

    print(
        "\nFIXED4 LR COMPARISON COMPLETED",
        flush=True,
    )


if __name__ == "__main__":
    main()
