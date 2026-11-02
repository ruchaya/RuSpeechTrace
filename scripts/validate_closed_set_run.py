#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


REQUIRED_FILES = [
    "metrics.json",
    "predictions.parquet",
    "probabilities.parquet",
    "class_metrics.csv",
    "confusion_matrix.csv",
    "model.joblib",
    "model_metadata.json",
    "artifact_hashes.json",
    "feature_columns.txt",
    "classes.txt",
    "run.log",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and summarize a completed closed-set baseline run."
    )
    p.add_argument("run_dir", help="Path to the completed run directory.")
    return p.parse_args()


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    k = min(int(k), proba.shape[1])
    idx = np.argpartition(proba, -k, axis=1)[:, -k:]
    top = classes[idx]
    return float(np.mean([yt in row for yt, row in zip(y_true, top)]))


def main() -> None:
    args = parse_args()
    run = Path(args.run_dir)

    problems: list[str] = []
    for name in REQUIRED_FILES:
        if not (run / name).exists():
            problems.append(f"Missing required artifact: {name}")

    if problems:
        print(json.dumps({"passed": False, "problems": problems}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    model_meta = json.loads((run / "model_metadata.json").read_text(encoding="utf-8"))
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))

    pred = pd.read_parquet(run / "predictions.parquet")
    prob = pd.read_parquet(run / "probabilities.parquet")

    if pred["file_id"].duplicated().any():
        problems.append("Duplicate file_id in predictions.parquet")
    if prob["file_id"].duplicated().any():
        problems.append("Duplicate file_id in probabilities.parquet")

    merged = pred.merge(prob, on="file_id", how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        problems.append(f"Probability coverage mismatch: {len(merged)} != {len(pred)}")

    prob_cols = [c for c in merged.columns if c.startswith("p::")]
    classes = np.array([c[3:] for c in prob_cols], dtype=object)
    proba = merged[prob_cols].to_numpy(dtype=float)

    if len(classes) != 34:
        problems.append(f"Expected 34 probability classes, got {len(classes)}")
    if len(merged) != 13600:
        problems.append(f"Expected 13600 test predictions, got {len(merged)}")

    row_sums = proba.sum(axis=1)
    max_prob_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    if max_prob_sum_error > 1e-5:
        problems.append(f"Probability rows do not sum to 1; max error={max_prob_sum_error}")

    y_true = merged["generator_true"].astype(str).to_numpy()
    y_pred = merged["generator_pred"].astype(str).to_numpy()

    recalculated = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "top1_accuracy": topk_accuracy(y_true, proba, classes, 1),
        "top3_accuracy": topk_accuracy(y_true, proba, classes, 3),
    }

    stored = metrics["test"]
    metric_deltas = {}
    for key, value in recalculated.items():
        delta = float(value - float(stored[key]))
        metric_deltas[key] = delta
        if abs(delta) > 1e-10:
            problems.append(f"Metric mismatch for {key}: delta={delta}")

    cm = pd.read_csv(run / "confusion_matrix.csv", index_col=0)
    if int(cm.to_numpy().sum()) != len(merged):
        problems.append("Confusion matrix total does not equal test sample count")

    confusions = []
    cm_classes = list(map(str, cm.index))
    arr = cm.to_numpy()
    for i, true_cls in enumerate(cm_classes):
        for j, pred_cls in enumerate(map(str, cm.columns)):
            if i != j and arr[i, j] > 0:
                confusions.append(
                    {"true": true_cls, "pred": pred_cls, "count": int(arr[i, j])}
                )
    confusions.sort(key=lambda x: x["count"], reverse=True)

    class_metrics = pd.read_csv(run / "class_metrics.csv")
    f1_col = next((c for c in class_metrics.columns if c.lower() in {"f1", "f1_score", "f1-score"}), None)
    label_col = next((c for c in class_metrics.columns if c.lower() in {"class", "label", "generator"}), None)

    worst_classes = []
    if f1_col is not None and label_col is not None:
        temp = class_metrics[[label_col, f1_col]].copy().sort_values(f1_col).head(10)
        worst_classes = [
            {"generator": str(r[label_col]), "f1": float(r[f1_col])}
            for _, r in temp.iterrows()
        ]

    correct_top1 = int(np.sum(y_true == y_pred))
    top3_hits = int(round(recalculated["top3_accuracy"] * len(y_true)))

    report = {
        "passed": not problems,
        "run_dir": str(run),
        "test_rows": int(len(merged)),
        "classes": int(len(classes)),
        "features": int(model_meta.get("n_features", -1)),
        "best_iteration": model_meta.get("best_iteration"),
        "stored_test_metrics": stored,
        "recalculated_test_metrics": recalculated,
        "metric_deltas": metric_deltas,
        "top1_correct": correct_top1,
        "top1_errors": int(len(y_true) - correct_top1),
        "top3_correct": top3_hits,
        "top3_errors": int(len(y_true) - top3_hits),
        "max_probability_sum_error": max_prob_sum_error,
        "largest_confusions": confusions[:15],
        "worst_classes_by_f1": worst_classes,
        "artifact_hashes": hashes,
        "problems": problems,
    }

    out = run / "run_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nValidation report: {out}")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
