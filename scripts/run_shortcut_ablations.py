#!/usr/bin/env python
from __future__ import annotations

import argparse
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

EXPECTED_FEATURE_SHA = "a21d537c5133a7e6daa852138f96ee7f8bf1491036cbf940da6451f688adaad8"
EXPECTED_SPLIT_SHA = "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pre-specified cheap shortcut ablations on the frozen RuASD feature store."
    )
    p.add_argument("--features", default="features/ruasd_closed_set_acoustic.parquet")
    p.add_argument("--splits", default="splits/closed_set.csv")
    p.add_argument("--config", default="configs/shortcut_ablations.yaml")
    return p.parse_args()


def choose_columns(feat: pd.DataFrame, spec: dict) -> list[str]:
    if spec["mode"] == "explicit":
        cols = list(spec["columns"])
    elif spec["mode"] == "acoustic_prefixes":
        prefixes = tuple(spec["include_prefixes"])
        exclude = set(spec.get("exclude_columns", []))
        cols = [c for c in feat.columns if c.startswith(prefixes) and c not in exclude]
    else:
        raise ValueError(f"Unknown mode: {spec['mode']}")
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        raise RuntimeError(f"Missing requested columns: {missing}")
    if not cols:
        raise RuntimeError("No features selected")
    return cols


def run_one(name: str, description: str, df: pd.DataFrame, cols: list[str], cfg: dict,
            split_sha: str, feature_sha: str) -> dict:
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    tr = df["split"].eq("train")
    va = df["split"].eq("validation")
    te = df["split"].eq("test")
    X = df[cols].replace([np.inf, -np.inf], np.nan)

    mc = cfg["model"]
    model = build_and_train(
        X.loc[tr], df.loc[tr, "generator"],
        X.loc[va], df.loc[va, "generator"],
        mc["type"], mc.get("params", {}),
        int(cfg.get("training", {}).get("early_stopping_rounds", 80)),
        seed,
    )

    val_proba = model.predict_proba(X.loc[va])
    val_pred = model.predict(X.loc[va])
    val_true = df.loc[va, "generator"].astype(str).to_numpy()

    test_proba = model.predict_proba(X.loc[te])
    test_pred = model.predict(X.loc[te])
    test_true = df.loc[te, "generator"].astype(str).to_numpy()

    val_metrics = closed_set_metrics(val_true, val_pred, val_proba, model.classes_)
    test_metrics = closed_set_metrics(test_true, test_pred, test_proba, model.classes_)

    run_cfg = {
        "experiment": {
            "name": f"shortcut_{name}",
            "seed": seed,
            "output_root": cfg["experiment"]["output_root"],
        },
        "description": description,
        "selected_columns": cols,
        "model": mc,
        "training": cfg.get("training", {}),
        "frozen_artifacts": {
            "features_sha256": feature_sha,
            "splits_sha256": split_sha,
        },
    }
    run = create_run_dir(
        cfg["experiment"]["output_root"],
        f"shortcut_{name}",
        seed,
    )
    record_run_context(run, run_cfg)

    json_dump(
        {
            "experiment": name,
            "description": description,
            "seed": seed,
            "n_features": len(cols),
            "validation": val_metrics,
            "test": test_metrics,
        },
        run / "metrics.json",
    )
    json_dump(
        {
            "features_sha256": feature_sha,
            "splits_sha256": split_sha,
            "selected_columns": cols,
        },
        run / "artifact_hashes.json",
    )

    pd.DataFrame(per_class_metrics(test_true, test_pred)).to_csv(
        run / "class_metrics.csv", index=False, encoding="utf-8-sig"
    )

    cm = confusion_matrix(test_true, test_pred, labels=model.classes_)
    pd.DataFrame(cm, index=model.classes_, columns=model.classes_).to_csv(
        run / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    predictions = df.loc[te, ["file_id", "generator", "text_hash"]].copy().reset_index(drop=True)
    predictions = predictions.rename(columns={"generator": "generator_true"})
    predictions["generator_pred"] = test_pred
    predictions["dataset"] = "RuASD"
    predictions["distortion"] = "clean"
    predictions["seed"] = seed
    predictions.to_parquet(run / "predictions.parquet", index=False)

    prob_df = pd.DataFrame({"file_id": predictions["file_id"].astype(str)})
    for i, cls in enumerate(model.classes_):
        prob_df[f"p::{cls}"] = test_proba[:, i].astype(np.float32)
    prob_df.to_parquet(run / "probabilities.parquet", index=False)

    (run / "classes.txt").write_text("\n".join(map(str, model.classes_)), encoding="utf-8")
    (run / "feature_columns.txt").write_text("\n".join(cols), encoding="utf-8")
    joblib.dump(model, run / "model.joblib")

    estimator = model.estimator
    json_dump(
        {
            "kind": model.kind,
            "seed": seed,
            "n_features": len(cols),
            "best_iteration": (
                int(estimator.best_iteration_)
                if getattr(estimator, "best_iteration_", None) is not None
                else None
            ),
        },
        run / "model_metadata.json",
    )

    return {
        "experiment": name,
        "n_features": len(cols),
        "validation_macro_f1": float(val_metrics["macro_f1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_top1_accuracy": float(test_metrics["top1_accuracy"]),
        "test_top3_accuracy": float(test_metrics["top3_accuracy"]),
        "run_dir": str(run),
    }


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features)
    split_path = Path(args.splits)
    config_path = Path(args.config)

    feature_sha = file_sha256(feature_path)
    split_sha = file_sha256(split_path)
    if feature_sha != EXPECTED_FEATURE_SHA:
        raise RuntimeError(f"Frozen feature store changed: {feature_sha}")
    if split_sha != EXPECTED_SPLIT_SHA:
        raise RuntimeError(f"Frozen split changed: {split_sha}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sp = pd.read_csv(split_path, low_memory=False)
    assert_no_leakage(sp)
    feat = pd.read_parquet(feature_path)

    needed = ["file_id", "duration", "sample_rate", "channels", "prosody_duration_sec"]
    acoustic = [c for c in feat.columns if c.startswith("spectral_") or c.startswith("prosody_")]
    feat_small = feat[list(dict.fromkeys(needed + acoustic))].copy()

    df = sp[["file_id", "split", "generator", "text_hash"]].merge(
        feat_small, on="file_id", how="left", validate="one_to_one"
    )
    if len(df) != 68000:
        raise RuntimeError(f"Expected 68000 rows, got {len(df)}")

    baseline_dir = Path(cfg["baseline"]["run_dir"])
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))

    rows = [{
        "experiment": "canonical_790",
        "n_features": int(baseline_metrics["n_features"]),
        "validation_macro_f1": float(baseline_metrics["validation"]["macro_f1"]),
        "test_macro_f1": float(baseline_metrics["test"]["macro_f1"]),
        "test_balanced_accuracy": float(baseline_metrics["test"]["balanced_accuracy"]),
        "test_top1_accuracy": float(baseline_metrics["test"]["top1_accuracy"]),
        "test_top3_accuracy": float(baseline_metrics["test"]["top3_accuracy"]),
        "run_dir": str(baseline_dir),
    }]

    for name, spec in cfg["ablations"].items():
        cols = choose_columns(feat, spec)
        print(f"\n=== {name} ===")
        print(f"Features: {len(cols)}")
        result = run_one(
            name, spec["description"], df, cols, cfg, split_sha, feature_sha
        )
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    summary = pd.DataFrame(rows)
    baseline_f1 = float(
        summary.loc[summary["experiment"].eq("canonical_790"), "test_macro_f1"].iloc[0]
    )
    summary["delta_macro_f1_vs_canonical"] = summary["test_macro_f1"] - baseline_f1

    summary_root = Path(cfg["experiment"]["summary_root"])
    summary_root.mkdir(parents=True, exist_ok=True)
    csv_path = summary_root / "shortcut_ablations_seed42.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md = [
        "# Shortcut ablation report",
        "",
        "Frozen clean RuASD closed-set protocol, seed 42.",
        "",
        f"- Feature store SHA-256: `{feature_sha}`",
        f"- Split SHA-256: `{split_sha}`",
        "",
        "## Results",
        "",
    ]
    for _, r in summary.iterrows():
        md.append(
            f"- {r['experiment']}: n_features={int(r['n_features'])}, "
            f"test Macro-F1={r['test_macro_f1']:.6f}, "
            f"BA={r['test_balanced_accuracy']:.6f}, "
            f"Top-1={r['test_top1_accuracy']:.6f}, "
            f"Top-3={r['test_top3_accuracy']:.6f}, "
            f"ΔMacro-F1={r['delta_macro_f1_vs_canonical']:.6f}"
        )
    md += [
        "",
        "## Interpretation rule",
        "",
        "Metadata-only and sample-rate-only results are diagnostics of corpus-level shortcut risk, "
        "not evidence of acoustic generator fingerprints.",
        "",
        "Peak-normalization and crop/full-duration ablations require separate feature extraction "
        "and are intentionally not part of this cheap pass.",
        "",
    ]
    report_path = summary_root / "shortcut_report_seed42.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSummary CSV: {csv_path}")
    print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()
