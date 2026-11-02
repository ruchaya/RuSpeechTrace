#!/usr/bin/env python3
"""Read-only cross-dataset reproducibility audit for the PUBLIC_RELEASE project.

Recomputes metrics from the stored predictions for five seeds, two transfer
orientations, three scoring methods, two target subsets; checks published
per-seed and mean/sd CSV summaries. Does not load models or modify the project.

Usage:
  python verify_cross_dataset_external.py \
      --project-root 'S:\\OpenSet_Source_Attribution_Project_PUBLIC_RELEASE' \
      --output-dir 'S:\\PUBLIC_RELEASE_SMOKE\\cross_dataset_verification'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

SEEDS = (42, 123, 456, 789, 2026)
DIRECTIONS = ("RuASD_to_MLAAD", "MLAAD_to_RuASD")
METHODS = ("msp", "energy", "sme")
SUBSETS = ("full_external", "frozen_test_external")
FAMILIES = ("Bark", "XTTS-v2", "MMS-TTS")
METRICS = (
    "auroc_id_vs_ood", "aupr_ood", "aupr_id", "fpr95", "oscr",
    "n_total", "n_shared", "n_unmapped", "macro_f1_four_labels",
    "shared_accept_rate", "shared_correct_attribution_rate",
    "unmapped_rejection_rate", "unmapped_false_accept_rate",
    "accepted_unmapped_source_label_count",
)
EPS_ABS = 2e-9
EPS_REL = 2e-10


def equal(actual, expected):
    if isinstance(actual, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        return int(actual) == int(expected)
    return bool(np.isclose(float(actual), float(expected), atol=EPS_ABS, rtol=EPS_REL))


def oscr_fast(true, predicted, known_score, unknown_score):
    """O(N log N) equivalence of src/source_attribution/metrics.py::oscr.

    Preserve its thresholds and its np.argsort order for coincident FPR values.
    """
    sk = np.asarray(known_score, dtype=float)
    su = np.asarray(unknown_score, dtype=float)
    correct = (np.asarray(true).astype(str) == np.asarray(predicted).astype(str))
    order = np.argsort(sk)
    sorted_known = sk[order]
    cum_correct = np.r_[0, np.cumsum(correct[order].astype(np.int64), dtype=np.int64)]
    sorted_unknown = np.sort(su)
    thresholds = np.r_[np.inf, np.unique(np.r_[sk, su])[::-1], -np.inf]
    known_lt = np.searchsorted(sorted_known, thresholds, side="left")
    unknown_lt = np.searchsorted(sorted_unknown, thresholds, side="left")
    ccr = (cum_correct[-1] - cum_correct[known_lt]) / len(sk)
    fpr = (len(su) - unknown_lt) / len(su)
    sorting = np.argsort(fpr)
    return float(np.trapezoid(ccr[sorting], fpr[sorting]))


def compute(frame, saved_threshold):
    truth = frame["target_family_or_unknown"].astype(str).to_numpy()
    mapped = frame["mapped_source_prediction"].astype(str).to_numpy()
    recorded_final = frame["final_prediction"].astype(str).to_numpy()
    score = pd.to_numeric(frame["id_score"], errors="raise").to_numpy(dtype=float)
    per_record_thr = pd.to_numeric(frame["threshold"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(score).all() and np.isfinite(per_record_thr).all()):
        raise ValueError("Nonfinite scores or thresholds")
    if not np.all(np.isclose(per_record_thr, saved_threshold, atol=EPS_ABS, rtol=EPS_REL)):
        raise ValueError("Prediction thresholds disagree with metrics.json")
    shared = truth != "unknown"
    if not shared.any() or shared.all():
        raise ValueError("Expected both shared and unmapped target families")
    if set(truth) != set(FAMILIES) | {"unknown"}:
        raise ValueError(f"Unexpected target labels: {set(truth)}")
    recorded_shared = frame["is_shared_family"].to_numpy()
    if not np.array_equal(recorded_shared.astype(bool), shared):
        raise ValueError("is_shared_family disagrees with labels")
    accepted = score >= saved_threshold
    if not np.array_equal(frame["accepted"].to_numpy().astype(bool), accepted):
        raise ValueError("accepted disagrees with score threshold")
    final = np.where(accepted, mapped, "unknown")
    if not np.array_equal(recorded_final, final):
        raise ValueError("final_prediction disagrees with acceptance/mapped labels")
    y_id = shared.astype(int)
    fpr, tpr, _ = roc_curve(y_id, score, pos_label=1)
    eligible = np.flatnonzero(tpr >= .95)
    data = {
        "auroc_id_vs_ood": float(roc_auc_score(y_id, score)),
        "aupr_ood": float(average_precision_score(1 - y_id, -score)),
        "aupr_id": float(average_precision_score(y_id, score)),
        "fpr95": float(fpr[eligible[0]]) if len(eligible) else 1.,
        "oscr": oscr_fast(truth[shared], mapped[shared], score[shared], score[~shared]),
        "n_total": int(len(truth)),
        "n_shared": int(shared.sum()),
        "n_unmapped": int((~shared).sum()),
        "macro_f1_four_labels": float(f1_score(truth, final, labels=list(FAMILIES) + ["unknown"], average="macro", zero_division=0)),
        "shared_accept_rate": float(accepted[shared].mean()),
        "shared_correct_attribution_rate": float((final[shared] == truth[shared]).mean()),
        "unmapped_rejection_rate": float((~accepted[~shared]).mean()),
        "unmapped_false_accept_rate": float(accepted[~shared].mean()),
        "accepted_unmapped_source_label_count": int(np.sum(accepted & (mapped == "unmapped_source_prediction"))),
    }
    family = {}
    for name in FAMILIES:
        mask = truth == name
        family[name] = {
            "n": int(mask.sum()),
            "correct_attribution_recall": float((final[mask] == name).mean()),
            "accept_rate": float(accepted[mask].mean()),
        }
    return data, family


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_table(computed, original_path, keys, test_cols=None):
    """Compare independently reconstructed rows with archived CSV values."""
    existing = pd.read_csv(original_path)
    actual = computed.copy()
    if len(existing) != len(actual):
        raise ValueError(f"{original_path.name}: {len(existing)} rows vs {len(actual)}")
    target_cols = list(test_cols or existing.columns)
    missing = sorted(set(target_cols) - set(actual.columns))
    if missing:
        raise ValueError(f"{original_path.name}: missing columns {missing}")
    merged = existing.merge(actual, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_saved", "_computed"))
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{original_path.name}: differing grouping keys")
    checks = 0
    for column in target_cols:
        if column in keys:
            continue
        a = merged[column + "_saved"]
        b = merged[column + "_computed"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a.to_numpy(float), b.to_numpy(float), atol=EPS_ABS, rtol=EPS_REL, equal_nan=True)
        else:
            same = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        if not same.all():
            bad = int((~same).sum())
            raise ValueError(f"{original_path.name}: {column} mismatched in {bad} row(s)")
        checks += len(merged)
    return checks


def summarize(df, keys):
    excluded = {"seed", "n", "n_total", "n_shared", "n_unmapped"}
    numerics = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    result = df.groupby(keys, sort=True)[numerics].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(str(v) for v in col if v) if isinstance(col, tuple) else col for col in result.columns]
    return result


def check_run(project, seed):
    raw = project / "results" / "raw"
    folder = raw / "cross_dataset_open_set_full_seed42" if seed == 42 else raw / "cross_dataset_open_set_multiseed_5seeds" / f"seed{seed}"
    pred_path, metrics_path = folder / "predictions.parquet", folder / "metrics.json"
    if not pred_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing predictions/metrics for seed {seed}: {folder}")
    saved = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    df = pd.read_parquet(pred_path)
    cols = ["direction", "record_id", "target_split", "target_generator", "target_family_or_unknown", "is_shared_family", "source_model_prediction", "mapped_source_prediction", "method", "id_score", "threshold", "accepted", "final_prediction"]
    absent = sorted(set(cols) - set(df.columns))
    if absent:
        raise ValueError(f"Missing columns in {pred_path}: {absent}")
    if df[cols].isna().any().any():
        raise ValueError("Null in prediction columns")
    if len(df) != 273000:
        raise ValueError(f"Expected 273000 prediction rows, found {len(df)}")
    if set(df["direction"]) != set(DIRECTIONS) or set(df["method"]) != set(METHODS):
        raise ValueError("Unexpected directions or methods")
    rows, families = [], []
    for direction in DIRECTIONS:
        exp_n = 23000 if direction == "RuASD_to_MLAAD" else 68000
        exp_test = 4600 if direction == "RuASD_to_MLAAD" else 13600
        if direction not in saved:
            raise ValueError(f"Missing direction {direction} in metrics")
        for method in METHODS:
            group = df.loc[(df["direction"] == direction) & (df["method"] == method)]
            if len(group) != exp_n or group["record_id"].duplicated().any():
                raise ValueError(f"Unexpected row count / duplicate IDs: {seed} {direction} {method}")
            if int(group["target_split"].eq("test").sum()) != exp_test:
                raise ValueError(f"Unexpected frozen test size: {seed} {direction} {method}")
            saved_method = saved[direction]["methods"][method]
            for subset in SUBSETS:
                sub = group if subset == "full_external" else group.loc[group["target_split"].eq("test")]
                recomputed, recomputed_family = compute(sub, float(saved_method["threshold"]))
                published = saved_method[subset]
                for key in METRICS:
                    if not equal(recomputed[key], published[key]):
                        raise ValueError(f"{seed} {direction} {method} {subset} {key}: computed {recomputed[key]} saved {published[key]}")
                for family, values in recomputed_family.items():
                    for key, value in values.items():
                        if not equal(value, published["per_shared_family"][family][key]):
                            raise ValueError(f"{seed} {direction} {method} {subset} family {family} {key}: computed {value} saved {published['per_shared_family'][family][key]}")
                    families.append({"seed": seed, "direction": direction, "method": method, "subset": subset, "family": family, **values})
                rows.append({"seed": seed, "direction": direction, "method": method, "subset": subset,
                             "source_test_macro_f1": saved[direction]["source_in_domain_test_macro_f1"],
                             "threshold": saved_method["threshold"],
                             "source_validation_accept_rate": saved_method["source_validation_accept_rate"],
                             "source_test_accept_rate": saved_method["source_test_accept_rate"],
                             **recomputed})
    return rows, families, {"seed": seed, "predictions_sha256": sha256(pred_path), "metrics_sha256": sha256(metrics_path), "checked_groups": 12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dest = args.output_dir.resolve()
    if not root.is_dir():
        parser.error("Project root not found")
    if dest == root or root in dest.parents:
        parser.error("Output must be OUTSIDE the project root")
    print("Read-only project:", root, flush=True)
    print("Output outside project:", dest, flush=True)
    all_rows, all_families, meta, errors = [], [], [], []
    for seed in SEEDS:
        try:
            rows, families, item = check_run(root, seed)
            all_rows.extend(rows)
            all_families.extend(families)
            meta.append(item)
            print(f"Seed {seed}: {len(rows)}/12 groups PASS", flush=True)
        except Exception as e:
            msg = f"Seed {seed}: {type(e).__name__}: {e}"
            errors.append(msg)
            print("ERROR:", msg, flush=True)
    checks = {}
    if not errors and len(all_rows) == 60:
        base = root / "results" / "raw" / "cross_dataset_open_set_multiseed_5seeds"
        metric_rows = pd.DataFrame(all_rows)
        family_rows = pd.DataFrame(all_families)
        comparisons = (
            ("metrics_by_seed", metric_rows, ["seed", "direction", "method", "subset"], None),
            ("family_metrics_by_seed", family_rows, ["seed", "direction", "method", "subset", "family"], None),
            ("metrics_mean_sd", summarize(metric_rows, ["direction", "method", "subset"]), ["direction", "method", "subset"], None),
            ("family_metrics_mean_sd", summarize(family_rows, ["direction", "method", "subset", "family"]), ["direction", "method", "subset", "family"], None),
        )
        for name, frame, key, cols in comparisons:
            try:
                checks[name] = {"status": "PASS", "cells": compare_table(frame, base / (name + ".csv"), key, cols)}
                print("AGGREGATE PASS:", name, flush=True)
            except Exception as e:
                checks[name] = {"status": "FAIL", "error": str(e)}
                errors.append(f"Aggregate {name}: {e}")
                print("AGGREGATE FAIL:", name, e, flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "Independent cross-dataset prediction verification", "seeds": list(SEEDS),
              "expected_fold_method_subset_groups": 60, "completed_groups": len(all_rows),
              "metrics_verified_per_group": list(METRICS), "families": list(FAMILIES),
              "source": "Stored individual predictions and metrics JSON; no model training", "inputs_sha256": meta,
              "aggregate_checks": checks, "errors": errors,
              "status": "PASS" if not errors and len(all_rows) == 60 and len(checks) == 4 else "REVIEW_REQUIRED"}
    out = dest / "cross_dataset_verification.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(dest / "cross_dataset_recomputed.csv", index=False)
    pd.DataFrame(all_families).to_csv(dest / "cross_dataset_family_recomputed.csv", index=False)
    print("\n=== FINAL VERIFICATION ===", flush=True)
    print(f"Groups: {len(all_rows)}/60", flush=True)
    print(f"Summary CSV files: {sum(c['status']=='PASS' for c in checks.values())}/4 PASS", flush=True)
    print("Errors:", len(errors), flush=True)
    print("Report:", out, flush=True)
    print("FINAL STATUS:", report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
