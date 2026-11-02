#!/usr/bin/env python3
"""Read-only verification of saved RuASD / MLAAD open-set predictions (v3).

Read source artifacts only from --project-root. Write diagnostics only to
--output-dir, which MUST lie outside the source project tree.

The score convention is higher id_score = more likely known/ID.
The OSCR implementation reproduces the project's threshold and sorting
conventions, using cumulative counts instead of a quadratic loop.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

METRICS = (
    "auroc_id_vs_ood", "aupr_ood", "aupr_id", "fpr95", "oscr",
    "known_accept_rate", "unknown_reject_rate", "macro_f1_with_unknown",
)
REQUIRED = (
    "generator", "method", "fold", "is_known_true", "generator_pred",
    "id_score", "accepted_known",
)


def as_bool(series, name):
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name}: missing booleans")
        return series.to_numpy(dtype=bool)
    convert = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.strip().str.lower().map(convert)
    if converted.isna().any():
        raise ValueError(f"{name}: invalid boolean value(s)")
    return converted.to_numpy(dtype=bool)


def oscr_fast_equivalent(known_truth, known_pred, known_scores, unknown_scores):
    """Same threshold and argsort(FPR) semantics as source_attribution.metrics.oscr."""
    sk = np.asarray(known_scores, dtype=float)
    su = np.asarray(unknown_scores, dtype=float)
    correct = (np.asarray(known_truth).astype(str) == np.asarray(known_pred).astype(str))
    if not len(sk) or not len(su):
        raise ValueError("OSCR requires both known and unknown test samples")
    order = np.argsort(sk)
    sorted_sk = sk[order]
    cumulative = np.r_[0, np.cumsum(correct[order].astype(np.int64))]
    sorted_su = np.sort(su)
    thresholds = np.r_[np.inf, np.unique(np.r_[sk, su])[::-1], -np.inf]
    known_below = np.searchsorted(sorted_sk, thresholds, side="left")
    unknown_below = np.searchsorted(sorted_su, thresholds, side="left")
    ccr = (cumulative[-1] - cumulative[known_below]) / len(sk)
    fpr = (len(su) - unknown_below) / len(su)
    sort_fpr = np.argsort(fpr)
    return float(np.trapezoid(ccr[sort_fpr], fpr[sort_fpr]))


def recompute(frame):
    true_known = as_bool(frame["is_known_true"], "is_known_true")
    accepted = as_bool(frame["accepted_known"], "accepted_known")
    scores = pd.to_numeric(frame["id_score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite ID scores")
    n_known = int(true_known.sum())
    n_unknown = int((~true_known).sum())
    if not n_known or not n_unknown:
        raise ValueError("Missing known or unknown test examples")

    truth = frame["generator"].astype(str).to_numpy()
    prediction = frame["generator_pred"].astype(str).to_numpy()
    if frame[["generator", "generator_pred"]].isna().any().any():
        raise ValueError("Null true or predicted generator")
    y_id = true_known.astype(int)
    fpr, tpr, _ = roc_curve(y_id, scores, pos_label=1)
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[eligible[0]]) if len(eligible) else 1.0

    true_with_unknown = np.where(true_known, truth, "unknown")
    pred_with_unknown = np.where(accepted, prediction, "unknown")
    computed = {
        "auroc_id_vs_ood": float(roc_auc_score(y_id, scores)),
        "aupr_ood": float(average_precision_score(1 - y_id, -scores)),
        "aupr_id": float(average_precision_score(y_id, scores)),
        "fpr95": fpr95,
        "oscr": oscr_fast_equivalent(truth[true_known], prediction[true_known], scores[true_known], scores[~true_known]),
        "known_accept_rate": float(accepted[true_known].mean()),
        "unknown_reject_rate": float((~accepted[~true_known]).mean()),
        "macro_f1_with_unknown": float(f1_score(true_with_unknown, pred_with_unknown, average="macro")),
    }
    return computed, n_known, n_unknown, true_known, accepted, scores


def iter_runs(root):
    base = root / "results" / "raw"
    for experiment in ("open_set_acoustic", "open_set_acoustic_fixed4_lr_C1"):
        parent = base / experiment
        if not parent.is_dir():
            yield (experiment, None, None, None, f"Missing directory: {parent}")
            continue
        folders = sorted(p for p in parent.glob("*seed*") if p.is_dir())
        if not folders:
            yield (experiment, None, None, None, f"No seed directories: {parent}")
        for folder in folders:
            yield (experiment, folder, folder / "metrics.json", folder / "predictions.parquet", None)

    mlaad_folders = sorted(p for p in base.glob("mlaad_23class_openset_fold*_seed*") if p.is_dir())
    if not mlaad_folders:
        yield ("mlaad_23class_openset", None, None, None, "Missing MLAAD open-set fold/seed directories")
    for folder in mlaad_folders:
        yield ("mlaad_23class_openset", folder, folder / "metrics.json", folder / "predictions.parquet", None)


def inspect_run(experiment, folder, metrics_path, predictions_path, tol, output, root):
    meta = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    columns = pd.read_parquet(predictions_path)
    missing = sorted(set(REQUIRED) - set(columns.columns))
    if missing:
        raise ValueError(f"Missing prediction columns {missing}")
    if "file_id" in columns:
        id_col = "file_id"
    elif "path" in columns:
        id_col = "path"
    else:
        raise ValueError("Neither file_id nor path identifier was found")

    columns["_fold_key"] = columns["fold"].astype(str).str.replace(r"\.0$", "", regex=True)
    columns["_method_key"] = columns["method"].astype(str)

    mlaad_match = re.fullmatch(r"mlaad_23class_openset_fold(\d+)_seed(\d+)", folder.name)
    if experiment == "mlaad_23class_openset" and not mlaad_match:
        raise ValueError(f"Unexpected MLAAD folder name: {folder.name}")
    if mlaad_match:
        expected_fold = mlaad_match.group(1)
        if set(columns["_fold_key"]) != {expected_fold}:
            raise ValueError(f"MLAAD folder fold {expected_fold} inconsistent with predictions")
        if "seed" in columns and set(columns["seed"].astype(str)) != {mlaad_match.group(2)}:
            raise ValueError("MLAAD seed inconsistent with predictions")
        if "split" in columns and set(columns["split"].astype(str)) != {"test"}:
            raise ValueError("MLAAD predictions contain non-test records")

    grouped = columns.groupby(["_fold_key", "_method_key"], sort=True, dropna=False)
    expected_groups = 4 if mlaad_match else 20
    if len(grouped) != expected_groups:
        raise ValueError(f"Expected {expected_groups} fold/method groups; got {len(grouped)}")

    expected_rows = 4600 if mlaad_match else 13600
    seen = 0
    passed = 0
    for (fold, method), frame in grouped:
        seen += 1
        if mlaad_match:
            method_saved = meta.get(method)
            mlaad_fold_meta = json.loads(
                (root / "splits" / "mlaad_23class_open_set_folds_v1.json").read_text(encoding="utf-8-sig")
            )["folds"][str(fold)]
            known_set = set(mlaad_fold_meta["known_generators"])
            unknown_set = set(mlaad_fold_meta["unknown_generators"])
        else:
            fold_saved = meta.get(str(fold), {})
            method_saved = fold_saved.get("methods", {}).get(method)
            known_set = fold_saved.get("known_generators")
            unknown_set = fold_saved.get("unknown_generators")
        if not isinstance(method_saved, dict):
            raise ValueError(f"Missing metrics for fold={fold}, method={method}")
        if len(frame) != expected_rows:
            raise ValueError(f"fold={fold}, method={method}: {len(frame)} rows, expected {expected_rows}")
        if frame[id_col].isna().any() or frame[id_col].duplicated().any():
            raise ValueError(f"fold={fold}, method={method}: null/duplicate identifiers")
        calc, nk, nu, known_mask, accepted, scores = recompute(frame)
        # Generator-disjoint folds have *variable* sizes: RuASD fold 5
        # contains 28/6 ID/OOD generators, MLAAD folds 4-5 contain 19/4.
        # Check against the frozen per-fold membership, not a global 27/7
        # or 18/5 constant.
        if known_set is None or unknown_set is None:
            raise ValueError(f"fold={fold}: missing frozen known/unknown generator lists")
        if experiment.startswith("open_set_acoustic"):
            canonical = json.loads(
                (root / "splits" / "open_set_folds.json").read_text(encoding="utf-8-sig")
            )["folds"][str(fold)]
            if (set(known_set) != set(canonical["known_generators"]) or
                set(unknown_set) != set(canonical["unknown_generators"])):
                raise ValueError(f"fold={fold}: metrics known/unknown lists differ from frozen folds")
        known_set = set(map(str, known_set))
        unknown_set = set(map(str, unknown_set))
        if known_set & unknown_set:
            raise ValueError(f"fold={fold}: overlap between known and unknown generator lists")
        actual_generators = frame["generator"].astype(str)
        if set(actual_generators) != known_set | unknown_set:
            raise ValueError(f"fold={fold}: generator identities disagree with frozen fold metadata")
        if not np.array_equal(known_mask, actual_generators.isin(known_set).to_numpy()):
            raise ValueError(f"fold={fold}: is_known_true inconsistent with frozen generator membership")
        # Per-generator TEST quotas come from the immutable split metadata.
        # RuASD: 400 per generator; MLAAD: 200 per generator.
        # Using the RuASD quota for MLAAD incorrectly doubles its counts.
        if mlaad_match:
            split_meta = json.loads(
                (root / "splits" / "mlaad_ru_23class_split_v1.meta.json").read_text(encoding="utf-8-sig")
            )
            per_generator = int(split_meta["target_per_generator"]["test"])
        else:
            split_meta = json.loads(
                (root / "splits" / "frozen_split_validation.json").read_text(encoding="utf-8-sig")
            )
            per_generator = int(split_meta["per_generator_quota"]["test"])
        if per_generator <= 0:
            raise ValueError("Frozen split reports a nonpositive per-generator test quota")
        expected_known = per_generator * len(known_set)
        expected_unknown = per_generator * len(unknown_set)
        if (nk, nu) != (expected_known, expected_unknown):
            raise ValueError(
                f"fold={fold}, method={method}: actual ID/OOD counts {nk}/{nu}; "
                f"frozen expected {expected_known}/{expected_unknown}"
            )
        class_counts = actual_generators.value_counts()
        if not class_counts.eq(per_generator).all():
            raise ValueError(
                f"fold={fold}, method={method}: per-generator test counts "
                f"do not all equal frozen quota {per_generator}"
            )

        checks = {}
        row = {"experiment": experiment, "run": folder.name, "fold": fold,
               "method": method, "rows": len(frame), "n_known": nk, "n_unknown": nu}
        for name in METRICS:
            original = method_saved.get(name)
            if original is None:
                checks[name] = "MISSING"
                row[f"saved_{name}"] = None
                row[f"delta_{name}"] = None
            else:
                delta = calc[name] - float(original)
                checks[name] = "PASS" if abs(delta) <= tol else "FAIL"
                row[f"saved_{name}"] = float(original)
                row[f"delta_{name}"] = delta
            row[f"computed_{name}"] = calc[name]
            row[f"check_{name}"] = checks[name]
        if "threshold" in method_saved:
            threshold = float(method_saved["threshold"])
            row["threshold_disagreements"] = int(np.count_nonzero(accepted != (scores >= threshold)))
        else:
            row["threshold_disagreements"] = None
        row["status"] = "PASS" if all(x == "PASS" for x in checks.values()) else "REVIEW"
        output.append(row)
        passed += row["status"] == "PASS"
    return seen, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"Project root does not exist: {root}")
    if out == root or root in out.parents:
        parser.error("Output directory MUST be outside PUBLIC_RELEASE")
    print("Source project (read only):", root, flush=True)
    print("Output directory:", out, flush=True)
    output = []
    errors = []
    n_runs = 0
    for experiment, folder, metrics_path, predictions_path, issue in iter_runs(root):
        if issue:
            errors.append(issue)
            print("ERROR:", issue, flush=True)
            continue
        n_runs += 1
        if not metrics_path.is_file() or not predictions_path.is_file():
            msg = f"{experiment}/{folder.name}: missing metrics or predictions"
            errors.append(msg)
            print("ERROR:", msg, flush=True)
            continue
        try:
            seen, passed = inspect_run(experiment, folder, metrics_path, predictions_path, args.tolerance, output, root)
            print(f"{experiment}/{folder.name}: {passed}/{seen} fold-method groups PASS", flush=True)
        except Exception as e:
            msg = f"{experiment}/{folder.name}: {type(e).__name__}: {e}"
            errors.append(msg)
            print("ERROR:", msg, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(output)
    json_path = out / "acoustic_openset_verification.json"
    csv_path = out / "acoustic_openset_verification.csv"
    report = {"project_root": str(root), "runs_checked": n_runs, "groups_checked": len(output),
              "groups_passed": sum(x["status"] == "PASS" for x in output),
              "metrics_compared": list(METRICS), "tolerance": args.tolerance,
              "results": output, "errors": errors}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    df.to_csv(csv_path, index=False)
    print("\n=== FINAL VERIFICATION ===")
    print("Runs attempted:", n_runs)
    print("Fold-method groups:", report["groups_checked"])
    print("All-metrics PASS groups:", report["groups_passed"])
    for metric in METRICS:
        count = Counter(row[f"check_{metric}"] for row in output)
        print(f"{metric}: {count['PASS']}/{len(output)} PASS, {count['FAIL']} FAIL, {count['MISSING']} MISSING")
    thresh = [x for x in output if x["threshold_disagreements"] is not None]
    if thresh:
        print("Threshold/accepted mismatches across groups:", sum(x["threshold_disagreements"] for x in thresh))
    if output:
        bad = [row for row in output if row["status"] != "PASS"]
        for row in bad[:12]:
            failures = [f"{name}={row['check_'+name]} (delta={row['delta_'+name]})" for name in METRICS if row["check_"+name] != "PASS"]
            print("REVIEW", row["run"], "fold", row["fold"], row["method"], "; ".join(failures))
        if len(bad) > 12:
            print("Additional REVIEW groups:", len(bad)-12, "(see CSV)")
    print("Errors:", len(errors))
    for e in errors[:15]:
        print("ERROR:", e)
    print("JSON:", json_path)
    print("CSV:", csv_path)
    ok = (not errors and len(output) > 0 and report["groups_passed"] == len(output))
    print("FINAL STATUS:", "PASS" if ok else "REVIEW REQUIRED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
