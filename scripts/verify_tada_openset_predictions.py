import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/raw/tada_w2vbert2_layer4_knn21"
OUT = ROOT / "reproducibility/verification"

FOLD_HASHES = {
    "ruasd": "cac3bf00c0e8f77a1a2a37a4b3b005c1ce65ba3dbaf364d3b028dd5dde20171d",
    "mlaad": "eac341ab00a413ff1360ed4324316ff7822a056c4d02717e62b6c9691a66cde0",
}

SPLIT_HASHES = {
    "ruasd": "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0",
    "mlaad": "f4b66a0c8cbce8e7629d85838eb371d5f7381a9d2eafb63917e62b6c9691a66",
}

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def close(a, b):
    return abs(float(a) - float(b)) < 1e-7

def verify(dataset, fold):
    folder = SOURCE / dataset / "open_set"
    predictions = folder / f"fold{fold}_test_predictions.csv"
    metrics_file = folder / f"fold{fold}_metrics.json"

    if not predictions.is_file() or not metrics_file.is_file():
        raise FileNotFoundError(
            f"Missing prediction or metrics file: {dataset} fold {fold}"
        )

    df = pd.read_csv(predictions)
    meta = json.loads(metrics_file.read_text(encoding="utf-8"))
    saved = meta["metrics"]

    assert meta["dataset"] == dataset
    assert int(meta["fold"]) == fold
    assert meta["folds_sha256"] == FOLD_HASHES[dataset]
    assert meta["split_sha256"] == SPLIT_HASHES[dataset]

    required = ["id", "true_label", "prediction", "known_id_score"]
    assert set(required).issubset(df.columns)
    assert not df[required].isna().any().any()
    assert df["id"].is_unique

    known = set(meta["known_generators"])
    unknown = set(meta["unknown_generators"])
    assert known.isdisjoint(unknown)

    labels = set(df["true_label"])
    assert labels == known | unknown

    known_mask = df["true_label"].isin(known).to_numpy()
    unknown_mask = ~known_mask

    assert int(known_mask.sum()) == meta["n_known_test"]
    assert int(unknown_mask.sum()) == meta["n_unknown_test"]

    # The stored known-identity score is negative kNN distance.
    known_score = pd.to_numeric(
        df["known_id_score"], errors="raise"
    ).to_numpy(dtype=float)

    assert np.isfinite(known_score).all()

    # Higher distance = more likely an unknown generator.
    ood_score = -known_score
    y_ood = unknown_mask.astype(int)

    auroc = float(roc_auc_score(y_ood, ood_score))
    average_precision = float(
        average_precision_score(y_ood, ood_score)
    )

    precision, recall, _ = precision_recall_curve(
        y_ood, ood_score
    )
    pr_trapezoid = float(auc(recall, precision))

    # A candidate FPR95 definition:
    # known generators are the positive class.
    fpr, tpr, _ = roc_curve(
        known_mask.astype(int),
        known_score,
        drop_intermediate=False,
    )
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95_candidate = float(fpr[eligible[0]])

    # The threshold comes from known-generator validation.
    threshold = float(meta["distance_threshold_known_val_q95"])
    accepted = ood_score <= threshold

    known_accept = float(accepted[known_mask].mean())
    unknown_reject = float((~accepted[unknown_mask]).mean())

    checks = {
        "auroc": close(auroc, saved["auroc"]),
        "known_accept": close(
            known_accept, saved["known_accept"]
        ),
        "unknown_reject": close(
            unknown_reject, saved["unknown_reject"]
        ),
        "aupr_average_precision": close(
            average_precision, saved["aupr_ood"]
        ),
        "aupr_trapezoid": close(
            pr_trapezoid, saved["aupr_ood"]
        ),
        "fpr95_roc_candidate": close(
            fpr95_candidate, saved["fpr95"]
        ),
    }

    result = {
        "dataset": dataset,
        "fold": fold,
        "n_known": int(known_mask.sum()),
        "n_unknown": int(unknown_mask.sum()),
        "auroc_recomputed": auroc,
        "auroc_saved": saved["auroc"],
        "aupr_average_precision": average_precision,
        "aupr_trapezoid": pr_trapezoid,
        "aupr_saved": saved["aupr_ood"],
        "fpr95_roc_candidate": fpr95_candidate,
        "fpr95_saved": saved["fpr95"],
        "known_accept_recomputed": known_accept,
        "known_accept_saved": saved["known_accept"],
        "unknown_reject_recomputed": unknown_reject,
        "unknown_reject_saved": saved["unknown_reject"],
        "threshold": threshold,
        "predictions_sha256": file_hash(predictions),
        "metrics_sha256": file_hash(metrics_file),
        "checks": checks,
    }

    status = "PASS" if (
        checks["auroc"]
        and checks["known_accept"]
        and checks["unknown_reject"]
    ) else "REVIEW"

    print(
        f"{dataset.upper():5} fold {fold}: "
        f"AUROC {auroc:.8f} "
        f"(saved {saved['auroc']:.8f}); "
        f"AP {average_precision:.8f}; "
        f"FPR95 candidate {fpr95_candidate:.8f}; "
        f"{status}"
    )

    return result

def main():
    results = []

    for dataset in ("ruasd", "mlaad"):
        print(f"\n=== {dataset.upper()} ===")
        for fold in range(1, 6):
            results.append(verify(dataset, fold))

    OUT.mkdir(parents=True, exist_ok=True)

    report = {
        "purpose": "Independent TADA open-set metric diagnostics",
        "score_direction": "OOD score = -known_id_score",
        "auroc_positive_class": "unknown generators",
        "fpr95_candidate_positive_class": "known generators",
        "note": (
            "OSCR is not recalculated here. "
            "AUPR and FPR95 conventions must be confirmed "
            "against the original experiment implementation."
        ),
        "results": results,
    }

    json_path = OUT / "tada_openset_diagnostic.json"
    csv_path = OUT / "tada_openset_diagnostic.csv"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    flat = []
    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key != "checks"
        }
        row.update({
            "check_" + key: value
            for key, value in result["checks"].items()
        })
        flat.append(row)

    pd.DataFrame(flat).to_csv(csv_path, index=False)

    print("\n=== FINAL DIAGNOSTIC ===")
    print("Folds analyzed:", len(results))

    for key in (
        "auroc",
        "known_accept",
        "unknown_reject",
        "aupr_average_precision",
        "aupr_trapezoid",
        "fpr95_roc_candidate",
    ):
        matched = sum(r["checks"][key] for r in results)
        print(f"{key}: {matched}/{len(results)} matches")

    print("JSON:", json_path.relative_to(ROOT))
    print("CSV:", csv_path.relative_to(ROOT))

    core_ok = all(
        r["checks"]["auroc"]
        and r["checks"]["known_accept"]
        and r["checks"]["unknown_reject"]
        for r in results
    )

    if core_ok:
        print("CORE DIAGNOSTIC: PASS")
    else:
        print("CORE DIAGNOSTIC: REVIEW REQUIRED")

    return 0 if core_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
