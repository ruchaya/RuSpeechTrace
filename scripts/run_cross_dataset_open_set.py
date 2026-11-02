
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from source_attribution.ood import (
    msp_id_score,
    energy_id_score,
    sme_id_score,
    fit_known_quantile_threshold,
)
from source_attribution.metrics import open_set_metrics


ROOT = Path(".")
OUT = Path("results/raw/cross_dataset_open_set_full_seed42")
MAPPING_PATH = Path(
    "data_inventory/family_mapping_ruasd_mlaad_three_family_candidate.csv"
)

FAMILIES = ("Bark", "XTTS-v2", "MMS-TTS")
METHODS = ("msp", "energy", "sme")
PRIVATE_PREDICTION = "unmapped_source_prediction"

DOMAINS = {
    "RuASD": {
        "split": Path("splits/closed_set.csv"),
        "features": Path("features/ruasd_closed_set_acoustic.parquet"),
        "meta": Path("features/ruasd_closed_set_acoustic.meta.json"),
        "key": "file_id",
        "model": Path(
            "results/raw/shortcut_acoustic_no_duration/"
            "20260928T091631Z_seed42/model.joblib"
        ),
        "expected_rows": 68000,
        "expected_test": 13600,
        "expected_source_f1_min": 0.95,
    },
    "MLAAD": {
        "split": Path("splits/mlaad_ru_23class_split_v1.csv"),
        "features": Path("features/mlaad_23class_acoustic.parquet"),
        "meta": Path("features/mlaad_23class_acoustic.meta.json"),
        "key": "path",
        "model": Path(
            "results/raw/mlaad_23class_closedset_seed42/model.joblib"
        ),
        "expected_rows": 23000,
        "expected_test": 4600,
        "expected_source_f1_min": 0.85,
    },
}


def load_mapping():
    mapping = pd.read_csv(MAPPING_PATH)
    assert len(mapping) == 3
    assert set(mapping["family_label"]) == set(FAMILIES)
    assert mapping["ruasd_model"].is_unique
    assert mapping["mlaad_model"].is_unique
    assert set(mapping["allowed_attribution_level"]) == {"family"}

    return {
        "RuASD": dict(
            zip(mapping["ruasd_model"], mapping["family_label"])
        ),
        "MLAAD": dict(
            zip(mapping["mlaad_model"], mapping["family_label"])
        ),
    }


def check_feature_signature():
    signatures = {}
    for domain, cfg in DOMAINS.items():
        meta = json.loads(cfg["meta"].read_text(encoding="utf-8"))
        signatures[domain] = meta["feature_signature"]

    assert signatures["RuASD"] == signatures["MLAAD"], signatures
    print("FEATURE SIGNATURE VERIFIED:", signatures, flush=True)
    return signatures


def load_data(domain, feature_names, selected_splits=None):
    cfg = DOMAINS[domain]
    key = cfg["key"]

    sp = pd.read_csv(
        cfg["split"],
        usecols=[key, "generator", "split"],
        dtype={key: str},
        low_memory=False,
    )

    assert len(sp) == cfg["expected_rows"]
    assert sp[key].is_unique
    assert sp["split"].isin(["train", "validation", "test"]).all()
    assert int(sp["split"].eq("test").sum()) == cfg["expected_test"]

    if selected_splits is not None:
        sp = sp.loc[sp["split"].isin(selected_splits)].copy()

    sp = sp.reset_index(drop=True)

    feat = pd.read_parquet(
        cfg["features"],
        columns=[key] + feature_names,
    )
    feat[key] = feat[key].astype(str)

    assert feat[key].is_unique
    assert len(feat) == cfg["expected_rows"]

    missing = set(sp[key]) - set(feat[key])
    assert not missing, f"{domain}: missing feature IDs: {len(missing)}"

    X = feat.set_index(key).loc[sp[key], feature_names]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reset_index(drop=True)

    assert len(X) == len(sp)
    assert X.columns.tolist() == feature_names
    assert not X.isna().all(axis=1).any()

    print(
        f"{domain}: loaded {len(X)} records, "
        f"{len(feature_names)} features",
        flush=True,
    )
    return sp, X


def predict_batches(model, X, batch_size=2048):
    predictions = []
    scores = {method: [] for method in METHODS}

    n = len(X)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = X.iloc[start:stop]

        probabilities = np.asarray(model.predict_proba(batch))
        raw_scores = np.asarray(model.decision_scores(batch))

        assert probabilities.shape == raw_scores.shape
        assert probabilities.shape[1] == len(model.classes_)

        idx = np.argmax(probabilities, axis=1)
        predictions.append(model.classes_[idx].astype(str))

        scores["msp"].append(msp_id_score(probabilities))
        scores["energy"].append(
            energy_id_score(raw_scores, temperature=1.0)
        )
        scores["sme"].append(
            sme_id_score(raw_scores, temperature=4.0)
        )

        if stop == n or stop % (batch_size * 5) == 0:
            print(f"  Predictions: {stop}/{n}", flush=True)

    predictions = np.concatenate(predictions)
    scores = {
        method: np.concatenate(parts)
        for method, parts in scores.items()
    }

    for method, values in scores.items():
        assert len(values) == n
        assert np.isfinite(values).all(), method

    return predictions, scores


def evaluate_subset(
    truth,
    mapped_predictions,
    raw_predictions,
    id_scores,
    threshold,
    subset_mask,
):
    truth = truth[subset_mask]
    mapped = mapped_predictions[subset_mask]
    raw = raw_predictions[subset_mask]
    scores = id_scores[subset_mask]

    shared = truth != "unknown"
    not_shared = ~shared

    assert shared.any()
    assert not_shared.any()

    accepted = scores >= threshold
    final_predictions = np.where(
        accepted,
        mapped,
        "unknown",
    )

    result = open_set_metrics(
        truth[shared],
        mapped[shared],
        scores[shared],
        scores[not_shared],
    )

    result.update({
        "n_total": int(len(truth)),
        "n_shared": int(shared.sum()),
        "n_unmapped": int(not_shared.sum()),
        "macro_f1_four_labels": float(
            f1_score(
                truth,
                final_predictions,
                labels=list(FAMILIES) + ["unknown"],
                average="macro",
                zero_division=0,
            )
        ),
        "shared_accept_rate": float(np.mean(accepted[shared])),
        "shared_correct_attribution_rate": float(
            np.mean(final_predictions[shared] == truth[shared])
        ),
        "unmapped_rejection_rate": float(
            np.mean(~accepted[not_shared])
        ),
        "unmapped_false_accept_rate": float(
            np.mean(accepted[not_shared])
        ),
        "accepted_unmapped_source_label_count": int(
            np.sum(
                accepted &
                (mapped == PRIVATE_PREDICTION)
            )
        ),
        "per_shared_family": {
            family: {
                "n": int(np.sum(truth == family)),
                "correct_attribution_recall": float(
                    np.mean(
                        final_predictions[truth == family] == family
                    )
                ),
                "accept_rate": float(
                    np.mean(accepted[truth == family])
                ),
            }
            for family in FAMILIES
        },
    })

    return result


def run_direction(source, target, mappings):
    direction = f"{source}_to_{target}"
    print(f"\n========== {direction} ==========", flush=True)

    model = joblib.load(DOMAINS[source]["model"])
    feature_names = list(model.estimator.booster_.feature_name())

    assert len(feature_names) == 789
    assert "prosody_duration_sec" not in feature_names
    assert len(set(feature_names)) == 789

    source_class_names = set(map(str, model.classes_))
    assert set(mappings[source]).issubset(source_class_names)
    assert len(source_class_names) == (
        34 if source == "RuASD" else 23
    )

    source_sp, source_X = load_data(
        source,
        feature_names,
        selected_splits=["validation", "test"],
    )

    val_mask = source_sp["split"].eq("validation").to_numpy()
    test_mask = source_sp["split"].eq("test").to_numpy()

    print("SOURCE VALIDATION:", flush=True)
    _, validation_scores = predict_batches(
        model, source_X.loc[val_mask]
    )

    print("SOURCE TEST SANITY CHECK:", flush=True)
    source_test_pred, source_test_scores = predict_batches(
        model, source_X.loc[test_mask]
    )

    source_test_true = source_sp.loc[
        test_mask, "generator"
    ].astype(str).to_numpy()

    source_f1 = float(
        f1_score(
            source_test_true,
            source_test_pred,
            average="macro",
        )
    )

    print("SOURCE TEST MACRO-F1:", round(source_f1, 6), flush=True)
    assert source_f1 >= DOMAINS[source]["expected_source_f1_min"], (
        "Source-domain sanity check failed: "
        f"{source_f1:.6f}"
    )

    thresholds = {
        method: fit_known_quantile_threshold(
            validation_scores[method],
            reject_quantile=0.05,
        )
        for method in METHODS
    }

    del source_X

    target_sp, target_X = load_data(
        target,
        feature_names,
    )

    print("EXTERNAL TARGET PREDICTIONS:", flush=True)
    target_pred, target_scores = predict_batches(
        model,
        target_X,
    )

    del target_X

    target_generator = target_sp["generator"].astype(str).to_numpy()

    truth = np.array(
        [
            mappings[target].get(label, "unknown")
            for label in target_generator
        ],
        dtype=str,
    )

    mapped_pred = np.array(
        [
            mappings[source].get(label, PRIVATE_PREDICTION)
            for label in target_pred
        ],
        dtype=str,
    )

    test_only = target_sp["split"].eq("test").to_numpy()
    full = np.ones(len(target_sp), dtype=bool)

    expected_known = (
        3000 if target == "MLAAD" else 6000
    )

    assert len(target_sp) == DOMAINS[target]["expected_rows"]
    assert int(np.sum(truth != "unknown")) == expected_known

    report = {
        "source": source,
        "target": target,
        "source_model": str(DOMAINS[source]["model"]),
        "source_in_domain_test_macro_f1": source_f1,
        "source_validation_n": int(val_mask.sum()),
        "source_test_n": int(test_mask.sum()),
        "target_total_n": int(len(target_sp)),
        "target_frozen_test_n": int(test_only.sum()),
        "methods": {},
    }

    prediction_frames = []

    for method in METHODS:
        values = target_scores[method]
        threshold = thresholds[method]
        accepted = values >= threshold

        final_pred = np.where(
            accepted,
            mapped_pred,
            "unknown",
        )

        metrics_full = evaluate_subset(
            truth, mapped_pred, target_pred,
            values, threshold, full,
        )
        metrics_test = evaluate_subset(
            truth, mapped_pred, target_pred,
            values, threshold, test_only,
        )

        report["methods"][method] = {
            "threshold": float(threshold),
            "source_validation_accept_rate": float(
                np.mean(validation_scores[method] >= threshold)
            ),
            "source_test_accept_rate": float(
                np.mean(source_test_scores[method] >= threshold)
            ),
            "full_external": metrics_full,
            "frozen_test_external": metrics_test,
        }

        prediction_frames.append(
            pd.DataFrame({
                "direction": direction,
                "record_id": target_sp[
                    DOMAINS[target]["key"]
                ].astype(str).to_numpy(),
                "target_split": target_sp["split"].to_numpy(),
                "target_generator": target_generator,
                "target_family_or_unknown": truth,
                "is_shared_family": truth != "unknown",
                "source_model_prediction": target_pred,
                "mapped_source_prediction": mapped_pred,
                "method": method,
                "id_score": values,
                "threshold": threshold,
                "accepted": accepted,
                "final_prediction": final_pred,
            })
        )

        print(
            f"{method.upper():7s}"
            f" AUROC={metrics_full['auroc_id_vs_ood']:.4f}"
            f" FPR95={metrics_full['fpr95']:.4f}"
            f" OSCR={metrics_full['oscr']:.4f}"
            f" F1={metrics_full['macro_f1_four_labels']:.4f}"
            f" shared_correct="
            f"{metrics_full['shared_correct_attribution_rate']:.4f}"
            f" unmapped_reject="
            f"{metrics_full['unmapped_rejection_rate']:.4f}",
            flush=True,
        )

    return report, pd.concat(prediction_frames, ignore_index=True)


def main():
    if OUT.exists():
        raise RuntimeError(
            f"Output already exists: {OUT}. "
            "Existing results will not be overwritten."
        )

    signatures = check_feature_signature()
    mappings = load_mapping()

    reports = {}
    predictions = []

    for source, target in [
        ("RuASD", "MLAAD"),
        ("MLAAD", "RuASD"),
    ]:
        report, pred = run_direction(source, target, mappings)
        reports[f"{source}_to_{target}"] = report
        predictions.append(pred)

    all_predictions = pd.concat(predictions, ignore_index=True)

    assert len(all_predictions) == (
        (23000 + 68000) * len(METHODS)
    )

    OUT.mkdir(parents=True, exist_ok=False)

    context = {
        "experiment": "cross_dataset_open_set_full_source_classes",
        "seed": 42,
        "feature_count": 789,
        "feature_signatures": signatures,
        "mapping_file": str(MAPPING_PATH),
        "mapping_level": "family_only",
        "shared_families": list(FAMILIES),
        "source_threshold_quantile": 0.05,
        "threshold_uses_target_data": False,
        "target_all_splits_are_evaluation_only": True,
        "methods": list(METHODS),
        "unmapped_source_prediction_is_not_unknown": True,
        "notes": (
            "Unknown ground truth means no confirmed pair among "
            "the three shared families, not necessarily an unseen "
            "technology. Four-label Macro-F1 excludes the auxiliary "
            "unmapped-source prediction label from averaging, "
            "but its predictions count as classification errors."
        ),
    }

    (OUT / "metrics.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_predictions.to_parquet(
        OUT / "predictions.parquet",
        index=False,
    )

    print("\nEXPERIMENT COMPLETED", flush=True)
    print("SAVED:", OUT, flush=True)
    print("PREDICTION ROWS:", len(all_predictions), flush=True)


if __name__ == "__main__":
    main()
