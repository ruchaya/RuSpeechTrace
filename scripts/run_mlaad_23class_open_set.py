
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from source_attribution.config import load_config
from source_attribution.experiment import feature_columns
from source_attribution.models import build_and_train
from source_attribution.ood import (
    msp_id_score,
    energy_id_score,
    sme_id_score,
    fit_known_quantile_threshold,
    class_prototypes,
    prototype_scores,
)
from source_attribution.metrics import open_set_metrics
from source_attribution.utils import set_seed


FEATURES = Path("features/mlaad_23class_acoustic.parquet")
FEATURE_META = Path("features/mlaad_23class_acoustic.meta.json")
SPLIT = Path("splits/mlaad_ru_23class_split_v1.csv")
FOLDS = Path("splits/mlaad_23class_open_set_folds_v1.json")
CONFIG = Path("configs/open_set.yaml")

SEEDS = [42, 123, 456, 789, 2026]
METHODS = ["msp", "energy", "sme", "prototype"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def classifier_score(method, model, X, cfg):
    if method == "msp":
        return msp_id_score(model.predict_proba(X))

    logits = model.decision_scores(X)

    if method == "energy":
        return energy_id_score(
            logits,
            float(cfg["ood"]["temperature_energy"]),
        )

    if method == "sme":
        return sme_id_score(
            logits,
            float(cfg["ood"]["temperature_sme"]),
        )

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.fold not in range(1, 6):
        raise ValueError("Fold must be 1..5")

    if args.seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")

    set_seed(args.seed)

    cfg = load_config(CONFIG)

    if cfg["model"]["type"] != "lightgbm":
        raise RuntimeError("Expected LightGBM config")

    if list(cfg["ood"]["methods"]) != METHODS:
        raise RuntimeError("OOD methods do not match baseline")

    fm = json.loads(FEATURE_META.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))

    split_hash = sha256(SPLIT)
    feature_hash = sha256(FEATURES)
    folds_hash = sha256(FOLDS)

    assert fm["split_sha256"] == split_hash
    assert fm["output_sha256"] == feature_hash
    assert folds["source_split_sha256"] == split_hash
    assert folds["n_folds"] == 5

    d = pd.read_parquet(FEATURES)

    assert len(d) == 23000
    assert d["path"].is_unique
    assert d["generator"].nunique() == 23

    original = pd.read_csv(
        SPLIT,
        usecols=["path", "generator", "split"],
        encoding="utf-8-sig",
    )

    a = d[["path", "generator", "split"]].sort_values(
        "path"
    ).reset_index(drop=True)

    b = original.sort_values("path").reset_index(drop=True)

    assert a.equals(b)

    for key in ["component", "text_key", "original_file"]:
        assert d.groupby(key)["split"].nunique().max() == 1

    all_classes = set(d["generator"].astype(str))

    unknown_occurrences = []

    for fd in folds["folds"].values():
        known = set(fd["known_generators"])
        unknown = set(fd["unknown_generators"])

        assert not (known & unknown)
        assert known | unknown == all_classes

        unknown_occurrences.extend(unknown)

    assert len(unknown_occurrences) == 23
    assert set(unknown_occurrences) == all_classes

    fd = folds["folds"][str(args.fold)]

    known = set(fd["known_generators"])
    unknown = set(fd["unknown_generators"])

    assert len(known) in (18, 19)
    assert len(unknown) == 23 - len(known)

    fs = cfg.get("feature_selection", {})

    cols = feature_columns(
        d,
        fs.get("include_prefixes"),
        fs.get("exclude_columns"),
        bool(fs.get("exclude_duration", True)),
    )

    assert len(cols) == 789
    assert "prosody_duration_sec" not in cols

    X = d[cols].replace([np.inf, -np.inf], np.nan)

    assert int(X.isna().sum().sum()) == 156
    assert not X.isna().all(axis=1).any()

    g = d["generator"].astype(str)

    tr = d["split"].eq("train") & g.isin(known)
    va = d["split"].eq("validation") & g.isin(known)
    kt = d["split"].eq("test") & g.isin(known)
    ut = d["split"].eq("test") & g.isin(unknown)

    nk = len(known)
    nu = len(unknown)

    assert int(tr.sum()) == nk * 650
    assert int(va.sum()) == nk * 150
    assert int(kt.sum()) == nk * 200
    assert int(ut.sum()) == nu * 200

    assert not g.loc[tr | va].isin(unknown).any()
    assert int((kt | ut).sum()) == 4600

    print("MLAAD OPEN-SET AUDIT PASSED")
    print("FOLD:", args.fold)
    print("SEED:", args.seed)
    print("KNOWN GENERATORS:", nk)
    print("UNKNOWN GENERATORS:", nu)
    print("UNKNOWN NAMES:", sorted(unknown))
    print("FEATURES:", len(cols))
    print("TRAIN KNOWN:", int(tr.sum()))
    print("VALIDATION KNOWN:", int(va.sum()))
    print("TEST KNOWN:", int(kt.sum()))
    print("TEST UNKNOWN:", int(ut.sum()))
    print("SPLIT SHA256:", split_hash)
    print("FOLDS SHA256:", folds_hash)
    print("UNKNOWN TRAIN/VALIDATION LEAKAGE: NONE")

    if args.dry_run:
        print("DRY RUN PASSED - NO TRAINING")
        return

    output = Path(
        f"results/raw/mlaad_23class_openset_"
        f"fold{args.fold}_seed{args.seed}"
    )

    if output.exists():
        raise RuntimeError(
            f"Output already exists: {output}"
        )

    print("TRAINING LIGHTGBM...")

    mc = cfg["model"]

    model = build_and_train(
        X.loc[tr],
        g.loc[tr],
        X.loc[va],
        g.loc[va],
        mc["type"],
        mc.get("params", {}),
        80,
        args.seed,
    )

    assert set(model.classes_) == known

    known_true = g.loc[kt].to_numpy()
    known_pred_model = model.predict(X.loc[kt])
    unknown_pred_model = model.predict(X.loc[ut])

    # Prototype preprocessing is fitted on known train only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Xt = scaler.fit_transform(
        imputer.fit_transform(X.loc[tr])
    )

    Xv = scaler.transform(
        imputer.transform(X.loc[va])
    )

    Xk = scaler.transform(
        imputer.transform(X.loc[kt])
    )

    Xu = scaler.transform(
        imputer.transform(X.loc[ut])
    )

    prototype_labels, centers = class_prototypes(
        Xt,
        g.loc[tr].to_numpy(),
    )

    results = {}
    frames = []

    for method in METHODS:
        print("EVALUATING:", method)

        if method == "prototype":
            _, val_scores, _ = prototype_scores(Xv, centers)

            ki, known_scores, _ = prototype_scores(
                Xk, centers
            )

            ui, unknown_scores, _ = prototype_scores(
                Xu, centers
            )

            known_pred = prototype_labels[ki]
            unknown_pred = prototype_labels[ui]

        else:
            val_scores = classifier_score(
                method, model, X.loc[va], cfg
            )

            known_scores = classifier_score(
                method, model, X.loc[kt], cfg
            )

            unknown_scores = classifier_score(
                method, model, X.loc[ut], cfg
            )

            known_pred = known_pred_model
            unknown_pred = unknown_pred_model

        for scores in [
            val_scores, known_scores, unknown_scores
        ]:
            assert np.isfinite(scores).all()

        threshold = fit_known_quantile_threshold(
            val_scores,
            float(cfg["ood"]["known_reject_quantile"]),
        )

        known_accepted = known_scores >= threshold
        unknown_accepted = unknown_scores >= threshold

        metrics = open_set_metrics(
            known_true,
            known_pred,
            known_scores,
            unknown_scores,
        )

        open_true = np.concatenate([
            known_true,
            np.repeat("unknown", int(ut.sum())),
        ])

        open_pred = np.concatenate([
            np.where(
                known_accepted, known_pred, "unknown"
            ),
            np.where(
                unknown_accepted, unknown_pred, "unknown"
            ),
        ])

        metrics.update({
            "threshold": float(threshold),
            "validation_known_accept_rate": float(
                np.mean(val_scores >= threshold)
            ),
            "known_accept_rate": float(
                np.mean(known_accepted)
            ),
            "unknown_reject_rate": float(
                np.mean(~unknown_accepted)
            ),
            "known_macro_f1": float(
                f1_score(
                    known_true,
                    known_pred,
                    average="macro",
                )
            ),
            "macro_f1_with_unknown": float(
                f1_score(
                    open_true,
                    open_pred,
                    average="macro",
                )
            ),
            "test_known_count": int(kt.sum()),
            "test_unknown_count": int(ut.sum()),
        })

        results[method] = metrics

        for mask, scores, predictions, accepted, is_known in [
            (
                kt, known_scores, known_pred,
                known_accepted, True
            ),
            (
                ut, unknown_scores, unknown_pred,
                unknown_accepted, False
            ),
        ]:
            part = d.loc[
                mask,
                ["path", "generator", "split"],
            ].copy()

            part["fold"] = args.fold
            part["seed"] = args.seed
            part["method"] = method
            part["is_known_true"] = is_known
            part["generator_pred"] = predictions
            part["id_score"] = scores
            part["accepted_known"] = accepted

            frames.append(part)

        print(
            method,
            "AUROC:", round(metrics["auroc_id_vs_ood"], 5),
            "FPR95:", round(metrics["fpr95"], 5),
            "OSCR:", round(metrics["oscr"], 5),
        )

    # Save only after training and metric calculations succeed.
    output.mkdir(parents=True, exist_ok=False)

    context = {
        "dataset": "MLAAD",
        "experiment": "23class_family_grouped_open_set_v1",
        "fold": args.fold,
        "seed": args.seed,
        "known_generators": sorted(known),
        "unknown_generators": sorted(unknown),
        "features": len(cols),
        "split_sha256": split_hash,
        "features_sha256": feature_hash,
        "folds_sha256": folds_hash,
        "config_sha256": sha256(CONFIG),
        "speaker_disjoint": False,
        "threshold_source": "known_validation_only",
        "early_stopping_rounds": 80,
    }

    save_json(context, output / "context.json")
    save_json(results, output / "metrics.json")

    pd.concat(frames, ignore_index=True).to_parquet(
        output / "predictions.parquet",
        index=False,
    )

    (output / "feature_columns.txt").write_text(
        "\n".join(cols),
        encoding="utf-8",
    )

    joblib.dump(model, output / "model.joblib")

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "prototype_labels": prototype_labels,
            "prototype_centers": centers,
        },
        output / "prototype.joblib",
    )

    print("SAVED:", output)
    print("OPEN-SET EXPERIMENT PASSED")


if __name__ == "__main__":
    main()
