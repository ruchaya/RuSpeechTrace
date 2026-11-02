#!/usr/bin/env python
"""Paired five-seed *training-composition* sensitivity study (not LR initialization SD).

All folds, validation and test objects remain frozen. In each seed the same
stratified 80% generator-wise training subset is used for Acoustic and WavLM.
No unknown generator is used for training, preprocessing or threshold selection.
This is a supplementary protocol separate from the original full-training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from fast_open_set_metrics import open_set_metrics
from source_attribution.config import load_config
from source_attribution.experiment import feature_columns
from source_attribution.leakage import assert_no_leakage, assert_unknown_absent
from source_attribution.models import build_and_train
from source_attribution.ood import (
    class_prototypes, energy_id_score, fit_known_quantile_threshold,
    msp_id_score, prototype_scores, sme_id_score,
)
from source_attribution.utils import json_load, set_seed

DEFAULT_SEEDS = (42, 123, 456, 789, 2026)


def stratum_sample(train: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    """Select the same train IDs for every representation (never sample val/test)."""
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for generator in sorted(train["generator"].astype(str).unique()):
        ids = np.sort(train.loc[train["generator"].astype(str) == generator, "file_id"].astype(str).to_numpy())
        if not len(ids):
            raise ValueError(f"Empty generator stratum: {generator}")
        count = max(1, int(np.floor(len(ids) * fraction)))
        selected.update(map(str, rng.choice(ids, count, replace=False)))
    return selected


def short_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def load_representation(path: str, split: pd.DataFrame, cfg_path: str) -> tuple[pd.DataFrame, list[str], dict]:
    cfg = load_config(cfg_path)
    feat = pd.read_parquet(path)
    if "file_id" not in feat:
        raise RuntimeError(f"Feature store lacks file_id: {path}")
    feat["file_id"] = feat["file_id"].astype(str)
    if feat.file_id.duplicated().any():
        raise RuntimeError(f"Duplicate file IDs in {path}")
    if len(feat) != len(split) or set(feat.file_id) != set(split.file_id):
        raise RuntimeError(f"Feature store and frozen split do not match: {path}")
    for field in ("generator", "split"):
        if field in feat:
            expected = split.set_index("file_id")[field].astype(str)
            actual = feat.set_index("file_id")[field].astype(str)
            if not actual.sort_index().equals(expected.sort_index()):
                raise RuntimeError(f"Feature {field} mismatch: {path}")
            feat = feat.drop(columns=[field])
    df = split.merge(feat, on="file_id", how="inner", validate="one_to_one")
    if len(df) != len(split):
        raise RuntimeError("Feature merge lost rows")
    fs = cfg.get("feature_selection", {})
    cols = feature_columns(df, fs.get("include_prefixes"), fs.get("exclude_columns"), bool(fs.get("exclude_duration", False)))
    if df[cols].select_dtypes(include=["number"]).shape[1] != len(cols):
        raise RuntimeError("Unexpected nonnumeric feature")
    return df, cols, cfg


def evaluate_method(method, cfg, model, tr_features, tr_generators,
                    va_features, va_generators,
                    kt_features, kt_generators,
                    ut_features, known_frame, unknown_frame, fold, seed, sample_signature):
    """Shared logistic backbone and frozen samples; prototype uses train-only scaling."""
    if method == "prototype":
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        train_p = sc.fit_transform(imp.fit_transform(tr_features))
        valid_p = sc.transform(imp.transform(va_features))
        known_p = sc.transform(imp.transform(kt_features))
        unknown_p = sc.transform(imp.transform(ut_features))
        labels, centers = class_prototypes(train_p, np.asarray(tr_generators).astype(str))
        v_idx, vs, _ = prototype_scores(valid_p, centers)
        k_idx, ks, _ = prototype_scores(known_p, centers)
        u_idx, us, _ = prototype_scores(unknown_p, centers)
        kp = labels[k_idx]
        up = labels[u_idx]
    else:
        def scores(X):
            if method == "msp":
                return msp_id_score(model.predict_proba(X))
            if method == "energy":
                return energy_id_score(model.decision_scores(X), float(cfg["ood"].get("temperature_energy", 1)))
            if method == "sme":
                return sme_id_score(model.decision_scores(X), float(cfg["ood"].get("temperature_sme", 4)))
            raise ValueError(f"Unrecognized method {method}")
        vs, ks, us = scores(va_features), scores(kt_features), scores(ut_features)
        kp, up = model.predict(kt_features), model.predict(ut_features)
    q = float(cfg["ood"].get("known_reject_quantile", .05))
    threshold = fit_known_quantile_threshold(vs, q)
    known_true = np.asarray(kt_generators).astype(str)
    metrics = open_set_metrics(known_true, kp, ks, us)
    ytrue = np.r_[known_true, ["unknown"] * len(us)].astype(str)
    ypred = np.r_[np.where(ks >= threshold, kp, "unknown"), np.where(us >= threshold, up, "unknown")].astype(str)
    metrics.update({
        "threshold": float(threshold),
        "known_accept_rate": float(np.mean(ks >= threshold)),
        "unknown_reject_rate": float(np.mean(us < threshold)),
        "macro_f1_with_unknown": float(f1_score(ytrue, ypred, average="macro")),
        "sample_signature": sample_signature,
    })
    known = known_frame[["file_id", "generator"]].copy()
    known["is_known_true"] = True
    known["generator_pred"] = np.asarray(kp).astype(str)
    known["id_score"] = ks
    known["accepted_known"] = ks >= threshold
    unknown = unknown_frame[["file_id", "generator"]].copy()
    unknown["is_known_true"] = False
    unknown["generator_pred"] = np.asarray(up).astype(str)
    unknown["id_score"] = us
    unknown["accepted_known"] = us >= threshold
    pred = pd.concat([known, unknown], ignore_index=True)
    pred["fold"] = str(fold)
    pred["seed"] = int(seed)
    pred["method"] = method
    pred["threshold"] = float(threshold)
    return metrics, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", default="features/ruasd_closed_set_acoustic_fixed4.parquet")
    parser.add_argument("--wavlm", default="embeddings/wavlm_base_ruasd_4s/all_68000_for_open_set.parquet")
    parser.add_argument("--acoustic-config", default="configs/open_set_acoustic_fixed4_lr_C1.yaml")
    parser.add_argument("--wavlm-config", default="configs/open_set_wavlm_fixed4_lr_C1.yaml")
    parser.add_argument("--splits", default="splits/closed_set.csv")
    parser.add_argument("--folds", default="splits/open_set_folds.json")
    parser.add_argument("--output", default="results/raw/fixed4_lr_C1_paired_80pct_train_5seeds")
    parser.add_argument("--fraction", type=float, default=.80)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--only-fold", default=None, help="Run only one fold, useful for preflight")
    args = parser.parse_args()
    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("fraction must be within (0, 1]")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    split = pd.read_csv(args.splits)
    split["file_id"] = split["file_id"].astype(str)
    split["generator"] = split["generator"].astype(str)
    assert_no_leakage(split)
    if split.file_id.duplicated().any():
        raise RuntimeError("Frozen split has duplicate file IDs")
    train_full = split[split.split == "train"]
    folds = json_load(args.folds)["folds"]
    representations = {
        "acoustic_lr_C1": (args.acoustic, args.acoustic_config),
        "wavlm_lr_C1": (args.wavlm, args.wavlm_config),
    }
    datasets = {}
    print("Checking both feature stores and configs...", flush=True)
    for name, (fp, config) in representations.items():
        data, columns, cfg = load_representation(fp, split, config)
        if cfg["model"]["type"] != "logistic":
            raise RuntimeError(f"Expected logistic classifier for {name}")
        if float(cfg["model"].get("params", {}).get("C", 0)) != 1.0:
            raise RuntimeError(f"Expected pre-fixed C=1.0 for {name}")
        datasets[name] = (data, columns, cfg)
        print(f"  {name}: {len(data)} records, {len(columns)} features", flush=True)
    outroot = Path(args.output)
    outroot.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        set_seed(seed)
        print(f"Seed {seed}", flush=True)
        for fold_id, fold in folds.items():
            if args.only_fold is not None and str(fold_id) != str(args.only_fold):
                continue
            known = set(map(str, fold["known_generators"]))
            unknown = set(map(str, fold["unknown_generators"]))
            if known & unknown:
                raise RuntimeError(f"Fold {fold_id} has intersecting known and unknown classes")
            # Sample only permitted known-generator training records in this fold.
            train_ids = stratum_sample(train_full[train_full.generator.isin(known)], args.fraction, seed)
            train_sig = short_hash(train_ids)
            print(f"  Fold {fold_id}: {len(train_ids)} permitted training IDs; sha256={train_sig[:16]}", flush=True)
            for rep, (df, cols, cfg) in datasets.items():
                folder = outroot / rep / f"seed{seed}" / f"fold{fold_id}"
                path = folder / "metrics.json"
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if (old.get("train_sha256") != train_sig
                            or old.get("train_fraction") != args.fraction
                            or old.get("model") != cfg["model"]
                            or old.get("feature_path") != representations[rep][0]
                            or old.get("feature_count") != len(cols)):
                        raise RuntimeError(f"Existing run config or subset differs: {path}")
                    if not (folder/"predictions.parquet").exists():
                        raise RuntimeError(f"Missing predictions for completed run: {path}")
                    print(f"  SKIP completed {rep} fold {fold_id} seed {seed}", flush=True)
                    continue
                tr = (df.split == "train") & df.file_id.isin(train_ids) & df.generator.isin(known)
                va = (df.split == "validation") & df.generator.isin(known)
                kt = (df.split == "test") & df.generator.isin(known)
                ut = (df.split == "test") & df.generator.isin(unknown)
                if min(tr.sum(), va.sum(), kt.sum(), ut.sum()) < 1:
                    raise RuntimeError(f"Empty fold partition at {fold_id}")
                assert_unknown_absent(df.loc[tr], df.loc[va], unknown)
                for mask_name, mask in (("train", tr), ("validation", va), ("known_test", kt), ("unknown_test", ut)):
                    if df.loc[mask, "file_id"].duplicated().any():
                        raise RuntimeError(f"Duplicate IDs in {mask_name}")
                print(f"  Training {rep}, seed {seed}, fold {fold_id} ({tr.sum()} train)...", flush=True)
                X = df[cols].replace([np.inf, -np.inf], np.nan)
                model = build_and_train(X.loc[tr], df.loc[tr,"generator"], X.loc[va], df.loc[va,"generator"],
                                        cfg["model"]["type"], cfg["model"].get("params", {}), 80, seed)
                methods = cfg["ood"]["methods"]
                allpred, scores = [], {}
                tr_sha = short_hash(df.loc[tr, "file_id"])
                for method in methods:
                    result, pred = evaluate_method(
                        method, cfg, model,
                        X.loc[tr], df.loc[tr,"generator"],
                        X.loc[va], df.loc[va,"generator"],
                        X.loc[kt], df.loc[kt,"generator"],
                        X.loc[ut], df.loc[kt], df.loc[ut], fold_id, seed, tr_sha,
                    )
                    scores[method] = result
                    allpred.append(pred)
                folder.mkdir(parents=True, exist_ok=True)
                pd.concat(allpred, ignore_index=True).to_parquet(folder/"predictions.parquet", index=False)
                payload = {
                    "representation": rep, "seed": seed, "fold": str(fold_id),
                    "train_fraction": args.fraction, "n_train": int(tr.sum()),
                    "n_known_test": int(kt.sum()), "n_unknown_test": int(ut.sum()),
                    "train_sha256": tr_sha,
                    "model": cfg["model"], "feature_count": len(cols),
                    "feature_path": representations[rep][0],
                    "methods": scores,
                }
                temp = path.with_name("metrics.tmp.json")
                temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                temp.replace(path)
                print(f"    Saved {path}", flush=True)
    print("Done. Run scripts/summarize_fixed4_openset_resampled.py", flush=True)


if __name__ == "__main__":
    main()
