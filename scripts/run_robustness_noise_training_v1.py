#!/usr/bin/env python
"""Leakage-safe RuASD noise-augmentation training, matched to robustness_main_v1.

Run from the project root. Does not modify the legacy robustness test or its model.
EXTRACT consumes TRAIN TAR members only. EVALUATE reads previously extracted
680-test-record condition FEATURES (never test WAVs for training).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from source_attribution.ruasd_shards import resolve_ruasd_shard

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from scripts.extract_acoustic_features import build_feature_config
from scripts.run_robustness_main_v1 import add_noise, CONDITIONS
from source_attribution.acoustic_features import extract_acoustic_features
from source_attribution.audio import _read_audio, _resample, trim_or_pad_audio, normalize_audio
from source_attribution.models import TrainedClassifier

SPLIT_HASH = "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0"
CONDITIONS_TRAIN = ("noise_30db", "noise_20db")
EXPECTED_TRAIN = 44200
EXPECTED_VAL = 10200
EXPECTED_TEST = 13600
REF = Path("results/raw/robustness_main_v1")
FEATS = Path("features/ruasd_closed_set_acoustic.parquet")
SPLITS = Path("splits/closed_set.csv")
CONFIG = Path("configs/closed_set_acoustic.yaml")


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def columns(df):
    all_cols = [c for c in df.columns if c.startswith(("spectral_", "prosody_"))]
    fit_cols = [c for c in all_cols if c != "prosody_duration_sec"]
    if len(all_cols) != 790 or len(fit_cols) != 789:
        raise ValueError(f"Expected 790/789 features, found {len(all_cols)}/{len(fit_cols)}")
    return all_cols, fit_cols


def check_splits(sp):
    if sha256_file(SPLITS) != SPLIT_HASH:
        raise RuntimeError("Frozen split SHA256 mismatch; refusing to proceed")
    if sp.file_id.isna().any() or not sp.file_id.is_unique:
        raise RuntimeError("Missing/duplicate file_id in split")
    counts = sp.split.value_counts().to_dict()
    if counts != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VAL, "test": EXPECTED_TEST}:
        raise RuntimeError(f"Unexpected split totals: {counts}")
    if sp.generator.nunique() != 34:
        raise RuntimeError("Expected 34 generators")
    for k, size in [("train",1300),("validation",300),("test",400)]:
        z = sp.loc[sp.split.eq(k)].groupby("generator").size()
        if not (len(z) == 34 and (z == size).all()):
            raise RuntimeError(f"Invalid generator quota: {k}")


def select_train(sp, seed, per_class):
    if not 1 <= per_class <= 1300:
        raise ValueError("per-class must be 1..1300")
    train = sp.loc[sp.split.eq("train")]
    result = []
    for i, (g, grp) in enumerate(sorted(train.groupby("generator"), key=lambda x: x[0])):
        rng = np.random.default_rng(seed + 7919 * (i + 1))
        grp = grp.sort_values("file_id").reset_index(drop=True)
        ix = np.sort(rng.choice(len(grp), size=per_class, replace=False))
        result.append(grp.iloc[ix])
    selected = pd.concat(result, ignore_index=True).sort_values("file_id").reset_index(drop=True)
    assert selected.file_id.is_unique and selected.generator.nunique() == 34
    return selected.loc[:, ["file_id", "generator", "shard_path", "audio_member"]]


def root(args):
    return Path(f"results/raw/robustness_training_noise30_noise20_n{args.per_class}_seed{args.seed}")


def save_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, dtype={"file_id": str})
        if not old.equals(rows):
            raise RuntimeError("Frozen training selection differs from on-disk manifest")
    else:
        rows.to_csv(path, index=False)


def check_test_reference(test_df, sp, feat_cols, clean_model=None):
    if not set(CONDITIONS).issubset(set(test_df.condition.unique())):
        raise RuntimeError("Missing existing robustness test conditions")
    selected = pd.read_csv(REF / "selected_test_records.csv", dtype={"file_id": str})
    selected_ids = set(selected.file_id)
    if len(selected_ids) != 680 or not selected_ids.issubset(set(sp.loc[sp.split.eq("test"), "file_id"])):
        raise RuntimeError("Test selection is not the 680 original frozen TEST IDs")
    if test_df.duplicated(["file_id", "condition"]).any():
        raise RuntimeError("Duplicated record/condition in existing test features")
    for c in CONDITIONS:
        subset = test_df.loc[test_df.condition.eq(c)]
        if len(subset) != 680 or set(subset.file_id) != selected_ids:
            raise RuntimeError(f"Incomplete/mismatched existing test condition {c}")
    label_map = sp.set_index("file_id").generator.to_dict()
    if (test_df.generator.astype(str) != test_df.file_id.map(label_map).astype(str)).any():
        raise RuntimeError("Test class labels mismatch frozen split")
    _, xcols = columns(test_df)
    if any(c not in test_df.columns for c in feat_cols):
        raise RuntimeError("Missing acoustic feature columns in test data")
    if clean_model is not None:
        clean = test_df.loc[test_df.condition.eq("clean")].sort_values("file_id")
        f1 = f1_score(clean.generator, clean_model.predict(clean[xcols].replace([np.inf,-np.inf],np.nan)), average="macro", labels=clean_model.classes_)
        if abs(f1 - 0.9765) > 0.001:
            raise RuntimeError(f"Clean reference F1={f1:.6f} disagrees with published 0.9765")
        print(f"REFERENCE VERIFIED: clean-only Macro-F1={f1:.6f}", flush=True)


def preflight():
    for p in [SPLITS, FEATS, CONFIG, REF / "all_features.parquet", REF / "selected_test_records.csv", REF / "clean_789_model.joblib"]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required prior artifact: {p}")
    sp = pd.read_csv(SPLITS, dtype={"file_id": str})
    check_splits(sp)
    feats = pd.read_parquet(FEATS)
    all_cols, model_cols = columns(feats)
    if len(feats) != 68000 or not feats.file_id.is_unique or set(feats.file_id) != set(sp.file_id):
        raise RuntimeError("Canonical 68k feature store has incorrect IDs")
    test = pd.read_parquet(REF / "all_features.parquet")
    clean_model = joblib.load(REF / "clean_789_model.joblib")
    if list(clean_model.classes_) != sorted(sp.generator.astype(str).unique()):
        raise RuntimeError("Saved clean baseline classes mismatch")
    if hasattr(clean_model, "estimator") and getattr(clean_model.estimator,"feature_name_", None) is not None:
        if list(clean_model.estimator.feature_name_) != model_cols:
            raise RuntimeError("Saved clean baseline feature order mismatch")
    check_test_reference(test, sp, all_cols, clean_model)
    print("PREFLIGHT OK: frozen SHA256, 68k features, saved clean model, 680 paired TEST recordings x 6 conditions",flush=True)
    return sp, feats, test, clean_model, all_cols, model_cols


def part_path(parts, file_id):
    return parts / (hashlib.sha256(file_id.encode("utf8")).hexdigest()[:24]+".parquet")


def extract_one(row, archive, all_cols, cfg, settings, ref_features, verify_clean):
    # Every source is a frozen TRAIN record, never from a derived RuASD augmentation.
    h = archive.extractfile(row.audio_member)
    if h is None:
        raise RuntimeError(f"Cannot read {row.audio_member}")
    audio_bytes = h.read()
    x,sr = _read_audio(io.BytesIO(audio_bytes), mono=True)
    x = _resample(x, sr, cfg.sr)
    original_duration = len(x)/cfg.sr
    x = trim_or_pad_audio(x, cfg.sr,
        max_duration_sec=settings["max_duration_sec"],
        min_audio_sec=settings["min_audio_sec"])
    if verify_clean:
        normal = normalize_audio(x, peak_normalize=settings["peak_normalize"])
        clean = extract_acoustic_features(normal,cfg.sr,original_duration=original_duration,cfg=cfg)
        expected = ref_features.loc[row.file_id,all_cols].to_numpy(dtype=float)
        observed = np.asarray([clean[c] for c in all_cols], dtype=float)
        if not np.isclose(expected, observed, rtol=1e-5, atol=1e-5, equal_nan=True).all():
            raise RuntimeError(f"Clean feature mismatch for TRAIN {row.file_id}; protocol mismatch")
    out=[]
    for condition in CONDITIONS_TRAIN:
        changed = add_noise(x, int(condition.split('_')[1].replace('db','')), row.file_id, condition)
        normal=normalize_audio(changed,peak_normalize=settings["peak_normalize"])
        acoustic=extract_acoustic_features(normal,cfg.sr,original_duration=original_duration,cfg=cfg)
        if set(acoustic) != set(all_cols):
            raise RuntimeError("Unexpected feature definition")
        entry={"file_id":row.file_id,"generator":row.generator,"condition":condition}
        entry.update(acoustic)
        out.append(entry)
    return pd.DataFrame(out)


def extract(args):
    sp, feats, test, model, all_cols, model_cols = preflight()
    selected = select_train(sp, args.seed, args.per_class)
    dst = root(args); dst.mkdir(parents=True, exist_ok=True)
    save_manifest(dst / "selected_train_records.csv", selected)
    if not set(selected.file_id).issubset(set(sp.loc[sp.split.eq("train")].file_id)):
        raise RuntimeError("Training selection leaked outside frozen train")
    settings=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["feature_extraction"]
    cfg=build_feature_config(settings)
    parts=dst / "parts";parts.mkdir(exist_ok=True)
    features_by_id=feats.set_index("file_id")
    ordered=selected.sort_values(["shard_path","audio_member"]).reset_index(drop=True)
    pending=[]
    for row in ordered.itertuples(index=False):
        f = part_path(parts, str(row.file_id))
        if f.exists():
            try:
                z = pd.read_parquet(f,columns=["file_id","condition"])
                if len(z)==2 and z.file_id.eq(row.file_id).all() and set(z.condition)==set(CONDITIONS_TRAIN):
                    continue
            except Exception:
                pass
        pending.append(row)
    print(f"TRAIN ONLY: {len(selected)} parent audio files, {len(selected)*2} new feature rows; remaining audio files={len(pending)}",flush=True)
    count=0
    # Open each TAR only once, rather than once per member.
    pending_df=pd.DataFrame(pending,columns=selected.columns)
    for shard_path, grp in pending_df.groupby("shard_path",sort=False):
        if args.limit is not None and count>=args.limit: break
        with tarfile.open(resolve_ruasd_shard(shard_path), "r:*") as archive:
            for row in grp.itertuples(index=False):
                if args.limit is not None and count>=args.limit: break
                try:
                    out=extract_one(row,archive,all_cols,cfg,settings,features_by_id,verify_clean=count<6)
                except Exception as ex:
                    raise RuntimeError(f"TRAIN extraction failed at file_id={row.file_id}") from ex
                for col in all_cols:
                    out[col]=out[col].astype("float32")
                f=part_path(parts,str(row.file_id))
                temp=f.with_suffix(".tmp")
                out.to_parquet(temp,index=False,compression="zstd")
                os.replace(temp,f)
                count+=1
                if count<=3 or count%25==0:
                    print(f"EXTRACTED NEW {count}/{len(pending)} | existing {len(selected)-len(pending)}",flush=True)
    if args.limit is not None:
        print("SMOKE extract done; resume with --mode extract without --limit",flush=True)
        return
    arr=[]
    for row in selected.itertuples(index=False):
        f=part_path(parts,str(row.file_id))
        if not f.exists():
            raise RuntimeError(f"Missing train augmentation checkpoint: {row.file_id}")
        arr.append(pd.read_parquet(f))
    aug=pd.concat(arr,ignore_index=True)
    if len(aug)!=34*args.per_class*len(CONDITIONS_TRAIN):
        raise RuntimeError("Unexpected number of augmentation features")
    if aug.duplicated(["file_id","condition"]).any() or set(aug.file_id)!=set(selected.file_id):
        raise RuntimeError("Augmented feature records do not match selected TRAIN IDs")
    final=dst / "train_augmented_features.parquet"
    aug.to_parquet(final,index=False,compression="zstd")
    print("EXTRACTION COMPLETE",final,"rows",len(aug),flush=True)


def paired_bootstrap(predictions, model_names, n_rep=2000):
    rng=np.random.default_rng(42)
    # Select exactly ONE row per recording: clean condition of the original model.
    # predictions has both clean_trained and augmented_trained rows, so selecting
    # only by condition would duplicate all 680 test IDs and invalidate strata.
    test_ids=predictions.loc[
        predictions.condition.eq("clean") & predictions.model.eq("clean_trained")
    ].sort_values("file_id")
    if test_ids.file_id.isna().any() or not test_ids.file_id.is_unique:
        raise RuntimeError("Duplicate or missing clean reference test IDs")
    ordered=test_ids.file_id.to_numpy()
    true=test_ids.generator.to_numpy()
    classes=np.sort(np.unique(true))
    strata=[np.flatnonzero(true==c) for c in classes]
    if len(ordered)!=680 or any(len(x)!=20 for x in strata):
        raise RuntimeError("Bootstrap requires 20 unique test recordings for each of 34 generators")
    test_map={(condition,model): predictions.loc[(predictions.condition==condition)&(predictions.model==model)].set_index("file_id").reindex(ordered) for condition in CONDITIONS for model in model_names}
    results=[]
    for condition in CONDITIONS:
        clean=test_map[(condition,"clean_trained")]; aug=test_map[(condition,"augmented_trained")]
        if clean.index.has_duplicates or aug.index.has_duplicates:
            raise RuntimeError("Duplicate test prediction IDs")
        if clean.predicted_generator.isna().any() or aug.predicted_generator.isna().any():
            raise RuntimeError("Incomplete prediction alignment")
        y1=clean.predicted_generator.to_numpy();y2=aug.predicted_generator.to_numpy()
        f1a=float(f1_score(true,y1,labels=classes,average="macro",zero_division=0)); f1b=float(f1_score(true,y2,labels=classes,average="macro",zero_division=0))
        boots=np.empty((n_rep,3), dtype=float)
        for i in range(n_rep):
            idx=np.concatenate([rng.choice(s,size=len(s),replace=True) for s in strata])
            a=f1_score(true[idx], y1[idx], labels=classes, average="macro",zero_division=0)
            b=f1_score(true[idx], y2[idx], labels=classes, average="macro",zero_division=0)
            boots[i]=[a,b,b-a]
        results.append({"condition":condition,"clean_trained_macro_f1":f1a,"augmented_trained_macro_f1":f1b,
          "delta_macro_f1_aug_minus_clean":f1b-f1a,"delta_macro_f1_95ci":np.percentile(boots[:,2],[2.5,97.5]).tolist(),
          "clean_macro_f1_95ci":np.percentile(boots[:,0],[2.5,97.5]).tolist(),"aug_macro_f1_95ci":np.percentile(boots[:,1],[2.5,97.5]).tolist(),
          "clean_trained_accuracy":float(accuracy_score(true,y1)),"augmented_trained_accuracy":float(accuracy_score(true,y2))})
    return results


def train(args):
    sp, feats, test, old_model, all_cols, cols = preflight()
    dst=root(args)
    selected=select_train(sp,args.seed,args.per_class)
    save_manifest(dst/"selected_train_records.csv",selected)
    aug_path=dst/"train_augmented_features.parquet"
    if not aug_path.exists():
        raise FileNotFoundError(f"Run --mode extract to completion first: {aug_path}")
    aug=pd.read_parquet(aug_path)
    if len(aug)!=34*args.per_class*2 or aug.duplicated(["file_id","condition"]).any():
        raise RuntimeError("Invalid augmented train feature table")
    train_id=set(sp.loc[sp.split.eq("train")].file_id)
    if not set(aug.file_id)==set(selected.file_id) or not set(aug.file_id).issubset(train_id):
        raise RuntimeError("Augmented rows contain non-train audio")
    allowed=selected.set_index("file_id").generator
    if not aug.generator.astype(str).eq(aug.file_id.map(allowed)).all() or set(aug.condition)!=set(CONDITIONS_TRAIN):
        raise RuntimeError("Augmentation labels or conditions mismatch")
    data=feats.set_index("file_id")
    train_rows=sp.loc[sp.split.eq("train"), ["file_id", "generator"]]
    val_rows=sp.loc[sp.split.eq("validation"), ["file_id", "generator"]]
    X_clean=data.loc[train_rows.file_id,cols].reset_index(drop=True)
    X_aug=aug[cols].reset_index(drop=True)
    X_train=pd.concat([X_clean,X_aug],ignore_index=True).replace([np.inf,-np.inf],np.nan).astype("float32")
    y_train=pd.concat([train_rows.generator.reset_index(drop=True),aug.generator.reset_index(drop=True)],ignore_index=True).astype(str)
    # Exactly half total effective sample mass is assigned to new noisy derivatives.
    # This is predeclared, class balanced, and not tuned on TEST.
    w=np.concatenate([np.ones(len(X_clean), dtype=np.float64), np.full(len(X_aug), len(X_clean)/len(X_aug),dtype=np.float64)])
    X_val=data.loc[val_rows.file_id,cols].replace([np.inf,-np.inf],np.nan).astype("float32")
    y_val=val_rows.generator.astype(str)
    le=LabelEncoder();ytr=le.fit_transform(y_train); yv=le.transform(y_val)
    old_classes=np.array(old_model.classes_)
    if not np.array_equal(le.classes_,old_classes):
        raise RuntimeError("Class order mismatch vs original clean-trained LightGBM")
    settings_all=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    params=dict(settings_all["model"]["params"])
    params.setdefault("num_class",len(le.classes_));params["random_state"]=args.seed;params["n_jobs"]=-1
    run_info={"protocol":"RuASD robustness_training_v1_noise30+noise20", "frozen_split_sha256":SPLIT_HASH,
              "train_clean_rows":len(X_clean),"train_augmented_rows":len(X_aug),"train_parent_records":len(selected),
              "augmentation_conditions":list(CONDITIONS_TRAIN),"effective_weight_share_clean":0.5,"effective_weight_share_augmented":0.5,
              "selection_seed":args.seed,"features_789_no_duration":True,"validation_clean_only":True,
              "test_conditions":list(CONDITIONS),"test_records":680,"test_used_for_training":False,
              "baseline_model_path":str(REF/"clean_789_model.joblib"),"train_feature_table":str(aug_path),
              "feature_extraction_config":str(CONFIG),"lightgbm_params":params, "seed":args.seed}
    dst.mkdir(parents=True,exist_ok=True)
    (dst/"protocol.json").write_text(json.dumps(run_info,ensure_ascii=False,indent=2),encoding="utf8")
    model_file=dst/"augmented_789_model.joblib"
    if model_file.exists() and not args.force_retrain:
        print("LOADING EXISTING AUGMENTED TRAINING MODEL", model_file,flush=True)
        trained=joblib.load(model_file)
    else:
        print("TRAINING 789 FEATURE LightGBM | clean=",len(X_clean),"aug=",len(X_aug),"effective_weights=50/50",flush=True)
        estimator=lgb.LGBMClassifier(**params)
        estimator.fit(X_train,ytr,sample_weight=w,eval_set=[(X_val,yv)],eval_metric="multi_logloss",callbacks=[lgb.early_stopping(int(settings_all["training"]["early_stopping_rounds"]),verbose=False)])
        trained=TrainedClassifier(estimator,le.classes_,le,"lightgbm")
        joblib.dump(trained,model_file,compress=3)
        print("TRAINED best_iteration_",estimator.best_iteration_,"SAVED",model_file,flush=True)
    if not np.array_equal(trained.classes_,old_classes):
        raise RuntimeError("Prediction classes mismatch")
    # Strictly test the *same precomputed* distorted audio as prior clean-trained study.
    rows=[];summary=[]
    for condition in CONDITIONS:
        subset=test.loc[test.condition.eq(condition)].sort_values("file_id")
        x=subset[cols].replace([np.inf,-np.inf],np.nan)
        for name, model in [("clean_trained",old_model),("augmented_trained",trained)]:
            pred=model.predict(x)
            result={"condition":condition,"model":name,"n":len(subset),"macro_f1":float(f1_score(subset.generator,pred,labels=old_classes,average="macro",zero_division=0)),
               "accuracy":float(accuracy_score(subset.generator,pred)), "balanced_accuracy":float(balanced_accuracy_score(subset.generator,pred))}
            print("RESULT",result,flush=True)
            summary.append(result)
            rows.append(pd.DataFrame({"file_id":subset.file_id.to_numpy(),"generator":subset.generator.to_numpy(),"condition":condition,"model":name,"predicted_generator":pred}))
    predictions=pd.concat(rows,ignore_index=True)
    predictions.to_parquet(dst/"test_predictions.parquet",index=False)
    summary_df=pd.DataFrame(summary);summary_df.to_csv(dst/"condition_summary.csv",index=False)
    pairs=paired_bootstrap(predictions,["clean_trained","augmented_trained"])
    pd.DataFrame(pairs).to_csv(dst/"paired_robustness_bootstrap.csv",index=False)
    output={"protocol":run_info,"summary":summary,"paired_bootstrap":pairs,"bootstrap_repetitions":2000,"bootstrap_seed":42,
        "uncertainty_scope":"recording-level, fixed models, 680 fixed test recordings; no seed/corpus/speaker uncertainty"}
    (dst/"metrics.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf8")
    print("ROBUSTNESS TRAINING COMPLETE",dst/"metrics.json",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=["preflight","extract","train"],required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--per-class",type=int,default=50,help="Selected TRAIN parent recordings per generator (two noise variants each)")
    ap.add_argument("--limit",type=int,default=None,help="Only for EXTRACT smoke; do not produce complete augmented feature store")
    ap.add_argument("--force-retrain",action="store_true",help="Recompute model from unchanged derived training features")
    args=ap.parse_args()
    if args.mode=="preflight":preflight()
    elif args.mode=="extract":extract(args)
    else:train(args)

if __name__=="__main__":main()
