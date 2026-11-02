from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from source_attribution.ruasd_shards import resolve_ruasd_shard

import joblib
import numpy as np
import pandas as pd
import yaml

from scripts.extract_acoustic_features import build_feature_config
from source_attribution.audio import (
    _read_audio,
    _resample,
    trim_or_pad_audio,
    normalize_audio,
)
from source_attribution.acoustic_features import extract_acoustic_features
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics


ROOT = Path("results/raw/robustness_pilot_v1")
SEED = 42
N_PER_GENERATOR = 2
CONDITIONS = ("clean", "noise_20db", "synthetic_echo")


def distort(x, condition, file_id, sr):
    x = np.asarray(x, dtype=np.float32)

    if condition == "clean":
        return x.copy()

    if condition == "noise_20db":
        key = f"{file_id}|{condition}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")

        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(x)).astype(np.float32)
        noise -= noise.mean()

        noise_rms = np.sqrt(np.mean(noise ** 2))
        signal_rms = np.sqrt(np.mean(x ** 2))

        target_rms = signal_rms / (10 ** (20 / 20))
        noise *= target_rms / max(noise_rms, 1e-12)

        return x + noise

    if condition == "synthetic_echo":
        result = x.copy()

        for delay_sec, gain in (
            (0.055, 0.50),
            (0.110, 0.25),
            (0.190, 0.12),
        ):
            delay = int(round(delay_sec * sr))

            if delay < len(x):
                result[delay:] += gain * x[:-delay]

        return result

    raise ValueError(condition)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(
        Path("configs/closed_set_acoustic.yaml").read_text(
            encoding="utf-8"
        )
    )

    settings = config["feature_extraction"]
    feature_cfg = build_feature_config(settings)

    splits = pd.read_csv(
        "splits/closed_set.csv",
        dtype={"file_id": str},
    )

    features = pd.read_parquet(
        "features/ruasd_closed_set_acoustic.parquet"
    )

    assert splits.file_id.is_unique
    assert features.file_id.is_unique

    cols = [
        c for c in features.columns
        if c.startswith(("spectral_", "prosody_"))
    ]

    assert len(cols) == 790, len(cols)

    feat = features.set_index("file_id")
    labels = splits.set_index("file_id")["generator"]

    train_ids = splits.loc[
        splits.split.eq("train"), "file_id"
    ].tolist()

    val_ids = splits.loc[
        splits.split.eq("validation"), "file_id"
    ].tolist()

    test_rows = splits.loc[
        splits.split.eq("test")
    ].copy()

    assert set(train_ids).isdisjoint(test_rows.file_id)
    assert set(val_ids).isdisjoint(test_rows.file_id)

    selected = (
        test_rows.groupby("generator", group_keys=False)
        .sample(n=N_PER_GENERATOR, random_state=SEED)
        .sort_values(["shard_path", "audio_member"])
        .reset_index(drop=True)
    )

    assert selected.generator.nunique() == 34
    assert len(selected) == 34 * N_PER_GENERATOR

    print("GENERATORS:", selected.generator.nunique(), flush=True)
    print("SELECTED TEST RECORDS:", len(selected), flush=True)
    print("FEATURES:", len(cols), flush=True)

    records = []
    matching = 0

    print("\nEXTRACTING CLEAN AND DISTORTED FEATURES", flush=True)

    for shard_path, group in selected.groupby("shard_path"):
        with tarfile.open(resolve_ruasd_shard(shard_path), "r:*") as tf:
            for row in group.itertuples(index=False):
                member = tf.extractfile(row.audio_member)

                if member is None:
                    raise RuntimeError(row.audio_member)

                audio_bytes = member.read()

                x, native_sr = _read_audio(
                    io.BytesIO(audio_bytes),
                    mono=True,
                )

                x = _resample(
                    x,
                    native_sr,
                    feature_cfg.sr,
                )

                original_duration = len(x) / feature_cfg.sr

                x = trim_or_pad_audio(
                    x,
                    feature_cfg.sr,
                    max_duration_sec=settings["max_duration_sec"],
                    min_audio_sec=settings["min_audio_sec"],
                )

                for condition in CONDITIONS:
                    changed = distort(
                        x,
                        condition,
                        row.file_id,
                        feature_cfg.sr,
                    )

                    changed = normalize_audio(
                        changed,
                        peak_normalize=settings["peak_normalize"],
                    )

                    extracted = extract_acoustic_features(
                        changed,
                        feature_cfg.sr,
                        original_duration=original_duration,
                        cfg=feature_cfg,
                    )

                    if condition == "clean":
                        original = feat.loc[row.file_id, cols]

                        a = np.asarray(
                            [extracted[c] for c in cols],
                            dtype=float,
                        )
                        b = original.to_numpy(dtype=float)

                        equal = np.isclose(
                            a,
                            b,
                            rtol=1e-5,
                            atol=1e-5,
                            equal_nan=True,
                        )

                        if not equal.all():
                            raise AssertionError(
                                f"Clean feature mismatch: "
                                f"{row.file_id}, "
                                f"{equal.sum()}/{len(equal)}"
                            )

                        matching += 1

                    record = {
                        "file_id": row.file_id,
                        "generator": row.generator,
                        "condition": condition,
                    }
                    record.update(extracted)

                    records.append(record)

                print(
                    f"PROCESSED {matching}/{len(selected)} "
                    f"{row.generator}",
                    flush=True,
                )

    print(
        "\nCLEAN FEATURES VERIFIED:",
        matching,
        "/",
        len(selected),
        flush=True,
    )

    data = pd.DataFrame(records)

    assert len(data) == len(selected) * len(CONDITIONS)

    data[cols] = data[cols].astype("float32")
    data.to_parquet(
        ROOT / "features.parquet",
        index=False,
        compression="zstd",
    )

    print("\nTRAINING CLEAN LIGHTGBM MODEL", flush=True)

    X_train = feat.loc[train_ids, cols].replace(
        [np.inf, -np.inf], np.nan
    )
    X_val = feat.loc[val_ids, cols].replace(
        [np.inf, -np.inf], np.nan
    )

    model = build_and_train(
        X_train,
        labels.loc[train_ids],
        X_val,
        labels.loc[val_ids],
        model_type=config["model"]["type"],
        params=config["model"]["params"],
        early_stopping_rounds=config["training"][
            "early_stopping_rounds"
        ],
        seed=SEED,
    )

    joblib.dump(
        model,
        ROOT / "clean_model.joblib",
        compress=3,
    )

    print("\nPILOT RESULTS", flush=True)

    results = {}
    prediction_rows = []

    for condition in CONDITIONS:
        subset = data.loc[
            data.condition.eq(condition)
        ].copy()

        X = subset[cols].replace(
            [np.inf, -np.inf], np.nan
        )

        y_true = subset.generator.to_numpy()
        y_pred = model.predict(X)
        probabilities = model.predict_proba(X)

        metrics = closed_set_metrics(
            y_true,
            y_pred,
            probabilities,
            model.classes_,
        )

        results[condition] = {
            "n": len(subset),
            "macro_f1": float(metrics["macro_f1"]),
            "balanced_accuracy": float(
                metrics["balanced_accuracy"]
            ),
            "accuracy": float(metrics["accuracy"]),
        }

        prediction_rows.append(
            pd.DataFrame({
                "file_id": subset.file_id.to_numpy(),
                "generator": y_true,
                "condition": condition,
                "predicted_generator": y_pred,
            })
        )

        print(
            condition,
            "N:",
            len(subset),
            "MACRO_F1:",
            round(metrics["macro_f1"], 6),
            "BAL_ACC:",
            round(metrics["balanced_accuracy"], 6),
            flush=True,
        )

    clean_f1 = results["clean"]["macro_f1"]

    for condition in CONDITIONS:
        results[condition]["delta_macro_f1"] = (
            results[condition]["macro_f1"] - clean_f1
        )

    pd.concat(
        prediction_rows,
        ignore_index=True,
    ).to_parquet(
        ROOT / "predictions.parquet",
        index=False,
    )

    summary = {
        "protocol": "clean-trained, paired test perturbations",
        "seed": SEED,
        "generators": 34,
        "records_per_generator": N_PER_GENERATOR,
        "features": len(cols),
        "conditions": list(CONDITIONS),
        "clean_feature_matches": matching,
        "results": results,
    }

    (ROOT / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSAVED:", ROOT)
    print("ROBUSTNESS PILOT PASSED")


if __name__ == "__main__":
    main()