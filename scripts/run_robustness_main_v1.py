from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from source_attribution.ruasd_shards import resolve_ruasd_shard

import imageio_ffmpeg
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from scripts.extract_acoustic_features import build_feature_config
from source_attribution.audio import (
    _read_audio,
    _resample,
    trim_or_pad_audio,
    normalize_audio,
)
from source_attribution.acoustic_features import extract_acoustic_features
from source_attribution.models import build_and_train


ROOT = Path("results/raw/robustness_main_v1")
PARTS = ROOT / "features_by_file"
MODEL_PATH = ROOT / "clean_789_model.joblib"

SEED = 42
N_PER_GENERATOR = 20

CONDITIONS = (
    "clean",
    "noise_30db",
    "noise_20db",
    "noise_10db",
    "synthetic_echo",
    "mp3_64kbps",
)


def deterministic_rng(file_id, condition):
    key = f"{file_id}|{condition}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed)


def add_noise(x, snr_db, file_id, condition):
    rng = deterministic_rng(file_id, condition)

    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise -= noise.mean()

    signal_rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    noise_rms = np.sqrt(np.mean(noise.astype(np.float64) ** 2))

    target_noise_rms = signal_rms / (10 ** (snr_db / 20))

    noise *= target_noise_rms / max(noise_rms, 1e-12)

    return x + noise


def add_echo(x, sr):
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


def mp3_roundtrip(x, sr):
    exe = imageio_ffmpeg.get_ffmpeg_exe()

    wav_buffer = io.BytesIO()

    sf.write(
        wav_buffer,
        x,
        sr,
        format="WAV",
        subtype="PCM_16",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        mp3_path = Path(temp_dir) / "audio.mp3"

        encoded = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", "pipe:0",
                "-ac", "1",
                "-ar", str(sr),
                "-c:a", "libmp3lame",
                "-b:a", "64k",
                str(mp3_path),
            ],
            input=wav_buffer.getvalue(),
            capture_output=True,
        )

        if encoded.returncode != 0:
            raise RuntimeError(
                "MP3 encoding failed: "
                + encoded.stderr.decode(errors="replace")
            )

        decoded = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(mp3_path),
                "-ac", "1",
                "-ar", str(sr),
                "-c:a", "pcm_f32le",
                "-f", "f32le",
                "pipe:1",
            ],
            capture_output=True,
        )

        if decoded.returncode != 0:
            raise RuntimeError(
                "MP3 decoding failed: "
                + decoded.stderr.decode(errors="replace")
            )

    result = np.frombuffer(
        decoded.stdout,
        dtype="<f4",
    ).copy()

    extra_samples = len(result) - len(x)

    if extra_samples < 0 or extra_samples > 1152:
        raise RuntimeError(
            f"Unexpected MP3 length change: "
            f"{len(x)} -> {len(result)}"
        )

    if extra_samples > 0:
        result = result[:len(x)]

    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite MP3 samples")

    return result


def apply_condition(x, condition, file_id, sr):
    if condition == "clean":
        return x.copy()

    if condition.startswith("noise_"):
        snr_db = float(
            condition.replace("noise_", "").replace("db", "")
        )

        return add_noise(x, snr_db, file_id, condition)

    if condition == "synthetic_echo":
        return add_echo(x, sr)

    if condition == "mp3_64kbps":
        return mp3_roundtrip(x, sr)

    raise ValueError(condition)


def checkpoint_path(file_id):
    digest = hashlib.sha256(
        file_id.encode("utf-8")
    ).hexdigest()[:24]

    return PARTS / f"{digest}.parquet"


def checkpoint_valid(path, file_id):
    if not path.exists():
        return False

    try:
        data = pd.read_parquet(
            path,
            columns=["file_id", "condition"],
        )

        return (
            len(data) == len(CONDITIONS)
            and data.file_id.eq(file_id).all()
            and set(data.condition) == set(CONDITIONS)
        )

    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Process only first N selected records for testing",
    )

    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    settings_all = yaml.safe_load(
        Path("configs/closed_set_acoustic.yaml").read_text(
            encoding="utf-8"
        )
    )

    settings = settings_all["feature_extraction"]
    cfg = build_feature_config(settings)

    splits = pd.read_csv(
        "splits/closed_set.csv",
        dtype={"file_id": str},
    )

    features = pd.read_parquet(
        "features/ruasd_closed_set_acoustic.parquet"
    )

    assert splits.file_id.is_unique
    assert features.file_id.is_unique

    feature_cols = [
        c for c in features.columns
        if c.startswith(("spectral_", "prosody_"))
    ]

    model_cols = [
        c for c in feature_cols
        if c != "prosody_duration_sec"
    ]

    assert len(feature_cols) == 790
    assert len(model_cols) == 789

    feature_table = features.set_index("file_id")
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

    assert set(train_ids).isdisjoint(set(test_rows.file_id))
    assert set(val_ids).isdisjoint(set(test_rows.file_id))

    selected = (
        test_rows.groupby("generator", group_keys=False)
        .sample(
            n=N_PER_GENERATOR,
            random_state=SEED,
        )
        .loc[:, [
            "file_id",
            "generator",
            "shard_path",
            "audio_member",
        ]]
        .sort_values("file_id")
        .reset_index(drop=True)
    )

    assert len(selected) == 680
    assert selected.generator.nunique() == 34
    assert selected.file_id.is_unique

    manifest_path = ROOT / "selected_test_records.csv"

    if manifest_path.exists():
        previous = pd.read_csv(
            manifest_path,
            dtype={"file_id": str},
        )

        if not previous.equals(selected):
            raise RuntimeError(
                "Frozen selection changed. "
                "Refusing to mix experiment versions."
            )
    else:
        selected.to_csv(manifest_path, index=False)

    print("TEST RECORDS:", len(selected), flush=True)
    print("GENERATORS:", selected.generator.nunique(), flush=True)
    print("EXTRACTED FEATURES:", len(feature_cols), flush=True)
    print("CLASSIFIER FEATURES:", len(model_cols), flush=True)
    print("CONDITIONS:", CONDITIONS, flush=True)

    if MODEL_PATH.exists():
        print("\nLOADING SAVED 789-FEATURE MODEL", flush=True)
        model = joblib.load(MODEL_PATH)

    else:
        print("\nTRAINING CLEAN 789-FEATURE MODEL", flush=True)

        X_train = feature_table.loc[
            train_ids, model_cols
        ].replace([np.inf, -np.inf], np.nan)

        X_val = feature_table.loc[
            val_ids, model_cols
        ].replace([np.inf, -np.inf], np.nan)

        model = build_and_train(
            X_train,
            labels.loc[train_ids],
            X_val,
            labels.loc[val_ids],
            model_type=settings_all["model"]["type"],
            params=settings_all["model"]["params"],
            early_stopping_rounds=settings_all["training"][
                "early_stopping_rounds"
            ],
            seed=SEED,
        )

        joblib.dump(model, MODEL_PATH, compress=3)
        print("MODEL SAVED:", MODEL_PATH, flush=True)

    model_names = list(getattr(model, "feature_name_", []))

    if model_names and model_names != model_cols:
        raise RuntimeError("Saved model feature order mismatch")

    if len(model.classes_) != 34:
        raise RuntimeError("Unexpected model class count")

    run_rows = selected.sort_values(
        ["shard_path", "audio_member"]
    )

    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")

        run_rows = run_rows.head(args.max_records)

    total = len(run_rows)
    completed = 0
    reused = 0

    print("\nPROCESSING AUDIO", flush=True)

    for shard_path, group in run_rows.groupby(
        "shard_path",
        sort=False,
    ):
        pending = [
            row for row in group.itertuples(index=False)
            if not checkpoint_valid(
                checkpoint_path(row.file_id),
                row.file_id,
            )
        ]

        reused += len(group) - len(pending)
        completed += len(group) - len(pending)

        if not pending:
            continue

        with tarfile.open(resolve_ruasd_shard(shard_path), "r:*") as archive:
            for row in pending:
                member = archive.extractfile(row.audio_member)

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
                    cfg.sr,
                )

                original_duration = len(x) / cfg.sr

                x = trim_or_pad_audio(
                    x,
                    cfg.sr,
                    max_duration_sec=settings["max_duration_sec"],
                    min_audio_sec=settings["min_audio_sec"],
                )

                records = []

                for condition in CONDITIONS:
                    changed = apply_condition(
                        x,
                        condition,
                        row.file_id,
                        cfg.sr,
                    )

                    changed = normalize_audio(
                        changed,
                        peak_normalize=settings["peak_normalize"],
                    )

                    extracted = extract_acoustic_features(
                        changed,
                        cfg.sr,
                        original_duration=original_duration,
                        cfg=cfg,
                    )

                    if condition == "clean":
                        reference = feature_table.loc[
                            row.file_id, feature_cols
                        ].to_numpy(dtype=float)

                        generated = np.array(
                            [extracted[c] for c in feature_cols],
                            dtype=float,
                        )

                        if not np.isclose(
                            reference,
                            generated,
                            rtol=1e-5,
                            atol=1e-5,
                            equal_nan=True,
                        ).all():
                            raise RuntimeError(
                                "Clean feature mismatch: "
                                + row.file_id
                            )

                    record = {
                        "file_id": row.file_id,
                        "generator": row.generator,
                        "condition": condition,
                    }

                    record.update(extracted)
                    records.append(record)

                part = checkpoint_path(row.file_id)
                temporary = part.with_suffix(".tmp")

                output = pd.DataFrame(records)
                output[feature_cols] = output[
                    feature_cols
                ].astype("float32")

                output.to_parquet(
                    temporary,
                    index=False,
                    compression="zstd",
                )

                os.replace(temporary, part)

                completed += 1

                if (
                    completed % 10 == 0
                    or completed == total
                    or args.max_records is not None
                ):
                    print(
                        f"PROCESSED {completed}/{total}: "
                        f"{row.generator}",
                        flush=True,
                    )

    print(
        "\nCHECKPOINTS COMPLETE:",
        completed,
        "REUSED:",
        reused,
        flush=True,
    )

    if args.max_records is not None:
        print(
            "SMOKE TEST PASSED. "
            "Run without --max-records for full evaluation.",
            flush=True,
        )
        return

    print("\nLOADING ALL CHECKPOINTS", flush=True)

    parts = []

    for row in selected.itertuples(index=False):
        path = checkpoint_path(row.file_id)

        if not checkpoint_valid(path, row.file_id):
            raise RuntimeError(
                f"Missing or invalid checkpoint: {row.file_id}"
            )

        parts.append(pd.read_parquet(path))

    data = pd.concat(parts, ignore_index=True)

    assert len(data) == 680 * len(CONDITIONS)

    assert not data.duplicated(
        ["file_id", "condition"]
    ).any()

    data.to_parquet(
        ROOT / "all_features.parquet",
        index=False,
        compression="zstd",
    )

    print("\nEVALUATING CONDITIONS", flush=True)

    summary = {}
    prediction_parts = []

    for condition in CONDITIONS:
        subset = data.loc[
            data.condition.eq(condition)
        ].sort_values("file_id")

        X = subset[model_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        y_true = subset.generator.to_numpy()
        y_pred = model.predict(X)

        metrics = {
            "n": int(len(subset)),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro")
            ),
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "predicted_classes": int(
                len(set(y_pred))
            ),
        }

        summary[condition] = metrics

        prediction_parts.append(
            pd.DataFrame({
                "file_id": subset.file_id.to_numpy(),
                "generator": y_true,
                "condition": condition,
                "predicted_generator": y_pred,
            })
        )

        print(
            condition,
            "MACRO_F1:",
            round(metrics["macro_f1"], 6),
            "ACCURACY:",
            round(metrics["accuracy"], 6),
            "PREDICTED_CLASSES:",
            metrics["predicted_classes"],
            flush=True,
        )

    clean_f1 = summary["clean"]["macro_f1"]

    for condition in CONDITIONS:
        summary[condition]["delta_macro_f1"] = (
            summary[condition]["macro_f1"] - clean_f1
        )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    predictions.to_parquet(
        ROOT / "predictions.parquet",
        index=False,
    )

    by_generator = predictions.assign(
        correct=lambda d: (
            d.generator == d.predicted_generator
        )
    ).groupby(
        ["condition", "generator"],
        as_index=False,
    ).agg(
        n=("correct", "size"),
        recall=("correct", "mean"),
    )

    by_generator.to_csv(
        ROOT / "per_generator_recall.csv",
        index=False,
    )

    metadata = {
        "protocol": "paired clean-trained robustness",
        "seed": SEED,
        "generators": 34,
        "records_per_generator": N_PER_GENERATOR,
        "total_test_records": 680,
        "extracted_features": 790,
        "classifier_features": 789,
        "duration_feature_excluded": True,
        "noise_type": "white Gaussian",
        "echo_delays_seconds": [0.055, 0.110, 0.190],
        "echo_gains": [0.50, 0.25, 0.12],
        "mp3_bitrate": "64k",
        "conditions": list(CONDITIONS),
        "results": summary,
    }

    (ROOT / "metrics.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRESULTS SAVED:", ROOT, flush=True)
    print("ROBUSTNESS MAIN EXPERIMENT PASSED", flush=True)


if __name__ == "__main__":
    main()