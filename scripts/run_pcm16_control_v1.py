
import argparse
import hashlib
import io
import os
import tarfile
from pathlib import Path
from source_attribution.ruasd_shards import resolve_ruasd_shard

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from sklearn.metrics import accuracy_score, f1_score
from scripts import run_robustness_main_v1 as m


BASE = Path("results/raw/robustness_main_v1")
OUT = Path("results/raw/robustness_pcm16_control_v1")
PARTS = OUT / "features_by_file"


def checkpoint_path(file_id):
    digest = hashlib.sha256(
        file_id.encode("utf-8")
    ).hexdigest()[:24]

    return PARTS / (digest + ".parquet")


def checkpoint_valid(path, file_id, generator, names):
    if not path.exists():
        return False

    try:
        frame = pd.read_parquet(path)

        return (
            len(frame) == 1
            and frame.iloc[0]["file_id"] == file_id
            and frame.iloc[0]["generator"] == generator
            and all(name in frame.columns for name in names)
        )
    except Exception:
        return False


def pcm16_roundtrip(x, sr):
    buffer = io.BytesIO()

    sf.write(
        buffer,
        x,
        sr,
        format="WAV",
        subtype="PCM_16",
    )

    buffer.seek(0)

    result, decoded_sr = sf.read(
        buffer,
        dtype="float32",
        always_2d=False,
    )

    if decoded_sr != sr:
        raise RuntimeError("PCM16 sample rate changed")

    if result.shape != x.shape:
        raise RuntimeError("PCM16 audio shape changed")

    return result


def report(label, actual, predicted):
    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(actual, predicted)

    print(
        label,
        "MACRO_F1:", round(macro_f1, 6),
        "ACCURACY:", round(accuracy, 6),
        "PREDICTED_CLASSES:", len(set(predicted)),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(
        BASE / "selected_test_records.csv"
    )

    required = {
        "file_id",
        "generator",
        "shard_path",
        "audio_member",
    }

    if not required.issubset(selected.columns):
        raise RuntimeError("Missing columns in selected records")

    if len(selected) != 680:
        raise RuntimeError("Expected 680 test records")

    if selected.generator.nunique() != 34:
        raise RuntimeError("Expected 34 generators")

    if selected.file_id.duplicated().any():
        raise RuntimeError("Duplicate file IDs")

    selected = selected.sort_values(
        ["shard_path", "audio_member"]
    ).reset_index(drop=True)

    model = joblib.load(
        BASE / "clean_789_model.joblib"
    )

    names = list(model.estimator.feature_name_)

    if len(names) != 789:
        raise RuntimeError("Expected 789 classifier features")

    with open(
        "configs/closed_set_acoustic.yaml",
        encoding="utf-8",
    ) as handle:
        settings = yaml.safe_load(handle)[
            "feature_extraction"
        ]

    cfg = m.build_feature_config(settings)

    run_rows = selected

    if args.max_records is not None:
        run_rows = selected.head(args.max_records)

    print("PCM16 CONTROL EXPERIMENT", flush=True)
    print("TOTAL SELECTED:", len(selected), flush=True)
    print("PROCESSING:", len(run_rows), flush=True)
    print("CLASSIFIER FEATURES:", len(names), flush=True)

    completed = 0
    reused = 0

    for shard_path, group in run_rows.groupby(
        "shard_path",
        sort=False,
    ):
        pending = []

        for row in group.itertuples(index=False):
            path = checkpoint_path(row.file_id)

            if checkpoint_valid(
                path,
                row.file_id,
                row.generator,
                names,
            ):
                reused += 1
                completed += 1
            else:
                pending.append(row)

        if not pending:
            continue

        with tarfile.open(resolve_ruasd_shard(shard_path), "r:*") as archive:
            for row in pending:
                member = archive.extractfile(
                    row.audio_member
                )

                if member is None:
                    raise RuntimeError(row.audio_member)

                audio_bytes = member.read()

                x, native_sr = m._read_audio(
                    io.BytesIO(audio_bytes),
                    mono=True,
                )

                x = m._resample(
                    x,
                    native_sr,
                    cfg.sr,
                )

                original_duration = len(x) / cfg.sr

                x = m.trim_or_pad_audio(
                    x,
                    cfg.sr,
                    max_duration_sec=settings[
                        "max_duration_sec"
                    ],
                    min_audio_sec=settings[
                        "min_audio_sec"
                    ],
                )

                pcm = pcm16_roundtrip(x, cfg.sr)

                x64 = x.astype(np.float64)
                pcm64 = pcm.astype(np.float64)

                error = pcm64 - x64

                signal_power = np.mean(x64 ** 2)
                error_power = np.mean(error ** 2)

                pcm_snr = 10 * np.log10(
                    max(signal_power, 1e-30)
                    / max(error_power, 1e-30)
                )

                clipping_fraction = float(
                    np.mean(np.abs(x64) >= 1.0)
                )

                changed = m.normalize_audio(
                    pcm,
                    peak_normalize=settings[
                        "peak_normalize"
                    ],
                )

                extracted = m.extract_acoustic_features(
                    changed,
                    cfg.sr,
                    original_duration=original_duration,
                    cfg=cfg,
                )

                record = {
                    "file_id": row.file_id,
                    "generator": row.generator,
                    "input_peak": float(
                        np.max(np.abs(x64))
                    ),
                    "clipping_fraction": clipping_fraction,
                    "pcm16_snr_db": float(pcm_snr),
                }

                for name in names:
                    record[name] = float(extracted[name])

                path = checkpoint_path(row.file_id)
                temporary = path.with_name(
                    path.name + ".tmp"
                )

                pd.DataFrame([record]).to_parquet(
                    temporary,
                    index=False,
                )

                os.replace(temporary, path)

                completed += 1

                if (
                    completed % 10 == 0
                    or completed == len(run_rows)
                ):
                    print(
                        "PROCESSED",
                        str(completed) + "/" + str(len(run_rows)),
                        row.generator,
                        flush=True,
                    )

    print(
        "CHECKPOINTS COMPLETE:",
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

    print("LOADING PCM16 FEATURES", flush=True)

    parts = [
        pd.read_parquet(checkpoint_path(file_id))
        for file_id in selected.file_id
    ]

    frame = pd.concat(
        parts,
        ignore_index=True,
    )

    actual = frame["generator"].to_numpy()

    pcm_predictions = np.asarray(
        model.predict(frame[names])
    )

    original_predictions = pd.read_parquet(
        BASE / "predictions.parquet"
    )

    clean = original_predictions.loc[
        original_predictions.condition.eq("clean")
    ].set_index("file_id")

    mp3 = original_predictions.loc[
        original_predictions.condition.eq("mp3_64kbps")
    ].set_index("file_id")

    clean = clean.loc[frame.file_id]
    mp3 = mp3.loc[frame.file_id]

    if not np.array_equal(
        actual,
        clean.generator.to_numpy(),
    ):
        raise RuntimeError("Clean labels do not match")

    if not np.array_equal(
        actual,
        mp3.generator.to_numpy(),
    ):
        raise RuntimeError("MP3 labels do not match")

    clean_predictions = clean[
        "predicted_generator"
    ].to_numpy()

    mp3_predictions = mp3[
        "predicted_generator"
    ].to_numpy()

    print("\nCOMPARATIVE EVALUATION", flush=True)

    report("CLEAN", actual, clean_predictions)
    report("PCM16_ONLY", actual, pcm_predictions)
    report("MP3_64KBPS", actual, mp3_predictions)

    print(
        "PCM16 CHANGED VS CLEAN:",
        int(np.sum(
            pcm_predictions != clean_predictions
        )),
    )

    print(
        "MP3 CHANGED VS PCM16:",
        int(np.sum(
            mp3_predictions != pcm_predictions
        )),
    )

    print(
        "PCM16 SNR MIN:",
        round(float(frame.pcm16_snr_db.min()), 4),
    )

    print(
        "PCM16 SNR MEDIAN:",
        round(float(frame.pcm16_snr_db.median()), 4),
    )

    print(
        "MAX INPUT PEAK:",
        round(float(frame.input_peak.max()), 6),
    )

    print(
        "RECORDS WITH CLIPPING:",
        int(np.sum(frame.clipping_fraction > 0)),
    )

    frame["pcm16_prediction"] = pcm_predictions
    frame["clean_prediction"] = clean_predictions
    frame["mp3_prediction"] = mp3_predictions

    frame[
        [
            "file_id",
            "generator",
            "clean_prediction",
            "pcm16_prediction",
            "mp3_prediction",
            "input_peak",
            "clipping_fraction",
            "pcm16_snr_db",
        ]
    ].to_csv(
        OUT / "predictions_comparison.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "condition": label,
                "macro_f1": f1_score(
                    actual,
                    pred,
                    average="macro",
                    zero_division=0,
                ),
                "accuracy": accuracy_score(
                    actual,
                    pred,
                ),
            }
            for label, pred in [
                ("clean", clean_predictions),
                ("pcm16_only", pcm_predictions),
                ("mp3_64kbps", mp3_predictions),
            ]
        ]
    )

    summary.to_csv(
        OUT / "metrics_comparison.csv",
        index=False,
    )

    print(
        "\nPCM16 CONTROL EXPERIMENT PASSED",
        flush=True,
    )

    print("RESULTS SAVED:", OUT, flush=True)


if __name__ == "__main__":
    main()
