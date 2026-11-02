
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
import yaml
from sklearn.metrics import accuracy_score, f1_score

from scripts import run_robustness_main_v1 as m
from scripts import run_pcm16_control_v1 as pc


BASE = Path("results/raw/robustness_main_v1")
OUT = Path("results/raw/robustness_safe_codec_control_v1")
PARTS = OUT / "features_by_file"

CONDITIONS = ("pcm16_safe", "mp3_safe")


def checkpoint_path(file_id):
    digest = hashlib.sha256(
        file_id.encode("utf-8")
    ).hexdigest()[:24]
    return PARTS / (digest + ".parquet")


def checkpoint_valid(path, file_id, names):
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
        return (
            len(frame) == 2
            and set(frame["condition"]) == set(CONDITIONS)
            and set(frame["file_id"]) == {file_id}
            and all(name in frame.columns for name in names)
        )
    except Exception:
        return False


def evaluate(label, y, pred):
    score = f1_score(
        y, pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y, pred)
    print(
        label,
        "MACRO_F1:", round(score, 6),
        "ACCURACY:", round(accuracy, 6),
        "PREDICTED_CLASSES:", len(set(pred)),
        flush=True,
    )
    return {
        "condition": label,
        "macro_f1": score,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")

    PARTS.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(
        BASE / "selected_test_records.csv"
    )

    if len(selected) != 680:
        raise RuntimeError("Expected 680 records")
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
        raise RuntimeError("Expected 789 features")

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

    total = len(run_rows)
    completed = 0
    reused = 0

    print("SAFE CODEC CONTROL", flush=True)
    print("RECORDS:", total, flush=True)
    print("GENERATORS:", selected.generator.nunique())
    print("FEATURES:", len(names), flush=True)
    print("CONDITIONS:", CONDITIONS, flush=True)

    for shard_path, group in run_rows.groupby(
        "shard_path", sort=False
    ):
        pending = [
            row
            for row in group.itertuples(index=False)
            if not checkpoint_valid(
                checkpoint_path(row.file_id),
                row.file_id,
                names,
            )
        ]

        reused += len(group) - len(pending)
        completed += len(group) - len(pending)

        if not pending:
            continue

        with tarfile.open(resolve_ruasd_shard(shard_path), "r:*") as archive:
            for row in pending:
                member = archive.extractfile(
                    row.audio_member
                )
                if member is None:
                    raise RuntimeError(row.audio_member)

                x, native_sr = m._read_audio(
                    io.BytesIO(member.read()),
                    mono=True,
                )

                x = m._resample(
                    x, native_sr, cfg.sr
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

                peak = float(np.max(np.abs(x)))
                if not np.isfinite(peak):
                    raise RuntimeError("Invalid audio peak")

                if peak > 0:
                    safe = x * (0.95 / peak)
                else:
                    safe = x.copy()

                safe_peak = float(
                    np.max(np.abs(safe))
                )

                if safe_peak > 0.951:
                    raise RuntimeError(
                        "Pre-encoding peak exceeds 0.95"
                    )

                # Verify that scaling before the
                # existing peak normalization does
                # not materially change clean audio.
                ref = m.normalize_audio(
                    x,
                    peak_normalize=settings[
                        "peak_normalize"
                    ],
                )
                safe_ref = m.normalize_audio(
                    safe,
                    peak_normalize=settings[
                        "peak_normalize"
                    ],
                )

                difference = float(
                    np.max(
                        np.abs(
                            np.asarray(ref, dtype=np.float64)
                            - np.asarray(
                                safe_ref, dtype=np.float64
                            )
                        )
                    )
                )

                if difference > 1e-4:
                    raise RuntimeError(
                        "Clean normalization mismatch: "
                        + str(difference)
                    )

                records = []

                for condition in CONDITIONS:
                    if condition == "pcm16_safe":
                        changed = pc.pcm16_roundtrip(
                            safe, cfg.sr
                        )
                    else:
                        changed = m.mp3_roundtrip(
                            safe, cfg.sr
                        )

                    changed = m.normalize_audio(
                        changed,
                        peak_normalize=settings[
                            "peak_normalize"
                        ],
                    )

                    features = m.extract_acoustic_features(
                        changed,
                        cfg.sr,
                        original_duration=original_duration,
                        cfg=cfg,
                    )

                    record = {
                        "file_id": row.file_id,
                        "generator": row.generator,
                        "condition": condition,
                        "input_peak": peak,
                        "safe_peak": safe_peak,
                        "clean_waveform_max_diff": difference,
                    }

                    for name in names:
                        record[name] = float(
                            features[name]
                        )

                    records.append(record)

                path = checkpoint_path(row.file_id)
                temporary = path.with_name(
                    path.name + ".tmp"
                )

                pd.DataFrame(records).to_parquet(
                    temporary, index=False
                )
                os.replace(temporary, path)

                completed += 1
                if (
                    completed % 10 == 0
                    or completed == total
                ):
                    print(
                        "PROCESSED",
                        str(completed) + "/" + str(total),
                        row.generator,
                        flush=True,
                    )

    print(
        "CHECKPOINTS COMPLETE:",
        completed,
        "REUSED:", reused,
        flush=True,
    )

    if args.max_records is not None:
        print(
            "SMOKE TEST PASSED. "
            "Run without --max-records for full evaluation.",
            flush=True,
        )
        return

    print("\nLOADING FEATURES", flush=True)

    frames = [
        pd.read_parquet(
            checkpoint_path(file_id)
        )
        for file_id in selected.file_id
    ]

    result = pd.concat(
        frames, ignore_index=True
    )

    old = pd.read_parquet(
        BASE / "predictions.parquet"
    )
    clean = old[
        old.condition.eq("clean")
    ].set_index("file_id")

    old_mp3 = old[
        old.condition.eq("mp3_64kbps")
    ].set_index("file_id")

    clean = clean.loc[selected.file_id]
    old_mp3 = old_mp3.loc[selected.file_id]

    actual = selected.generator.to_numpy()
    if not np.array_equal(
        actual, clean.generator.to_numpy()
    ):
        raise RuntimeError("Clean labels mismatch")

    if not np.array_equal(
        actual, old_mp3.generator.to_numpy()
    ):
        raise RuntimeError("MP3 labels mismatch")

    predictions = {}
    for condition in CONDITIONS:
        subset = result[
            result.condition.eq(condition)
        ].set_index("file_id")

        subset = subset.loc[selected.file_id]
        if not np.array_equal(
            actual, subset.generator.to_numpy()
        ):
            raise RuntimeError("Labels mismatch")

        predictions[condition] = np.asarray(
            model.predict(subset[names])
        )

    clean_pred = clean[
        "predicted_generator"
    ].to_numpy()

    old_mp3_pred = old_mp3[
        "predicted_generator"
    ].to_numpy()

    old_pcm = pd.read_csv(
        "results/raw/robustness_pcm16_control_v1/"
        "predictions_comparison.csv"
    ).set_index("file_id").loc[selected.file_id]

    if not np.array_equal(
        actual, old_pcm.generator.to_numpy()
    ):
        raise RuntimeError("PCM labels mismatch")

    old_pcm_pred = old_pcm[
        "pcm16_prediction"
    ].to_numpy()

    print("\nCOMPARATIVE EVALUATION", flush=True)

    metrics = []
    for label, pred in [
        ("clean", clean_pred),
        ("pcm16_original", old_pcm_pred),
        ("mp3_original", old_mp3_pred),
        ("pcm16_safe", predictions["pcm16_safe"]),
        ("mp3_safe", predictions["mp3_safe"]),
    ]:
        metrics.append(evaluate(label, actual, pred))

    print(
        "SAFE PCM CHANGED VS CLEAN:",
        int(np.sum(
            predictions["pcm16_safe"] != clean_pred
        )),
    )
    print(
        "SAFE MP3 CHANGED VS SAFE PCM:",
        int(np.sum(
            predictions["mp3_safe"]
            != predictions["pcm16_safe"]
        )),
    )

    print(
        "MAX SAFE INPUT PEAK:",
        round(float(result.safe_peak.max()), 6),
    )
    print(
        "MAX CLEAN WAVEFORM DIFFERENCE:",
        float(result.clean_waveform_max_diff.max()),
    )

    pd.DataFrame(metrics).to_csv(
        OUT / "metrics_comparison.csv",
        index=False,
    )

    pd.DataFrame({
        "file_id": selected.file_id,
        "generator": actual,
        "clean_prediction": clean_pred,
        "pcm16_original_prediction": old_pcm_pred,
        "mp3_original_prediction": old_mp3_pred,
        "pcm16_safe_prediction": predictions[
            "pcm16_safe"
        ],
        "mp3_safe_prediction": predictions[
            "mp3_safe"
        ],
        "original_input_peak": old_pcm.input_peak.to_numpy(),
    }).to_csv(
        OUT / "predictions_comparison.csv",
        index=False,
    )

    print(
        "\nSAFE CODEC CONTROL PASSED",
        flush=True,
    )
    print("RESULTS SAVED:", OUT, flush=True)


if __name__ == "__main__":
    main()
