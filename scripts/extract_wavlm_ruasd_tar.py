
#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
import transformers

from source_attribution.audio import (
    fixed_length,
    preprocess_audio,
)
from source_attribution.ssl_embeddings import (
    SSLConfig,
    SSLExtractor,
)


SPLITS = Path("splits/closed_set.csv")
CONFIG = Path("configs/closed_set_ssl.yaml")
DEFAULT_ROOT = Path("embeddings/wavlm_base_ruasd_4s")
DATASET_ROOT = Path(os.environ.get("RUASD_ROOT", "data/raw/RuASD"))
PROTOCOL = "ruasd_wavlm_fixed4s_start_peaknorm_v1"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_ids(rows):
    text = "\n".join(
        rows["file_id"].astype(str).tolist()
    )
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def save_json_atomic(path, data):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(
        json.dumps(
            data, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Process only the first N TAR shards.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ROOT),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DATASET_ROOT),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started = perf_counter()

    root = Path(args.output_dir)
    chunks = root / "chunks"
    metadata = root / "metadata"

    chunks.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(
        CONFIG.read_text(encoding="utf-8")
    )
    sc = cfg["ssl"]

    model_id = str(sc["model_id"])
    sample_rate = int(
        sc.get("sample_rate", 16000)
    )
    max_seconds = float(
        sc.get("max_seconds", 4.0)
    )
    batch_size = int(
        sc.get("batch_size", 8)
    )

    if model_id != "microsoft/wavlm-base":
        raise RuntimeError(
            f"Unexpected model: {model_id}"
        )
    if sample_rate != 16000 or max_seconds != 4.0:
        raise RuntimeError(
            "Expected 16 kHz / 4 seconds"
        )
    if sc.get("pooling", "mean") != "mean":
        raise RuntimeError(
            "Expected mean pooling"
        )
    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive"
        )

    print("LOADING FROZEN SPLITS", flush=True)

    df = pd.read_csv(
        SPLITS,
        usecols=[
            "file_id",
            "generator",
            "split",
            "shard_name",
            "audio_member",
        ],
    )

    if len(df) != 68000:
        raise RuntimeError(
            f"Expected 68000 rows, got {len(df)}"
        )
    if df["file_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate file_id in splits"
        )
    if df.isna().any().any():
        raise RuntimeError(
            "Missing required metadata"
        )
    if df["generator"].nunique() != 34:
        raise RuntimeError(
            "Expected 34 generators"
        )

    expected_splits = {
        "train": 44200,
        "validation": 10200,
        "test": 13600,
    }

    if df["split"].value_counts().to_dict() != expected_splits:
        raise RuntimeError(
            "Unexpected split distribution"
        )

    df = df.sort_values(
        ["shard_name", "file_id"]
    ).reset_index(drop=True)

    groups = list(
        df.groupby("shard_name", sort=True)
    )

    if len(groups) != 120:
        raise RuntimeError(
            f"Expected 120 shards, got {len(groups)}"
        )

    print(
        "LOADING FROZEN WAVLM-BASE",
        flush=True,
    )

    extractor = SSLExtractor(
        SSLConfig(
            model_id=model_id,
            revision=sc.get("revision"),
            sample_rate=sample_rate,
            pooling="mean",
            device="cpu",
        )
    )

    commit_hash = getattr(
        extractor.model.config,
        "_commit_hash",
        None,
    )

    if commit_hash is None:
        print(
            "WARNING: resolved model commit unavailable",
            flush=True,
        )

    settings = {
        "protocol": PROTOCOL,
        "model_id": model_id,
        "configured_revision": sc.get("revision"),
        "resolved_model_commit": commit_hash,
        "sample_rate": sample_rate,
        "max_seconds": max_seconds,
        "crop": "first_4_seconds",
        "short_clip_padding": "right_zero",
        "normalization": (
            "DC removal and peak normalization "
            "before WavLM feature extraction"
        ),
        "pooling": "unmasked_mean",
        "embedding_dimension": 768,
        "splits_sha256": sha256_file(SPLITS),
        "config_sha256": sha256_file(CONFIG),
        "audio_module_sha256": sha256_file(
            Path("src/source_attribution/audio.py")
        ),
        "ssl_module_sha256": sha256_file(
            Path(
                "src/source_attribution/"
                "ssl_embeddings.py"
            )
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    signature = hashlib.sha256(
        json.dumps(
            settings, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    manifest_path = root / "manifest.json"

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
        if previous.get("signature") != signature:
            raise RuntimeError(
                "Existing embeddings use a different "
                "model or preprocessing protocol. "
                "Do not mix incompatible results."
            )
    else:
        save_json_atomic(
            manifest_path,
            {
                "signature": signature,
                "settings": settings,
                "total_expected_records": 68000,
                "total_expected_shards": 120,
            },
        )

    if args.max_shards is not None:
        if args.max_shards < 1:
            raise ValueError(
                "--max-shards must be positive"
            )
        groups = groups[:args.max_shards]

    embedding_columns = [
        f"emb_{i:04d}" for i in range(768)
    ]

    total_processed = 0
    total_cached = 0
    total_padded = 0

    print(
        f"SHARDS TO VISIT: {len(groups)}",
        flush=True,
    )
    print(
        f"BATCH SIZE: {batch_size}",
        flush=True,
    )
    print(
        f"OUTPUT: {root}",
        flush=True,
    )

    for shard_index, (shard_name, rows) in enumerate(
        groups, 1
    ):
        shard_start = perf_counter()
        rows = rows.reset_index(drop=True)

        stem = Path(shard_name).stem

        output = chunks / f"{stem}.parquet"
        meta_path = metadata / f"{stem}.json"

        ids_hash = fingerprint_ids(rows)

        if output.exists() and meta_path.exists():
            old_meta = json.loads(
                meta_path.read_text(encoding="utf-8")
            )

            cached_rows = pq.read_metadata(
                output
            ).num_rows

            if (
                old_meta.get("signature") == signature
                and old_meta.get("ids_sha256") == ids_hash
                and old_meta.get("records") == len(rows)
                and cached_rows == len(rows)
            ):
                total_cached += len(rows)
                print(
                    f"[{shard_index}/{len(groups)}] "
                    f"{shard_name}: CACHED "
                    f"({len(rows)} records)",
                    flush=True,
                )
                continue

            raise RuntimeError(
                f"Incompatible existing cache: {output}"
            )

        shard_path = (
            Path(args.dataset_root) / shard_name
        )
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)

        print(
            f"[{shard_index}/{len(groups)}] "
            f"{shard_name}: {len(rows)} records",
            flush=True,
        )

        embedding_batches = []
        padded_count = 0
        audio_seconds = 0.0
        model_seconds = 0.0

        with tarfile.open(shard_path, "r:*") as tf:
            for offset in range(
                0, len(rows), batch_size
            ):
                batch = rows.iloc[
                    offset:offset + batch_size
                ]
                audios = []

                t_audio = perf_counter()

                for row in batch.itertuples(index=False):
                    member_name = str(
                        row.audio_member
                    )

                    member = tf.getmember(
                        member_name
                    )

                    handle = tf.extractfile(member)
                    if handle is None:
                        raise RuntimeError(
                            f"Missing audio: {member_name}"
                        )

                    with handle:
                        audio_bytes = handle.read()

                    y, sr, duration = preprocess_audio(
                        io.BytesIO(audio_bytes),
                        target_sr=sample_rate,
                        mono=True,
                        peak_normalize=True,
                        max_duration_sec=max_seconds,
                    )

                    if duration < max_seconds:
                        padded_count += 1

                    y = fixed_length(
                        y,
                        sr,
                        max_seconds,
                        mode="start",
                    )

                    if len(y) != int(
                        sample_rate * max_seconds
                    ):
                        raise RuntimeError(
                            "Wrong audio length"
                        )

                    if not np.isfinite(y).all():
                        raise RuntimeError(
                            f"Non-finite audio: {row.file_id}"
                        )

                    audios.append(y)

                audio_seconds += (
                    perf_counter() - t_audio
                )

                t_model = perf_counter()

                embeddings = extractor.encode_batch(
                    audios
                )

                model_seconds += (
                    perf_counter() - t_model
                )

                if embeddings.shape != (
                    len(batch), 768
                ):
                    raise RuntimeError(
                        f"Wrong embedding shape: "
                        f"{embeddings.shape}"
                    )

                if not np.isfinite(
                    embeddings
                ).all():
                    raise RuntimeError(
                        "Non-finite SSL embeddings"
                    )

                embedding_batches.append(
                    embeddings.astype(
                        np.float32, copy=False
                    )
                )

                finished = offset + len(batch)
                if (
                    finished == len(rows)
                    or finished // 200
                    != offset // 200
                ):
                    print(
                        f"  {finished}/{len(rows)}",
                        flush=True,
                    )

        matrix = np.concatenate(
            embedding_batches,
            axis=0,
        )

        if matrix.shape != (len(rows), 768):
            raise RuntimeError(
                "Final shard dimensions mismatch"
            )

        frame = rows[
            [
                "file_id",
                "generator",
                "split",
                "shard_name",
            ]
        ].copy()

        embedding_frame = pd.DataFrame(
            matrix,
            columns=embedding_columns,
        )

        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                embedding_frame,
            ],
            axis=1,
        )

        temp_path = chunks / f"{stem}.tmp.parquet"

        frame.to_parquet(
            temp_path,
            index=False,
        )

        if pq.read_metadata(
            temp_path
        ).num_rows != len(rows):
            raise RuntimeError(
                "Parquet row count mismatch"
            )

        os.replace(temp_path, output)

        elapsed = perf_counter() - shard_start

        save_json_atomic(
            meta_path,
            {
                "signature": signature,
                "shard_name": shard_name,
                "ids_sha256": ids_hash,
                "records": len(rows),
                "padded_records": padded_count,
                "audio_seconds": round(
                    audio_seconds, 3
                ),
                "model_seconds": round(
                    model_seconds, 3
                ),
                "total_seconds": round(
                    elapsed, 3
                ),
            },
        )

        total_processed += len(rows)
        total_padded += padded_count

        print(
            f"  SAVED: {output}",
            flush=True,
        )
        print(
            f"  TIME: {elapsed:.1f}s "
            f"(audio {audio_seconds:.1f}s, "
            f"WavLM {model_seconds:.1f}s)",
            flush=True,
        )

    print(
        "\n========== EXTRACTION SUMMARY ==========",
        flush=True,
    )
    print(
        "NEW RECORDS:", total_processed,
        flush=True,
    )
    print(
        "CACHED RECORDS:", total_cached,
        flush=True,
    )
    print(
        "NEW PADDED RECORDS:", total_padded,
        flush=True,
    )
    print(
        "TOTAL TIME:",
        round(perf_counter() - started, 1),
        "seconds",
        flush=True,
    )
    print(
        "OUTPUT DIRECTORY:",
        root,
        flush=True,
    )
    print(
        "WAVLM EXTRACTION RUN COMPLETED",
        flush=True,
    )


if __name__ == "__main__":
    main()
