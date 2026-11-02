
#!/usr/bin/env python
from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path
from source_attribution.ruasd_shards import resolve_ruasd_shard

import numpy as np
import pandas as pd
import torch
import yaml

from source_attribution.audio import (
    preprocess_audio,
    fixed_length,
)
from source_attribution.ssl_embeddings import (
    SSLConfig,
    SSLExtractor,
)


SPLITS = Path("splits/closed_set.csv")
CONFIG = Path("configs/closed_set_ssl.yaml")

N_RECORDS = 8
BATCH_SIZE = 2
MIN_DURATION = 4.1


def main():
    total_start = time.perf_counter()

    print("=== ENVIRONMENT ===", flush=True)
    print("Torch:", torch.__version__, flush=True)
    print("CUDA:", torch.cuda.is_available(), flush=True)
    print("CPU threads:", torch.get_num_threads(), flush=True)

    cfg = yaml.safe_load(
        CONFIG.read_text(encoding="utf-8")
    )
    sc = cfg["ssl"]

    model_id = sc["model_id"]
    sample_rate = int(sc.get("sample_rate", 16000))
    max_seconds = float(sc.get("max_seconds", 4.0))

    print("Model:", model_id, flush=True)
    print("Sample rate:", sample_rate, flush=True)
    print("Max seconds:", max_seconds, flush=True)

    if max_seconds != 4.0:
        raise RuntimeError(
            "Expected 4-second SSL configuration"
        )

    print("\n=== SELECTING TEST RECORDS ===", flush=True)

    df = pd.read_csv(
        SPLITS,
        usecols=[
            "file_id",
            "generator",
            "split",
            "shard_path",
            "audio_member",
            "audio_duration",
        ],
    )

    test = df[df["split"] == "test"].copy()
    test["audio_duration"] = pd.to_numeric(
        test["audio_duration"], errors="coerce"
    )

    test = test[
        test["audio_duration"] >= MIN_DURATION
    ].copy()

    if len(test) < N_RECORDS:
        raise RuntimeError(
            "Not enough recordings for smoke test"
        )

    # Choose a populated TAR to minimize archive overhead.
    shard = test["shard_path"].value_counts().idxmax()

    selected = (
        test[test["shard_path"] == shard]
        .sort_values("file_id")
        .head(N_RECORDS)
        .copy()
    )

    if len(selected) != N_RECORDS:
        raise RuntimeError(
            "Selected TAR has insufficient records"
        )

    print("TAR:", shard, flush=True)
    print("Records:", len(selected), flush=True)
    print(
        "Generators:",
        selected["generator"].nunique(),
        flush=True,
    )

    path = resolve_ruasd_shard(shard)
    if not path.is_file():
        raise FileNotFoundError(path)

    print("\n=== READING AUDIO FROM TAR ===", flush=True)

    read_start = time.perf_counter()
    audios = []

    with tarfile.open(path, "r:*") as tf:
        for row in selected.itertuples(index=False):
            member = tf.getmember(
                str(row.audio_member)
            )

            fh = tf.extractfile(member)
            if fh is None:
                raise RuntimeError(
                    f"Could not read {row.audio_member}"
                )

            audio_bytes = fh.read()

            y, sr, duration = preprocess_audio(
                io.BytesIO(audio_bytes),
                target_sr=sample_rate,
                mono=True,
                peak_normalize=True,
                max_duration_sec=max_seconds,
            )

            # No padding is expected in this test.
            if len(y) != int(
                sample_rate * max_seconds
            ):
                raise RuntimeError(
                    f"Unexpected audio length: {len(y)}"
                )

            y = fixed_length(
                y,
                sr,
                max_seconds,
                mode="start",
            )

            if not np.isfinite(y).all():
                raise RuntimeError(
                    "Non-finite audio detected"
                )

            audios.append(y)

    print(
        "Audio reading time:",
        round(time.perf_counter() - read_start, 2),
        "seconds",
        flush=True,
    )

    print("\n=== LOADING WAVLM ===", flush=True)

    model_start = time.perf_counter()

    extractor = SSLExtractor(
        SSLConfig(
            model_id=model_id,
            revision=sc.get("revision"),
            sample_rate=sample_rate,
            pooling="mean",
            device="cpu",
        )
    )

    print(
        "Model initialization time:",
        round(time.perf_counter() - model_start, 2),
        "seconds",
        flush=True,
    )

    print("\n=== EXTRACTING EMBEDDINGS ===", flush=True)

    embedding_start = time.perf_counter()
    outputs = []

    for start in range(0, len(audios), BATCH_SIZE):
        batch = audios[start:start + BATCH_SIZE]

        t0 = time.perf_counter()
        emb = extractor.encode_batch(batch)

        if not np.isfinite(emb).all():
            raise RuntimeError(
                "Non-finite embeddings detected"
            )

        outputs.append(emb)

        print(
            f"Batch {start // BATCH_SIZE + 1}: "
            f"{len(batch)} recordings, "
            f"{time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    embeddings = np.concatenate(outputs, axis=0)

    elapsed = time.perf_counter() - embedding_start

    print("\n=== SSL SMOKE TEST RESULTS ===", flush=True)
    print("Model:", model_id)
    print("Device: CPU")
    print("Recordings:", len(embeddings))
    print("Embedding shape:", embeddings.shape)
    print("Extraction time:", round(elapsed, 2), "s")
    print(
        "Seconds per recording:",
        round(elapsed / len(embeddings), 3),
    )
    print(
        "Estimated hours for 6800 recordings:",
        round(elapsed / len(embeddings) * 6800 / 3600, 2),
    )
    print(
        "Total test time:",
        round(time.perf_counter() - total_start, 2),
        "s",
    )
    print("\nSSL SMOKE TEST COMPLETED", flush=True)


if __name__ == "__main__":
    main()
