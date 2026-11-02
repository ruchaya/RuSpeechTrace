
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from scripts.extract_acoustic_features import (
    load_settings,
    build_feature_config,
    feature_signature,
    sha256_file,
)
from source_attribution.audio import preprocess_audio
from source_attribution.acoustic_features import extract_acoustic_features


SPLIT = Path("splits/mlaad_ru_23class_split_v1.csv")
OLD_FEATURES = Path("features/mlaad_three_family_acoustic.parquet")
OLD_META = Path("features/mlaad_three_family_acoustic.meta.json")
CONFIG = Path("configs/closed_set_acoustic.yaml")
RU_FEATURES = Path("features/ruasd_closed_set_acoustic.parquet")
RU_META = Path("features/ruasd_closed_set_acoustic.meta.json")

ROOT = Path(
    os.environ.get("MLAAD_ROOT", "data/external/MLAAD_fake_ru")
)

OUTPUT = Path("features/mlaad_23class_acoustic.parquet")
BATCH_SIZE = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)

    settings = load_settings(CONFIG)
    cfg = build_feature_config(settings)
    signature, _ = feature_signature(settings, cfg)

    ru_meta = json.loads(RU_META.read_text(encoding="utf-8"))
    old_meta = json.loads(OLD_META.read_text(encoding="utf-8"))

    assert signature == ru_meta["feature_signature"]
    assert signature == old_meta["feature_signature"]

    names = [
        col for col in pq.read_schema(RU_FEATURES).names
        if col.startswith(("spectral_", "prosody_"))
    ]

    assert len(names) == 790

    full = pd.read_csv(
        SPLIT,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    ).sort_values("path").reset_index(drop=True)

    assert len(full) == 23000
    assert full["path"].is_unique
    assert full["generator"].nunique() == 23
    assert full.groupby("generator").size().eq(1000).all()

    for key in ("text_key", "original_file"):
        assert full.groupby(key)["split"].nunique().max() == 1

    old = pd.read_parquet(
        OLD_FEATURES,
        columns=["path"] + names,
    )

    assert len(old) == 3000
    assert old["path"].is_unique
    assert set(old["path"]).issubset(set(full["path"]))

    d = full.copy()

    if args.limit is not None:
        d = d.head(args.limit).copy()
        out = Path(
            f"features/mlaad_23class_smoke_{args.limit}.parquet"
        )
    else:
        out = OUTPUT

    if out.exists():
        raise RuntimeError(
            f"Output already exists; refusing to overwrite: {out}"
        )

    # Existing 3000 rows are reused without recalculation.
    reusable = old.loc[
        old["path"].isin(d["path"])
    ].copy()

    missing = d.loc[
        ~d["path"].isin(old["path"])
    ].sort_values("path").reset_index(drop=True)

    split_hash = sha256_file(SPLIT)

    cache = (
        Path("features/cache/mlaad_23class_v1")
        / signature[:16]
        / split_hash[:16]
    )
    cache.mkdir(parents=True, exist_ok=True)

    print("TOTAL SELECTED:", len(d))
    print("REUSABLE OLD FEATURES:", len(reusable))
    print("NEW WAVS TO PROCESS:", len(missing))
    print("FEATURE COUNT:", len(names))
    print("CACHE DIRECTORY:", cache)

    cached_parts = []
    computed = 0
    cache_reused = 0

    for start in tqdm(
        range(0, len(missing), BATCH_SIZE),
        desc="MLAAD 23-class batches",
    ):
        batch = missing.iloc[start:start + BATCH_SIZE]
        cp = cache / f"batch_{start:06d}.parquet"

        valid = False

        if cp.is_file():
            saved = pd.read_parquet(cp)
            valid = (
                len(saved) == len(batch)
                and saved["path"].tolist() == batch["path"].tolist()
                and all(col in saved.columns for col in names)
            )

        if valid:
            cache_reused += len(batch)
        else:
            records = []

            for row in batch.to_dict("records"):
                wav = ROOT / str(row["path"])

                if not wav.is_file():
                    raise FileNotFoundError(wav)

                y, sr, duration = preprocess_audio(
                    str(wav),
                    target_sr=cfg.sr,
                    mono=True,
                    peak_normalize=bool(
                        settings["peak_normalize"]
                    ),
                    max_duration_sec=float(
                        settings["max_duration_sec"]
                    ),
                    min_audio_sec=float(
                        settings["min_audio_sec"]
                    ),
                )

                features = extract_acoustic_features(
                    y,
                    sr,
                    original_duration=duration,
                    cfg=cfg,
                )

                if list(features) != names:
                    raise RuntimeError(
                        f"Feature columns mismatch: {wav}"
                    )

                records.append({
                    "path": row["path"],
                    **features,
                })

                computed += 1

            saved = pd.DataFrame.from_records(records)

            for col in names:
                saved[col] = saved[col].astype("float32")

            temp = cp.with_suffix(".tmp.parquet")
            saved.to_parquet(
                temp,
                index=False,
                compression="zstd",
            )
            temp.replace(cp)

        cached_parts.append(cp)

    pieces = [reusable]

    for cp in cached_parts:
        pieces.append(
            pd.read_parquet(cp, columns=["path"] + names)
        )

    combined = pd.concat(pieces, ignore_index=True)

    assert len(combined) == len(d)
    assert combined["path"].is_unique
    assert set(combined["path"]) == set(d["path"])

    result = d.merge(
        combined,
        on="path",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    assert len(result) == len(d)

    if result[names].isna().all(axis=1).any():
        raise RuntimeError("Missing complete feature rows")

    out.parent.mkdir(parents=True, exist_ok=True)

    temp = out.with_suffix(".tmp.parquet")
    result.to_parquet(
        temp,
        index=False,
        compression="zstd",
    )
    temp.replace(out)

    metadata = {
        "rows": len(result),
        "generators": int(result["generator"].nunique()),
        "feature_columns": len(names),
        "feature_signature": signature,
        "split_sha256": split_hash,
        "config_sha256": sha256_file(CONFIG),
        "old_features_sha256": sha256_file(OLD_FEATURES),
        "old_features_reused": len(reusable),
        "newly_computed": computed,
        "cache_reused": cache_reused,
        "output_sha256": sha256_file(out),
    }

    out.with_suffix(".meta.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("OUTPUT:", out)
    print("ROWS:", len(result))
    print("GENERATORS:", result["generator"].nunique())
    print("FEATURES:", len(names))
    print("OLD FEATURES REUSED:", len(reusable))
    print("NEWLY COMPUTED:", computed)
    print("CACHE REUSED:", cache_reused)
    print("SHA256:", metadata["output_sha256"])
    print("EXTRACTION PASSED")


if __name__ == "__main__":
    main()
