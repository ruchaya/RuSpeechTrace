#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

import source_attribution.acoustic_features as acoustic_module
import source_attribution.audio as audio_module
from source_attribution.acoustic_features import (
    AcousticFeatureConfig,
    FEATURE_DEFINITION_VERSION,
    extract_acoustic_features,
)
from source_attribution.audio import preprocess_audio, fixed_length


META_COLUMNS = [
    "file_id",
    "sample_id",
    "generator",
    "generator_family",
    "checkpoint",
    "speaker",
    "voice",
    "text_hash",
    "duration",
    "sample_rate",
    "channels",
    "raw_or_augmented",
    "augmentation_type",
    "source_audio",
    "source_dataset",
    "split",
    "shard_name",
    "audio_member",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract resumable legacy-compatible acoustic features from RuASD TAR shards."
    )
    p.add_argument(
        "--splits",
        default="splits/closed_set.csv",
        help="Frozen split CSV containing shard_name and audio_member.",
    )
    p.add_argument(
        "--dataset-root",
        default="data/raw/RuASD",
        help="Directory containing ruasd-*.tar.",
    )
    p.add_argument(
        "--config",
        default="configs/closed_set_acoustic.yaml",
        help="YAML containing feature_extraction settings.",
    )
    p.add_argument(
        "--output",
        default="features/ruasd_closed_set_acoustic.parquet",
    )
    p.add_argument(
        "--cache-root",
        default="features/cache/acoustic",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test only: process the first N split rows.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing per-shard cache for the selected rows.",
    )
    return p.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def module_sha256(module) -> str:
    path = Path(module.__file__)
    return sha256_file(path)


def load_settings(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = cfg.get("feature_extraction", {})
    if not section:
        raise RuntimeError(
            f"{path} has no 'feature_extraction' section."
        )
    return section


def build_feature_config(settings: dict) -> AcousticFeatureConfig:
    return AcousticFeatureConfig(
        sr=int(settings.get("sample_rate", 16000)),
        n_fft=int(settings.get("n_fft", 512)),
        win_length=int(settings.get("win_length", 400)),
        hop_length=int(settings.get("hop_length", 160)),
        n_mfcc=int(settings.get("n_mfcc", 20)),
        n_lfcc=int(settings.get("n_lfcc", 20)),
        n_cqcc=int(settings.get("n_cqcc", 20)),
        n_lfcc_filters=int(settings.get("n_lfcc_filters", 40)),
        cqcc_hop_length=int(settings.get("cqcc_hop_length", 256)),
        cqcc_n_bins=int(settings.get("cqcc_n_bins", 72)),
        cqcc_bins_per_octave=int(
            settings.get("cqcc_bins_per_octave", 12)
        ),
        f0_min_hz=float(settings.get("f0_min_hz", 60.0)),
        f0_max_hz=float(settings.get("f0_max_hz", 500.0)),
        f0_frame_length=int(settings.get("f0_frame_length", 1024)),
        energy_frame_length=int(
            settings.get("energy_frame_length", 400)
        ),
        pause_percentile=float(settings.get("pause_percentile", 25.0)),
        pause_threshold_multiplier=float(
            settings.get("pause_threshold_multiplier", 1.5)
        ),
        pause_threshold_floor=float(
            settings.get("pause_threshold_floor", 1e-4)
        ),
        pause_min_duration_sec=float(
            settings.get("pause_min_duration_sec", 0.10)
        ),
        include_duration=bool(settings.get("include_duration", True)),
    )


def feature_signature(
    settings: dict,
    cfg: AcousticFeatureConfig,
) -> tuple[str, dict]:
    payload = {
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "settings": settings,
        "resolved_feature_config": asdict(cfg),
        "acoustic_features_py_sha256": module_sha256(acoustic_module),
        "audio_py_sha256": module_sha256(audio_module),
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(raw)[:16], payload


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("spectral_") or c.startswith("prosody_")
    ]


def process_one_shard(
    shard_name: str,
    rows: pd.DataFrame,
    dataset_root: Path,
    cache_path: Path,
    cfg: AcousticFeatureConfig,
    settings: dict,
    force: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    existing = pd.DataFrame()
    if cache_path.exists() and not force:
        existing = pd.read_parquet(cache_path)

    existing_ids = (
        set(existing["file_id"].astype(str))
        if not existing.empty and "file_id" in existing.columns
        else set()
    )

    needed = rows[
        ~rows["file_id"].astype(str).isin(existing_ids)
    ].copy()

    errors: list[dict] = []
    new_records: list[dict] = []

    if not needed.empty:
        shard_path = dataset_root / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)

        with tarfile.open(shard_path, "r:*") as tf:
            member_map = {
                m.name: m for m in tf.getmembers() if m.isfile()
            }

            for row in needed.itertuples(index=False):
                file_id = str(row.file_id)
                member_name = str(row.audio_member)

                try:
                    member = member_map.get(member_name)
                    if member is None:
                        raise KeyError(
                            f"Missing TAR member: {member_name}"
                        )

                    fh = tf.extractfile(member)
                    if fh is None:
                        raise RuntimeError(
                            f"tar.extractfile returned None: {member_name}"
                        )

                    # BytesIO is deliberately used for robust seek behaviour in libsndfile.
                    audio_bytes = fh.read()
                    y, sr, original_duration = preprocess_audio(
                        io.BytesIO(audio_bytes),
                        target_sr=cfg.sr,
                        mono=True,
                        peak_normalize=bool(
                            settings.get("peak_normalize", True)
                        ),
                        max_duration_sec=(
                            None
                            if settings.get("max_duration_sec") is None
                            else float(settings["max_duration_sec"])
                        ),
                        min_audio_sec=float(
                            settings.get("min_audio_sec", 0.10)
                        ),
                    )
                    fixed_sec = settings.get("fixed_length_sec")
                    if fixed_sec is not None:
                        y = fixed_length(
                            y,
                            sr,
                            float(fixed_sec),
                            mode=str(
                                settings.get(
                                    "fixed_length_mode",
                                    "start",
                                )
                            ),
                        )
                    feats = extract_acoustic_features(
                        y,
                        sr,
                        original_duration=original_duration,
                        cfg=cfg,
                    )

                    rec = {
                        c: getattr(row, c)
                        for c in META_COLUMNS
                        if hasattr(row, c)
                    }
                    rec.update(feats)
                    new_records.append(rec)

                except Exception as exc:
                    errors.append(
                        {
                            "file_id": file_id,
                            "shard_name": shard_name,
                            "audio_member": member_name,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=3),
                        }
                    )

    if new_records:
        new_df = pd.DataFrame(new_records)

        # Store feature payload as float32 to keep the 68k x 790 cache compact.
        for col in feature_columns(new_df):
            new_df[col] = pd.to_numeric(
                new_df[col], errors="coerce"
            ).astype("float32")

        if existing.empty or force:
            combined = new_df
        else:
            combined = pd.concat(
                [existing, new_df],
                ignore_index=True,
            )
            combined = combined.drop_duplicates(
                "file_id", keep="last"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(
            cache_path,
            index=False,
            compression="zstd",
        )
        existing = combined

    selected_ids = set(rows["file_id"].astype(str))
    selected = existing[
        existing["file_id"].astype(str).isin(selected_ids)
    ].copy()

    return selected, errors


def assemble_parquet(
    cache_files: list[Path],
    selected_ids: set[str],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    written = 0
    try:
        for cache_path in cache_files:
            df = pd.read_parquet(cache_path)
            df = df[
                df["file_id"].astype(str).isin(selected_ids)
            ].copy()
            if df.empty:
                continue

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                )
            else:
                if table.schema != writer.schema:
                    table = table.cast(writer.schema)

            writer.write_table(table)
            written += len(df)
    finally:
        if writer is not None:
            writer.close()

    return written


def main() -> None:
    args = parse_args()
    split_path = Path(args.splits)
    dataset_root = Path(args.dataset_root)
    config_path = Path(args.config)
    output_path = Path(args.output)

    split_df = pd.read_csv(split_path, low_memory=False)

    required = {
        "file_id",
        "generator",
        "split",
        "shard_name",
        "audio_member",
    }
    missing = sorted(required - set(split_df.columns))
    if missing:
        raise SystemExit(
            f"Split CSV missing required columns: {missing}"
        )

    if split_df["file_id"].duplicated().any():
        raise SystemExit("Duplicate file_id in split CSV.")

    if args.limit is not None:
        split_df = split_df.head(int(args.limit)).copy()

    settings = load_settings(config_path)
    cfg = build_feature_config(settings)
    signature, signature_payload = feature_signature(
        settings, cfg
    )

    cache_dir = Path(args.cache_root) / signature
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Feature definition: {FEATURE_DEFINITION_VERSION}")
    print(f"Feature signature:  {signature}")
    print(f"Selected rows:      {len(split_df)}")
    print(f"Cache directory:    {cache_dir}")

    all_errors: list[dict] = []
    cache_files: list[Path] = []

    grouped = list(split_df.groupby("shard_name", sort=True))
    for shard_name, rows in tqdm(
        grouped,
        desc="RuASD shards",
        unit="shard",
    ):
        cache_path = cache_dir / f"{Path(str(shard_name)).stem}.parquet"
        selected, errors = process_one_shard(
            str(shard_name),
            rows,
            dataset_root,
            cache_path,
            cfg,
            settings,
            args.force,
        )
        all_errors.extend(errors)
        if cache_path.exists():
            cache_files.append(cache_path)

    selected_ids = set(split_df["file_id"].astype(str))
    written = assemble_parquet(
        cache_files,
        selected_ids,
        output_path,
    )

    errors_path = output_path.with_suffix(".errors.csv")
    if all_errors:
        pd.DataFrame(all_errors).to_csv(
            errors_path,
            index=False,
            encoding="utf-8-sig",
        )
    elif errors_path.exists():
        errors_path.unlink()

    # Lightweight integrity inspection without loading the full final table.
    pf = pq.ParquetFile(output_path)
    schema_names = pf.schema_arrow.names
    feat_cols = [
        c for c in schema_names
        if c.startswith("spectral_") or c.startswith("prosody_")
    ]

    meta = {
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "feature_signature": signature,
        "signature_payload": signature_payload,
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selected_rows": int(len(split_df)),
        "written_rows": int(written),
        "feature_columns": int(len(feat_cols)),
        "spectral_columns": int(
            sum(c.startswith("spectral_") for c in feat_cols)
        ),
        "prosody_columns": int(
            sum(c.startswith("prosody_") for c in feat_cols)
        ),
        "errors": int(len(all_errors)),
        "cache_dir": str(cache_dir),
    }

    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nExtraction summary:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    expected_features = 790 if cfg.include_duration else 789
    problems = []
    if written != len(split_df):
        problems.append(
            f"written_rows={written}, expected={len(split_df)}"
        )
    if len(feat_cols) != expected_features:
        problems.append(
            f"feature_columns={len(feat_cols)}, expected={expected_features}"
        )
    if all_errors:
        problems.append(f"errors={len(all_errors)}")

    if problems:
        raise SystemExit(
            "Feature extraction validation failed: "
            + "; ".join(problems)
        )


if __name__ == "__main__":
    main()
