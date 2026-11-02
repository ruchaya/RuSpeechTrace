#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize RuASD inventory: semantic labels + WAV-header metadata."
    )
    p.add_argument(
        "--raw-inventory",
        default="data_inventory/ruasd_inventory_raw.csv",
        help="Input CSV produced by the TAR audit.",
    )
    p.add_argument(
        "--dataset-root",
        default="data/raw/RuASD",
        help="Directory containing ruasd-*.tar.",
    )
    p.add_argument(
        "--output",
        default="data_inventory/ruasd_inventory.csv",
        help="Final inventory CSV.",
    )
    p.add_argument(
        "--summary",
        default="data_inventory/inventory_summary.json",
        help="Final summary JSON.",
    )
    p.add_argument(
        "--cache-dir",
        default="data_inventory/ruasd_wav_header_cache",
        help="Per-shard resumable WAV-header cache.",
    )
    p.add_argument(
        "--skip-header-scan",
        action="store_true",
        help="Create semantic inventory without scanning WAV headers.",
    )
    return p.parse_args()


def clean_str_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.mask(out.eq(""))


def normalize_speaker(s: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(s, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    return numeric.astype("Int64")


def resolve_shard_path(row: pd.Series, dataset_root: Path) -> Path:
    raw_path = str(row.get("shard_path", "") or "").strip()
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return p
    return dataset_root / str(row["shard_name"])


def scan_one_shard(
    shard_name: str,
    rows: pd.DataFrame,
    dataset_root: Path,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    shard_path = resolve_shard_path(rows.iloc[0], dataset_root)
    if not shard_path.exists():
        out = pd.DataFrame({
            "file_id": rows["file_id"].astype(str),
            "sample_rate": pd.array([pd.NA] * len(rows), dtype="Int64"),
            "channels": pd.array([pd.NA] * len(rows), dtype="Int64"),
            "audio_frames": pd.array([pd.NA] * len(rows), dtype="Int64"),
            "audio_duration": np.nan,
            "audio_header_error": f"missing shard: {shard_path}",
        })
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
        return out

    records: list[dict[str, Any]] = []
    with tarfile.open(shard_path, "r:*") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}

        for row in rows.itertuples(index=False):
            member_name = str(getattr(row, "audio_member", "") or "")
            rec: dict[str, Any] = {
                "file_id": str(getattr(row, "file_id")),
                "sample_rate": pd.NA,
                "channels": pd.NA,
                "audio_frames": pd.NA,
                "audio_duration": np.nan,
                "audio_header_error": "",
            }

            member = members.get(member_name)
            if member is None:
                rec["audio_header_error"] = f"missing member: {member_name}"
                records.append(rec)
                continue

            try:
                fh = tf.extractfile(member)
                if fh is None:
                    raise RuntimeError("tar.extractfile returned None")
                info = sf.info(fh)
                rec["sample_rate"] = int(info.samplerate)
                rec["channels"] = int(info.channels)
                rec["audio_frames"] = int(info.frames)
                rec["audio_duration"] = float(info.duration)
            except Exception as exc:
                rec["audio_header_error"] = f"{type(exc).__name__}: {exc}"

            records.append(rec)

    out = pd.DataFrame(records)
    out["sample_rate"] = pd.array(out["sample_rate"], dtype="Int64")
    out["channels"] = pd.array(out["channels"], dtype="Int64")
    out["audio_frames"] = pd.array(out["audio_frames"], dtype="Int64")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def build_semantic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    label = clean_str_series(out["label"]).str.lower()
    group = clean_str_series(out["group"]).str.lower()
    subset = clean_str_series(out["subset"])
    model_raw = clean_str_series(out["model_raw"])

    # Preserve original semantics. "subset" has different meaning for fake and real.
    out["generator"] = pd.Series(pd.NA, index=out.index, dtype="string")
    raw_fake = label.eq("fake") & group.eq("raw")
    out.loc[raw_fake, "generator"] = subset[raw_fake]

    out["source_dataset"] = pd.Series(pd.NA, index=out.index, dtype="string")
    raw_real = label.eq("real") & group.eq("raw")
    out.loc[raw_real, "source_dataset"] = subset[raw_real]

    # Raw model field is retained, but is NOT promoted to the system-level target.
    out["checkpoint_raw"] = model_raw
    out["checkpoint"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["voice"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["generator_family"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "speakers" in out.columns:
        out["speaker"] = normalize_speaker(out["speakers"])
    else:
        out["speaker"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    out["raw_or_augmented"] = group
    out["augmentation_type"] = clean_str_series(out["augmentation"])

    # Portable locator instead of an absolute Windows path.
    out["path"] = (
        "tar://"
        + out["shard_name"].astype(str)
        + "::"
        + out["audio_member"].astype(str)
    )

    has_text_hash = clean_str_series(out["text_hash"]).notna()
    out["split_eligibility"] = "not_assigned"
    out.loc[raw_fake & has_text_hash, "split_eligibility"] = (
        "primary_closed_set_candidate"
    )
    out.loc[raw_fake & ~has_text_hash, "split_eligibility"] = (
        "missing_text_requires_decision"
    )
    out.loc[raw_real, "split_eligibility"] = "bona_fide_auxiliary"
    out.loc[group.eq("augmented"), "split_eligibility"] = (
        "blocked_until_provenance_resolved"
    )

    return out


def main() -> None:
    args = parse_args()

    raw_inventory = Path(args.raw_inventory)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    cache_dir = Path(args.cache_dir)

    print(f"Reading: {raw_inventory}")
    df = pd.read_csv(raw_inventory, low_memory=False)

    required = {
        "file_id", "label", "group", "subset", "model_raw", "duration",
        "text_hash", "augmentation", "shard_name", "audio_member"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    if df["file_id"].duplicated().any():
        dup_n = int(df["file_id"].duplicated(keep=False).sum())
        raise SystemExit(f"Duplicate file_id rows detected: {dup_n}")

    df = build_semantic_columns(df)

    duration_meta = pd.to_numeric(df["duration"], errors="coerce")
    df["duration_metadata"] = duration_meta

    if not args.skip_header_scan:
        cache_dir.mkdir(parents=True, exist_ok=True)
        header_parts = []

        groups = list(df.groupby("shard_name", sort=True))
        for shard_name, rows in tqdm(groups, desc="Scanning WAV headers", unit="shard"):
            cache_path = cache_dir / f"{Path(str(shard_name)).stem}.parquet"
            part = scan_one_shard(
                str(shard_name), rows, dataset_root, cache_path
            )
            header_parts.append(part)

        headers = pd.concat(header_parts, ignore_index=True)

        if headers["file_id"].duplicated().any():
            raise SystemExit("Duplicate file_id detected in WAV-header cache.")

        df = df.merge(headers, on="file_id", how="left", validate="one_to_one")

        # Audio header is authoritative when available.
        df["duration"] = pd.to_numeric(df["audio_duration"], errors="coerce")
        fallback = df["duration"].isna() | (df["duration"] <= 0)
        df.loc[fallback, "duration"] = df.loc[fallback, "duration_metadata"]

        df["duration_difference_sec"] = (
            pd.to_numeric(df["audio_duration"], errors="coerce")
            - df["duration_metadata"]
        )
    else:
        df["sample_rate"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["channels"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["audio_frames"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["audio_duration"] = np.nan
        df["audio_header_error"] = ""
        df["duration_difference_sec"] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    raw_fake = (
        df["label"].astype(str).str.lower().eq("fake")
        & df["group"].astype(str).str.lower().eq("raw")
    )
    raw_real = (
        df["label"].astype(str).str.lower().eq("real")
        & df["group"].astype(str).str.lower().eq("raw")
    )
    augmented = df["group"].astype(str).str.lower().eq("augmented")

    generators = (
        df.loc[raw_fake, "generator"]
        .dropna()
        .astype(str)
        .value_counts()
        .sort_index()
    )

    summary = {
        "rows": int(len(df)),
        "duplicate_file_id_rows": int(df["file_id"].duplicated().sum()),
        "raw_fake_rows": int(raw_fake.sum()),
        "raw_real_rows": int(raw_real.sum()),
        "augmented_rows": int(augmented.sum()),
        "raw_fake_generator_count": int(generators.size),
        "raw_fake_generator_counts": {
            k: int(v) for k, v in generators.items()
        },
        "primary_closed_set_candidate_rows": int(
            (df["split_eligibility"] == "primary_closed_set_candidate").sum()
        ),
        "missing_text_requires_decision_rows": int(
            (df["split_eligibility"] == "missing_text_requires_decision").sum()
        ),
        "blocked_augmented_rows": int(
            (df["split_eligibility"] == "blocked_until_provenance_resolved").sum()
        ),
        "wav_header_errors": int(
            clean_str_series(df["audio_header_error"]).notna().sum()
        ),
        "missing_generator_on_raw_fake": int(
            df.loc[raw_fake, "generator"].isna().sum()
        ),
        "sample_rate_distribution": (
            df["sample_rate"].dropna().astype(int).value_counts().sort_index().to_dict()
            if "sample_rate" in df.columns else {}
        ),
        "channels_distribution": (
            df["channels"].dropna().astype(int).value_counts().sort_index().to_dict()
            if "channels" in df.columns else {}
        ),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Inventory: {output_path}")
    print(f"Summary:   {summary_path}")
    print(f"Rows: {summary['rows']}")
    print(f"Raw fake: {summary['raw_fake_rows']}")
    print(f"Raw fake generators: {summary['raw_fake_generator_count']}")
    print(
        "Primary text-safe candidates:",
        summary["primary_closed_set_candidate_rows"],
    )
    print(
        "Missing-text raw fake:",
        summary["missing_text_requires_decision_rows"],
    )
    print("WAV header errors:", summary["wav_header_errors"])


if __name__ == "__main__":
    main()
