#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPECTED_ROWS = 68000
EXPECTED_GENERATORS = 34
EXPECTED_FEATURES = 789
EXPECTED_SPECTRAL = 777
EXPECTED_PROSODY = 12
EXPECTED_SPLIT_COUNTS = {
    "train": 44200,
    "validation": 10200,
    "test": 13600,
}
EXPECTED_PER_GENERATOR = {
    "train": 1300,
    "validation": 300,
    "test": 400,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Acceptance checks for the frozen 68k RuASD acoustic feature store."
    )
    p.add_argument(
        "--features",
        default="features/ruasd_closed_set_acoustic_fixed4.parquet",
    )
    p.add_argument(
        "--splits",
        default="splits/closed_set.csv",
    )
    p.add_argument(
        "--meta",
        default="features/ruasd_closed_set_acoustic_fixed4.meta.json",
    )
    p.add_argument(
        "--report",
        default="features/ruasd_closed_set_acoustic_fixed4_validation.json",
    )
    p.add_argument(
        "--feature-qc",
        default="features/ruasd_closed_set_acoustic_fixed4_feature_qc.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    feature_path = Path(args.features)
    split_path = Path(args.splits)
    meta_path = Path(args.meta)
    report_path = Path(args.report)
    qc_path = Path(args.feature_qc)

    problems: list[str] = []

    pf = pq.ParquetFile(feature_path)
    parquet_rows = int(pf.metadata.num_rows)
    schema_names = pf.schema_arrow.names

    spectral_cols = [c for c in schema_names if c.startswith("spectral_")]
    prosody_cols = [c for c in schema_names if c.startswith("prosody_")]
    feature_cols = spectral_cols + prosody_cols
    if "prosody_duration_sec" in feature_cols:
        problems.append(
            "Fixed-4s feature store unexpectedly contains prosody_duration_sec"
        )
    if parquet_rows != EXPECTED_ROWS:
        problems.append(f"Expected {EXPECTED_ROWS} parquet rows, got {parquet_rows}")
    if len(feature_cols) != EXPECTED_FEATURES:
        problems.append(
            f"Expected {EXPECTED_FEATURES} feature columns, got {len(feature_cols)}"
        )
    if len(spectral_cols) != EXPECTED_SPECTRAL:
        problems.append(
            f"Expected {EXPECTED_SPECTRAL} spectral columns, got {len(spectral_cols)}"
        )
    if len(prosody_cols) != EXPECTED_PROSODY:
        problems.append(
            f"Expected {EXPECTED_PROSODY} prosody columns, got {len(prosody_cols)}"
        )

    needed_cols = ["file_id", "generator", "split", "text_hash"] + feature_cols
    missing_schema = [c for c in needed_cols if c not in schema_names]
    if missing_schema:
        problems.append(f"Missing parquet columns: {missing_schema[:20]}")

    feat = pd.read_parquet(feature_path, columns=needed_cols)
    split = pd.read_csv(
        split_path,
        low_memory=False,
        usecols=["file_id", "generator", "split", "text_hash"],
    )

    if feat["file_id"].duplicated().any():
        problems.append(
            f"Duplicate feature file_id rows: "
            f"{int(feat['file_id'].duplicated(keep=False).sum())}"
        )

    feature_ids = set(feat["file_id"].astype(str))
    split_ids = set(split["file_id"].astype(str))
    missing_features = sorted(split_ids - feature_ids)
    extra_features = sorted(feature_ids - split_ids)

    if missing_features:
        problems.append(f"Split rows missing features: {len(missing_features)}")
    if extra_features:
        problems.append(f"Feature rows absent from split: {len(extra_features)}")

    left = feat[["file_id", "generator", "split", "text_hash"]].copy()
    right = split.copy()
    for col in ["file_id", "generator", "split", "text_hash"]:
        left[col] = left[col].astype("string")
        right[col] = right[col].astype("string")

    merged = left.merge(
        right,
        on="file_id",
        how="outer",
        suffixes=("_feat", "_split"),
        indicator=True,
        validate="one_to_one",
    )

    metadata_mismatch = 0
    both = merged["_merge"].eq("both")
    for col in ["generator", "split", "text_hash"]:
        mismatch = both & (
            merged[f"{col}_feat"].fillna("<NA>")
            != merged[f"{col}_split"].fillna("<NA>")
        )
        metadata_mismatch += int(mismatch.sum())

    if metadata_mismatch:
        problems.append(f"Feature/split metadata mismatches: {metadata_mismatch}")

    split_counts = feat["split"].value_counts().to_dict()
    for split_name, expected in EXPECTED_SPLIT_COUNTS.items():
        got = int(split_counts.get(split_name, 0))
        if got != expected:
            problems.append(f"{split_name}: expected {expected}, got {got}")

    generators = sorted(feat["generator"].astype(str).unique())
    if len(generators) != EXPECTED_GENERATORS:
        problems.append(
            f"Expected {EXPECTED_GENERATORS} generators, got {len(generators)}"
        )

    quota = feat.groupby(["generator", "split"]).size().unstack(fill_value=0)
    for generator in generators:
        for split_name, expected in EXPECTED_PER_GENERATOR.items():
            got = int(quota.loc[generator, split_name])
            if got != expected:
                problems.append(
                    f"{generator}/{split_name}: expected {expected}, got {got}"
                )

    X = feat[feature_cols]
    arr = X.to_numpy(dtype=np.float64, copy=False)

    inf_total = int(np.isinf(arr).sum())
    if inf_total:
        problems.append(f"Infinite feature values: {inf_total}")

    qc_rows = []
    all_nan_cols = []
    constant_cols = []

    for col in feature_cols:
        s = pd.to_numeric(X[col], errors="coerce")
        finite_mask = np.isfinite(s.to_numpy(dtype=np.float64, copy=False))
        finite = s[finite_mask]
        n_missing = int(s.isna().sum())
        missing_pct = 100.0 * n_missing / len(s)

        if n_missing == len(s):
            all_nan_cols.append(col)

        nunique_finite = int(finite.nunique(dropna=True))
        if len(finite) > 0 and nunique_finite <= 1:
            constant_cols.append(col)

        qc_rows.append(
            {
                "feature": col,
                "group": (
                    "spectral" if col.startswith("spectral_") else "prosody"
                ),
                "dtype": str(s.dtype),
                "n_missing": n_missing,
                "missing_pct": round(missing_pct, 6),
                "n_finite": int(len(finite)),
                "n_unique_finite": nunique_finite,
                "min_finite": (
                    float(finite.min()) if len(finite) else np.nan
                ),
                "max_finite": (
                    float(finite.max()) if len(finite) else np.nan
                ),
            }
        )

    if all_nan_cols:
        problems.append(f"All-NaN feature columns: {all_nan_cols}")

    qc = pd.DataFrame(qc_rows).sort_values(
        ["missing_pct", "feature"],
        ascending=[False, True],
    )
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("written_rows", -1)) != EXPECTED_ROWS:
            problems.append(
                f"Meta written_rows={meta.get('written_rows')} != {EXPECTED_ROWS}"
            )
        if int(meta.get("feature_columns", -1)) != EXPECTED_FEATURES:
            problems.append(
                f"Meta feature_columns={meta.get('feature_columns')} != {EXPECTED_FEATURES}"
            )
        expected_split_sha = meta.get("split_sha256")
        actual_split_sha = sha256_file(split_path)
        if expected_split_sha and expected_split_sha != actual_split_sha:
            problems.append("Split SHA-256 no longer matches extraction metadata.")
    else:
        problems.append(f"Missing meta JSON: {meta_path}")

    report = {
        "passed": not problems,
        "rows": int(len(feat)),
        "unique_file_ids": int(feat["file_id"].nunique()),
        "generator_count": len(generators),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "feature_columns": len(feature_cols),
        "spectral_columns": len(spectral_cols),
        "prosody_columns": len(prosody_cols),
        "infinite_values": inf_total,
        "all_nan_feature_columns": all_nan_cols,
        "constant_feature_columns": constant_cols,
        "features_with_missing_values": int((qc["n_missing"] > 0).sum()),
        "max_missing_pct": float(qc["missing_pct"].max()) if len(qc) else 0.0,
        "top_missing_features": (
            qc.loc[qc["n_missing"] > 0, ["feature", "n_missing", "missing_pct"]]
            .head(20)
            .to_dict(orient="records")
        ),
        "feature_split_metadata_mismatches": metadata_mismatch,
        "missing_feature_rows": len(missing_features),
        "extra_feature_rows": len(extra_features),
        "sha256": {
            str(feature_path): sha256_file(feature_path),
            str(split_path): sha256_file(split_path),
            str(meta_path): sha256_file(meta_path) if meta_path.exists() else None,
        },
        "problems": problems,
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nFeature QC: {qc_path}")
    print(f"Validation: {report_path}")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
