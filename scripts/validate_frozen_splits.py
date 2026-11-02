#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


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
EXPECTED_GENERATORS = 34


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate and fingerprint frozen RuASD splits.")
    p.add_argument("--closed-set", default="splits/closed_set.csv")
    p.add_argument("--folds", default="splits/open_set_folds.json")
    p.add_argument("--report", default="splits/frozen_split_validation.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    closed_path = Path(args.closed_set)
    folds_path = Path(args.folds)
    report_path = Path(args.report)

    df = pd.read_csv(closed_path, low_memory=False)

    required = {"file_id", "generator", "text_hash", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    problems: list[str] = []

    # Basic size / ID checks
    if len(df) != sum(EXPECTED_SPLIT_COUNTS.values()):
        problems.append(f"Unexpected total rows: {len(df)}")
    if df["file_id"].duplicated().any():
        problems.append(
            f"Duplicate file_id rows: {int(df['file_id'].duplicated(keep=False).sum())}"
        )

    split_counts = df["split"].value_counts().to_dict()
    for split_name, expected in EXPECTED_SPLIT_COUNTS.items():
        got = int(split_counts.get(split_name, 0))
        if got != expected:
            problems.append(f"{split_name}: expected {expected}, got {got}")

    generators = sorted(df["generator"].astype(str).unique())
    if len(generators) != EXPECTED_GENERATORS:
        problems.append(
            f"Expected {EXPECTED_GENERATORS} generators, got {len(generators)}"
        )

    # Exact per-generator quotas
    quota_table = (
        df.groupby(["generator", "split"])
        .size()
        .unstack(fill_value=0)
    )
    for generator in generators:
        for split_name, expected in EXPECTED_PER_GENERATOR.items():
            got = int(quota_table.loc[generator, split_name])
            if got != expected:
                problems.append(
                    f"{generator}/{split_name}: expected {expected}, got {got}"
                )

    # Text leakage checks
    hashes = {}
    for split_name in EXPECTED_SPLIT_COUNTS:
        hashes[split_name] = set(
            df.loc[df["split"].eq(split_name), "text_hash"]
            .dropna()
            .astype(str)
        )

    text_intersections = {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }
    if any(text_intersections.values()):
        problems.append(f"Text-hash leakage: {text_intersections}")

    # Open-set fold checks
    folds_data = json.loads(folds_path.read_text(encoding="utf-8"))
    folds = folds_data["folds"]

    unknown_occurrences = Counter()
    fold_sizes = {}
    for fold_id, fold in folds.items():
        known = set(map(str, fold["known_generators"]))
        unknown = set(map(str, fold["unknown_generators"]))
        if known & unknown:
            problems.append(f"Fold {fold_id}: known/unknown overlap")
        if known | unknown != set(generators):
            problems.append(f"Fold {fold_id}: generator coverage mismatch")
        for g in unknown:
            unknown_occurrences[g] += 1
        fold_sizes[str(fold_id)] = {
            "known": len(known),
            "unknown": len(unknown),
        }

    bad_unknown_counts = {
        g: c for g, c in unknown_occurrences.items() if c != 1
    }
    missing_unknown = sorted(set(generators) - set(unknown_occurrences))
    if bad_unknown_counts:
        problems.append(
            f"Generators unknown != exactly once: {bad_unknown_counts}"
        )
    if missing_unknown:
        problems.append(
            f"Generators never unknown: {missing_unknown}"
        )

    report = {
        "passed": not problems,
        "closed_set_rows": int(len(df)),
        "generator_count": len(generators),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "per_generator_quota": EXPECTED_PER_GENERATOR,
        "unique_text_hashes_by_split": {
            s: len(h) for s, h in hashes.items()
        },
        "text_hash_intersections": text_intersections,
        "open_set_fold_sizes": fold_sizes,
        "unknown_occurrence_counts": dict(sorted(unknown_occurrences.items())),
        "sha256": {
            str(closed_path): sha256_file(closed_path),
            str(folds_path): sha256_file(folds_path),
        },
        "problems": problems,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
