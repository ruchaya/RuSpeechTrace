from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .leakage import leakage_report


PRIMARY_ELIGIBILITY = "primary_closed_set_candidate"


def is_raw_mask(df: pd.DataFrame) -> pd.Series:
    """Return True for rows that are clean/raw rather than augmented/distorted."""
    vals = (
        df.get("raw_or_augmented", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    aug = (
        df.get("augmentation_type", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    return ~(
        vals.str.contains("aug", na=False)
        | vals.str.contains("distort", na=False)
        | ~aug.isin(["", "none", "raw", "clean", "nan"])
    )


def _eligibility_mask(df: pd.DataFrame) -> pd.Series:
    """Support both the old boolean flag and the new string-valued eligibility."""
    if "split_eligibility" not in df.columns:
        return pd.Series(True, index=df.index)

    s = df["split_eligibility"]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    normalized = s.astype("string").str.strip().str.lower()
    return normalized.eq(PRIMARY_ELIGIBILITY)


def _nonempty_text_hash_mask(df: pd.DataFrame) -> pd.Series:
    s = df["text_hash"].astype("string")
    return s.notna() & s.str.strip().ne("")


def _stable_score(file_id: str, seed: int, salt: str = "") -> str:
    payload = f"{seed}|{salt}|{file_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_primary_pool(
    inv: pd.DataFrame,
    primary_generators: Iterable[str],
    *,
    raw_only: bool = True,
) -> pd.DataFrame:
    """
    Select the strict clean/raw synthetic pool before train/val/test assignment.

    Important: no 2000-per-generator sampling is performed here. We first assign
    text_hash groups globally to splits, then draw exact per-split quotas.
    """
    required = {"file_id", "generator", "text_hash"}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = inv.copy()
    generators = set(map(str, primary_generators))

    if "label" in df.columns:
        df = df[df["label"].astype(str).str.lower().eq("fake")]
    if "group" in df.columns:
        df = df[df["group"].astype(str).str.lower().eq("raw")]
    if raw_only:
        df = df[is_raw_mask(df)]

    df = df[_eligibility_mask(df)]
    df = df[df["generator"].notna()].copy()
    df = df[df["generator"].astype(str).isin(generators)]
    df = df[_nonempty_text_hash_mask(df)].copy()

    present = set(df["generator"].astype(str).unique())
    missing_generators = sorted(generators - present)
    if missing_generators:
        raise RuntimeError(
            f"Primary generators absent from strict pool: {missing_generators}"
        )

    if df["file_id"].duplicated().any():
        n = int(df["file_id"].duplicated(keep=False).sum())
        raise RuntimeError(f"Duplicate file_id rows in primary pool: {n}")

    return df.reset_index(drop=True)


def one_row_per_generator_text(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Keep one deterministic recording for each (generator, text_hash) pair.

    This avoids repeated utterances of the same text within one generator from
    dominating the group splitter while preserving cross-generator text groups.
    """
    work = df.copy()
    work["_stable_score"] = [
        _stable_score(str(fid), seed, "dedupe") for fid in work["file_id"]
    ]
    work = work.sort_values(
        ["generator", "text_hash", "_stable_score", "file_id"],
        kind="mergesort",
    )
    work = work.drop_duplicates(["generator", "text_hash"], keep="first")
    return work.drop(columns="_stable_score").reset_index(drop=True)


def _groups(df: pd.DataFrame) -> pd.Series:
    group = df["text_hash"].astype("string")
    empty = group.isna() | (group.str.len() == 0)

    source = df.get(
        "source_audio", pd.Series(pd.NA, index=df.index)
    ).astype("string")
    source_nonempty = source.notna() & source.str.strip().ne("")
    fallback = source.where(
        source_nonempty,
        df["file_id"].astype("string"),
    )
    return group.where(~empty, fallback).astype(str)


def make_global_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Globally assign text groups to train/validation/test.

    The two nested 5-fold group splits create an approximately 64/16/20
    candidate pool. Exact 65/15/20 final counts are imposed afterwards by
    select_split_quotas().
    """
    work = df.copy().reset_index(drop=True)
    if len(work) == 0:
        raise RuntimeError("Cannot split an empty dataframe.")

    y = work["generator"].astype(str).to_numpy()
    groups = _groups(work).to_numpy()

    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(sg.split(work, y=y, groups=groups))

    tv = work.iloc[tv_idx].copy()
    sg2 = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed + 1
    )
    tr_rel, val_rel = next(
        sg2.split(
            tv,
            y=tv["generator"].astype(str),
            groups=_groups(tv),
        )
    )

    split = np.empty(len(work), dtype=object)
    split[tv_idx[tr_rel]] = "train"
    split[tv_idx[val_rel]] = "validation"
    split[test_idx] = "test"
    work["split"] = split

    rep = leakage_report(work)
    if not rep["passed"]:
        raise RuntimeError(rep)

    return work


def select_split_quotas(
    split_df: pd.DataFrame,
    quotas: Mapping[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw an exact deterministic quota for every generator inside each split.

    Because the global text-group split has already been assigned, this cannot
    introduce cross-split text leakage.
    """
    required_splits = ("train", "validation", "test")
    missing_quota = [s for s in required_splits if s not in quotas]
    if missing_quota:
        raise ValueError(f"Missing quotas for: {missing_quota}")

    generators = sorted(split_df["generator"].dropna().astype(str).unique())
    parts: list[pd.DataFrame] = []
    shortages: list[dict[str, object]] = []

    for split_name in required_splits:
        quota = int(quotas[split_name])
        part = split_df[split_df["split"].eq(split_name)]

        for generator in generators:
            g = part[part["generator"].astype(str).eq(generator)].copy()
            available = len(g)
            if available < quota:
                shortages.append(
                    {
                        "split": split_name,
                        "generator": generator,
                        "available": available,
                        "required": quota,
                    }
                )
                continue

            g["_stable_score"] = [
                _stable_score(str(fid), seed, f"quota:{split_name}")
                for fid in g["file_id"]
            ]
            chosen = (
                g.sort_values(["_stable_score", "file_id"], kind="mergesort")
                .head(quota)
                .drop(columns="_stable_score")
            )
            parts.append(chosen)

    if shortages:
        preview = shortages[:20]
        raise RuntimeError(
            "Per-split quota cannot be satisfied. "
            f"First shortages: {preview}"
        )

    out = pd.concat(parts, ignore_index=True)

    # Strong acceptance checks.
    rep = leakage_report(out)
    if not rep["passed"]:
        raise RuntimeError(rep)

    expected_total = len(generators) * sum(int(v) for v in quotas.values())
    if len(out) != expected_total:
        raise RuntimeError(
            f"Unexpected final size: {len(out)} != {expected_total}"
        )

    counts = (
        out.groupby(["split", "generator"])
        .size()
        .unstack(fill_value=0)
    )
    for split_name, quota in quotas.items():
        bad = counts.loc[split_name][counts.loc[split_name] != int(quota)]
        if len(bad):
            raise RuntimeError(
                f"Quota mismatch in {split_name}: {bad.to_dict()}"
            )

    return out.sort_values(
        ["split", "generator", "file_id"], kind="mergesort"
    ).reset_index(drop=True)


def select_balanced_subset(
    inv: pd.DataFrame,
    max_per_generator: int = 2000,
    min_per_generator: int = 1,
    raw_only: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Backward-compatible helper for older code.

    New strict RuASD experiments should use select_primary_pool() ->
    one_row_per_generator_text() -> make_global_splits() ->
    select_split_quotas().
    """
    df = inv.copy()
    if raw_only:
        df = df[is_raw_mask(df)]
    if "split_eligibility" in df.columns:
        df = df[_eligibility_mask(df)]
    df = df[df["generator"].notna()].copy()

    counts = df["generator"].value_counts()
    df = df[df["generator"].isin(counts[counts >= min_per_generator].index)]

    parts = []
    for generator, g in df.groupby("generator", sort=True):
        g = g.copy()
        g["_stable_score"] = [
            _stable_score(str(fid), seed, f"legacy:{generator}")
            for fid in g["file_id"]
        ]
        parts.append(
            g.sort_values(["_stable_score", "file_id"])
            .head(min(max_per_generator, len(g)))
            .drop(columns="_stable_score")
        )

    if not parts:
        raise RuntimeError("No generator satisfies selection criteria.")

    return pd.concat(parts, ignore_index=True)


def make_open_set_folds(
    split_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    generators = sorted(
        split_df["generator"].dropna().astype(str).unique()
    )
    rng = np.random.default_rng(seed)
    arr = np.array(generators, dtype=object)
    rng.shuffle(arr)

    chunks = np.array_split(arr, n_folds)
    allg = set(generators)
    folds = {}

    for i, ch in enumerate(chunks, 1):
        unknown = sorted(map(str, ch.tolist()))
        folds[str(i)] = {
            "known_generators": sorted(allg - set(unknown)),
            "unknown_generators": unknown,
            "rule": (
                "Unknown generators are absent from training, validation, "
                "and threshold tuning for this fold."
            ),
        }

    return {
        "n_folds": n_folds,
        "seed": seed,
        "folds": folds,
    }
