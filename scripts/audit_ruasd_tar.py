#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import os
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd

from source_attribution.inventory import normalize_text
from source_attribution.utils import stable_hash_text


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, (list, tuple)):
                text = " ".join(str(x) for x in obj if x is not None).strip()
        except (ValueError, SyntaxError):
            pass
    return text


def _first_text(meta: dict) -> tuple[str, str]:
    # Keep the actual source field so we can later decide which field is preferable.
    for key in ("true_lines", "ground_truth", "transcription", "transcribe"):
        value = _clean(meta.get(key))
        if value:
            return value, key
    return "", ""


def _model_flag(model: str, subset: str, text: str) -> str:
    m = model.strip()
    if not m:
        return "missing"
    if subset and m.casefold() == subset.strip().casefold():
        return "equals_subset"
    nm = normalize_text(m)
    nt = normalize_text(text)
    if nm and nt and len(nm) >= 6 and nm in nt:
        return "matches_text_fragment"
    return "other"


def scan_shard(shard: Path) -> list[dict]:
    rows: list[dict] = []
    with tarfile.open(shard, "r") as tf:
        members = tf.getmembers()
        names = {m.name for m in members}
        json_members = [m for m in members if m.isfile() and m.name.lower().endswith(".json")]
        for member in json_members:
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                meta = json.load(fh)
            except Exception as exc:
                rows.append({
                    "shard_name": shard.name,
                    "shard_path": str(shard.resolve()),
                    "json_member": member.name,
                    "metadata_error": repr(exc),
                })
                continue

            stem = member.name[:-5]
            guessed_wav = stem + ".wav"
            audio_member = guessed_wav if guessed_wav in names else ""
            text, text_source = _first_text(meta)
            normalized_text = normalize_text(text)
            model = _clean(meta.get("model"))
            subset = _clean(meta.get("subset"))
            sample_id = _clean(meta.get("sample_id")) or Path(stem).name

            row = {
                "file_id": sample_id,
                "sample_id": sample_id,
                "label": _clean(meta.get("label")),
                "group": _clean(meta.get("group")),
                "subset": subset,
                "model_raw": model,
                "model_quality_flag": _model_flag(model, subset, text),
                "source_type": _clean(meta.get("source_type")),
                "augmentation": _clean(meta.get("augmentation")),
                "source_audio": _clean(meta.get("source_audio")),
                "metadata_source": _clean(meta.get("metadata_source")),
                "filename": _clean(meta.get("filename")),
                "audio_relpath": _clean(meta.get("audio_relpath")),
                "duration": meta.get("duration"),
                "speakers": meta.get("speakers"),
                "text": text,
                "text_source": text_source,
                "normalized_text": normalized_text,
                "text_hash": stable_hash_text(normalized_text) if normalized_text else pd.NA,
                "shard_name": shard.name,
                "shard_path": str(shard.resolve()),
                "json_member": member.name,
                "audio_member": audio_member,
                "audio_found_in_shard": bool(audio_member),
                "metadata_error": "",
            }
            for key in ("mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred", "cer"):
                row[key] = meta.get(key)
            rows.append(row)
    return rows


def _top(counter: Counter, n: int = 50):
    return [[k, int(v)] for k, v in counter.most_common(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit RuASD WebDataset TAR shards without extracting WAV files.")
    ap.add_argument("--dataset-root", default="data/raw/RuASD")
    ap.add_argument("--output", default="data_inventory/ruasd_inventory_raw.csv")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    shards = sorted(root.glob("ruasd-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No ruasd-*.tar shards found under {root}")

    all_rows: list[dict] = []
    for i, shard in enumerate(shards, 1):
        print(f"[{i}/{len(shards)}] scanning {shard.name}")
        all_rows.extend(scan_shard(shard))

    inv = pd.DataFrame(all_rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out, index=False, encoding="utf-8-sig")

    valid = inv[inv["metadata_error"].fillna("").eq("")].copy()
    raw_fake = valid[(valid["label"] == "fake") & (valid["group"] == "raw")].copy()
    raw_real = valid[(valid["label"] == "real") & (valid["group"] == "raw")].copy()
    aug = valid[valid["group"] == "augmented"].copy()

    # Conservative candidate rows for the main closed-set study. We do not assign the final generator target here.
    inv["closed_set_candidate"] = (
        (inv["label"] == "fake")
        & (inv["group"] == "raw")
        & inv["subset"].fillna("").astype(str).str.len().gt(0)
        & inv["audio_found_in_shard"].fillna(False)
        & inv["text_hash"].notna()
    )
    inv.to_csv(out, index=False, encoding="utf-8-sig")

    raw_relpaths = set(raw_fake["audio_relpath"].fillna("").astype(str)) | set(raw_real["audio_relpath"].fillna("").astype(str))
    raw_filenames = set(raw_fake["filename"].fillna("").astype(str)) | set(raw_real["filename"].fillna("").astype(str))
    source_audio = aug["source_audio"].fillna("").astype(str)
    exact_link = source_audio.isin(raw_relpaths)
    basename_link = source_audio.map(lambda x: os.path.basename(x) if x else "").isin(raw_filenames)

    summary = {
        "shards": int(len(shards)),
        "rows": int(len(inv)),
        "duplicate_file_id_rows": int(inv["file_id"].duplicated(keep=False).sum()),
        "audio_missing": int((~inv["audio_found_in_shard"].fillna(False)).sum()),
        "metadata_errors": int(inv["metadata_error"].fillna("").ne("").sum()),
        "text_missing_all": int(inv["text_hash"].isna().sum()),
        "closed_set_candidate_rows": int(inv["closed_set_candidate"].fillna(False).sum()),
        "labels": _top(Counter(valid["label"].fillna("").astype(str))),
        "groups": _top(Counter(valid["group"].fillna("").astype(str))),
        "label_group": _top(Counter(zip(valid["label"].fillna("").astype(str), valid["group"].fillna("").astype(str)))),
        "raw_fake_subsets": _top(Counter(raw_fake["subset"].fillna("").astype(str))),
        "raw_real_subsets": _top(Counter(raw_real["subset"].fillna("").astype(str))),
        "raw_fake_model_quality": _top(Counter(raw_fake["model_quality_flag"].fillna("").astype(str))),
        "text_sources_all": _top(Counter(valid["text_source"].fillna("").astype(str))),
        "text_sources_raw_fake": _top(Counter(raw_fake["text_source"].fillna("").astype(str))),
        "augmentations": _top(Counter(aug["augmentation"].fillna("").astype(str))),
        "augmented_rows": int(len(aug)),
        "augmented_source_audio_missing": int(source_audio.eq("").sum()),
        "augmented_source_exact_match_in_downloaded_raw": int(exact_link.sum()),
        "augmented_source_basename_match_in_downloaded_raw": int((~exact_link & basename_link).sum()),
    }

    summary_path = out.with_name("ruasd_inventory_raw_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-shard structure is useful because RuASD shards are not random mixtures of all sources.
    shard_rows = []
    for shard_name, df in valid.groupby("shard_name", sort=True):
        rf = df[(df["label"] == "fake") & (df["group"] == "raw")]
        rr = df[(df["label"] == "real") & (df["group"] == "raw")]
        ag = df[df["group"] == "augmented"]
        shard_rows.append({
            "shard_name": shard_name,
            "n": len(df),
            "raw_fake": len(rf),
            "raw_real": len(rr),
            "augmented": len(ag),
            "raw_fake_subsets": json.dumps(Counter(rf["subset"].fillna("").astype(str)), ensure_ascii=False),
            "raw_real_subsets": json.dumps(Counter(rr["subset"].fillna("").astype(str)), ensure_ascii=False),
            "augmentations": json.dumps(Counter(ag["augmentation"].fillna("").astype(str)), ensure_ascii=False),
        })
    pd.DataFrame(shard_rows).to_csv(out.with_name("ruasd_shard_summary.csv"), index=False, encoding="utf-8-sig")

    # Preserve the observed relation without pretending model==generator.
    sm = (
        raw_fake.groupby(["subset", "model_raw", "model_quality_flag"], dropna=False)
        .size().reset_index(name="n").sort_values(["subset", "n"], ascending=[True, False])
    )
    sm.to_csv(out.with_name("ruasd_subset_model_summary.csv"), index=False, encoding="utf-8-sig")

    anomalies = valid[valid["model_quality_flag"].isin(["matches_text_fragment", "other"])].copy()
    anomalies.to_csv(out.with_name("ruasd_model_anomalies.csv"), index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Inventory:     {out}")
    print(f"Summary:       {summary_path}")
    print(f"Per-shard:     {out.with_name('ruasd_shard_summary.csv')}")
    print(f"Subset/model:  {out.with_name('ruasd_subset_model_summary.csv')}")
    print(f"Anomalies:     {out.with_name('ruasd_model_anomalies.csv')}")


if __name__ == "__main__":
    main()
