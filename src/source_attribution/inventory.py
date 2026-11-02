from __future__ import annotations
from pathlib import Path
from io import BytesIO
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
from .io import find_metadata_tables, read_table, maybe_parse_struct
from .utils import stable_hash_text

ALIASES = {
    "file_id":["file_id","id","utt_id","sample_id","audio_id"],
    "path":["path","file","filepath","file_path","audio_path","wav_path"],
    "label":["label","class","target"],
    "generator":["generator","model","system","source_model","tts_model"],
    "generator_family":["generator_family","family","model_family"],
    "checkpoint":["checkpoint","checkpoint_or_voice","variant"],
    "speaker":["speaker","speaker_id","speakers"],
    "voice":["voice","voice_id"],
    "text":["text","transcription","transcript","sentence"],
    "duration":["duration","duration_sec","seconds"],
    "sample_rate":["sample_rate","sampling_rate","sr"],
    "channels":["channels","num_channels"],
    "raw_or_augmented":["raw_or_augmented","group","subset","source_type"],
    "augmentation_type":["augmentation_type","augmentation","ops"],
    "source_audio":["source_audio","source","parent_audio","original_audio"],
    "source_dataset":["source_dataset","dataset","corpus"],
}

def normalize_text(text):
    if text is None or (isinstance(text,float) and np.isnan(text)): return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]+"," ",text,flags=re.UNICODE)
    return re.sub(r"\s+"," ",text).strip()

def _find_col(df, canonical):
    lower = {str(c).lower(): c for c in df.columns}
    for alias in ALIASES[canonical]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None

def _audio_descriptor(value):
    obj = maybe_parse_struct(value)
    if obj: return obj.get("path"), obj.get("bytes")
    return None, None

def _probe_audio(path=None, audio_bytes=None):
    try:
        if audio_bytes is not None:
            with sf.SoundFile(BytesIO(audio_bytes)) as f:
                return len(f)/f.samplerate, int(f.samplerate), int(f.channels)
        if path and Path(path).exists():
            with sf.SoundFile(path) as f:
                return len(f)/f.samplerate, int(f.samplerate), int(f.channels)
    except Exception:
        pass
    return np.nan, np.nan, np.nan

def canonicalize_metadata(df, dataset_root, source_table):
    root = Path(dataset_root)
    out = pd.DataFrame(index=df.index)
    for canonical in ALIASES:
        col = _find_col(df, canonical)
        out[canonical] = df[col] if col is not None else pd.NA

    audio_col = next((c for c in df.columns if str(c).lower()=="audio"), None)
    paths, bytes_list = [], []
    for idx in df.index:
        p = out.at[idx,"path"]; b = None
        if (p is None or pd.isna(p)) and audio_col is not None:
            p,b = _audio_descriptor(df.at[idx,audio_col])
        if p is not None and not pd.isna(p):
            pp = Path(str(p))
            if not pp.is_absolute(): pp = root/pp
            paths.append(str(pp.resolve()))
        else:
            paths.append(pd.NA)
        bytes_list.append(b)
    out["path"] = paths

    ids = []
    for i,row in out.iterrows():
        fid = row["file_id"]
        if fid is None or pd.isna(fid) or str(fid).strip()=="":
            base = str(row["path"]) if not pd.isna(row["path"]) else f"{source_table}:{i}"
            fid = stable_hash_text(base)[:24]
        ids.append(str(fid))
    out["file_id"] = ids

    durations=[]; srs=[]; channels=[]
    for pos,(_,row) in enumerate(out.iterrows()):
        dur = pd.to_numeric(pd.Series([row["duration"]]),errors="coerce").iloc[0]
        sr = pd.to_numeric(pd.Series([row["sample_rate"]]),errors="coerce").iloc[0]
        ch = pd.to_numeric(pd.Series([row["channels"]]),errors="coerce").iloc[0]
        if pd.isna(dur) or pd.isna(sr) or pd.isna(ch):
            pdur,psr,pch = _probe_audio(None if pd.isna(row["path"]) else row["path"], bytes_list[pos])
            if pd.isna(dur): dur = pdur
            if pd.isna(sr): sr = psr
            if pd.isna(ch): ch = pch
        durations.append(dur); srs.append(sr); channels.append(ch)
    out["duration"]=durations; out["sample_rate"]=srs; out["channels"]=channels
    out["text"] = out["text"].fillna("").astype(str)
    out["normalized_text"] = out["text"].map(normalize_text)
    out["text_hash"] = out["normalized_text"].map(lambda x: stable_hash_text(x) if x else pd.NA)
    out["source_table"] = source_table
    has_gen = out["generator"].notna() & out["generator"].astype(str).str.strip().ne("")
    has_audio = out["path"].notna() | (audio_col is not None)
    out["split_eligibility"] = has_gen & has_audio
    return out

def audit_dataset(dataset_root):
    tables = find_metadata_tables(dataset_root)
    if not tables:
        raise FileNotFoundError(f"No metadata tables found under {dataset_root}")
    frames=[]
    for table in tables:
        try:
            raw = read_table(table)
            if len(raw)==0: continue
            can = canonicalize_metadata(raw,dataset_root,str(table))
            if can["generator"].notna().sum()==0 and can["path"].notna().sum()==0:
                continue
            frames.append(can)
        except Exception as exc:
            warnings.warn(f"Skipping {table}: {exc}")
    if not frames:
        raise RuntimeError("No usable sample-level metadata table could be canonicalized.")
    return pd.concat(frames,ignore_index=True)

def inventory_summary(inv):
    summary = {
        "rows": int(len(inv)),
        "duplicate_file_ids": int(inv["file_id"].duplicated(keep=False).sum()),
        "eligible_rows": int(inv["split_eligibility"].fillna(False).sum()),
        "missing_generator": int(inv["generator"].isna().sum()),
        "missing_path": int(inv["path"].isna().sum()),
        "per_generator": {},
    }
    for gen,g in inv.groupby("generator",dropna=False):
        dur = pd.to_numeric(g["duration"],errors="coerce")
        summary["per_generator"][str(gen)] = {
            "n":int(len(g)),
            "hours":float(dur.sum(skipna=True)/3600),
            "median_duration":float(dur.median()) if dur.notna().any() else None,
            "min_duration":float(dur.min()) if dur.notna().any() else None,
            "max_duration":float(dur.max()) if dur.notna().any() else None,
            "unique_texts":int(g["text_hash"].nunique(dropna=True)),
            "speakers":int(g["speaker"].nunique(dropna=True)),
            "voices":int(g["voice"].nunique(dropna=True)),
        }
    return summary
