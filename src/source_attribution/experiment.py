from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from .config import save_config
from .utils import environment_info,git_info,json_dump

META_COLUMNS={"file_id","path","generator","generator_family","checkpoint","speaker","voice","text","normalized_text","text_hash","duration","sample_rate","channels","raw_or_augmented","augmentation_type","source_audio","source_dataset","split_eligibility","source_table","split"}

def feature_columns(df,include_prefixes=None,exclude_columns=None,exclude_duration=False):
    cols=[]
    for c in df.columns:
        if c in META_COLUMNS: continue
        if include_prefixes and not any(str(c).startswith(p) for p in include_prefixes): continue
        if exclude_columns and c in exclude_columns: continue
        if exclude_duration and c=="prosody_duration_sec": continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    if not cols: raise RuntimeError("No numeric feature columns selected.")
    return cols

def create_run_dir(output_root,experiment_name,seed):
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); p=Path(output_root)/experiment_name/f"{stamp}_seed{seed}"
    p.mkdir(parents=True,exist_ok=False); return p

def record_run_context(run_dir,cfg):
    save_config(cfg,Path(run_dir)/"config_resolved.yaml"); json_dump(environment_info(),Path(run_dir)/"environment.json"); json_dump(git_info(),Path(run_dir)/"git_info.json")
