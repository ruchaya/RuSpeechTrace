from __future__ import annotations
from pathlib import Path
import ast
import json
import pandas as pd

SUPPORTED_TABLES = {".csv",".tsv",".json",".jsonl",".parquet",".pq"}

def read_table(path):
    path = Path(path); ext = path.suffix.lower()
    if ext == ".csv": return pd.read_csv(path)
    if ext == ".tsv": return pd.read_csv(path, sep="\t")
    if ext in {".parquet",".pq"}: return pd.read_parquet(path)
    if ext == ".jsonl": return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("data","records","items"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]; break
            return pd.DataFrame(data)
    raise ValueError(f"Unsupported table format: {path}")

def write_table(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv": df.to_csv(path,index=False)
    elif ext == ".tsv": df.to_csv(path,sep="\t",index=False)
    elif ext in {".parquet",".pq"}: df.to_parquet(path,index=False)
    elif ext == ".jsonl": df.to_json(path,orient="records",lines=True,force_ascii=False)
    else: raise ValueError(f"Unsupported output format: {path}")

def find_metadata_tables(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_TABLES)

def maybe_parse_struct(value):
    if isinstance(value, dict): return value
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("{") and v.endswith("}"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(v)
                    if isinstance(obj, dict): return obj
                except Exception:
                    pass
    return None
