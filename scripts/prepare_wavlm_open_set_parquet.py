#!/usr/bin/env python
"""Assemble checked 120-chunk WavLM embeddings for existing run_open_set.py."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunks',default='embeddings/wavlm_base_ruasd_4s/chunks')
    p.add_argument('--splits',default='splits/closed_set.csv')
    p.add_argument('--output',default='embeddings/wavlm_base_ruasd_4s/all_68000_for_open_set.parquet')
    a=p.parse_args()
    cols=[f'emb_{i:04d}' for i in range(768)]
    chunk_paths=sorted(Path(a.chunks).glob('*.parquet'))
    if len(chunk_paths)!=120: raise RuntimeError(f'Expected 120 chunks, got {len(chunk_paths)}')
    ids=pd.read_csv(a.splits,usecols=['file_id'],encoding='utf-8-sig')['file_id'].astype(str)
    if len(ids)!=68000 or ids.duplicated().any(): raise RuntimeError('Frozen split integrity failed')
    frames=[]
    for path in chunk_paths:
        frame=pd.read_parquet(path,columns=['file_id']+cols)
        frames.append(frame)
    df=pd.concat(frames,ignore_index=True)
    df['file_id']=df['file_id'].astype(str)
    if len(df)!=68000 or df.file_id.duplicated().any() or set(df.file_id)!=set(ids):
        raise RuntimeError('Missing, extra, or duplicate embeddings')
    if not np.isfinite(df[cols].to_numpy()).all():raise RuntimeError('Nonfinite WavLM embeddings')
    out=Path(a.output)
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)
    print(f'Wrote verified {len(df)} rows x {len(cols)} embedding dimensions to {out}')
if __name__=='__main__':main()
