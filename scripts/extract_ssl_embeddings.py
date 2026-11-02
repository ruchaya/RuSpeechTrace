#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from source_attribution.audio import load_audio,fixed_length
from source_attribution.config import load_config
from source_attribution.ssl_embeddings import SSLConfig,SSLExtractor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--inventory",required=True); ap.add_argument("--splits",default=None); ap.add_argument("--config",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    cfg=load_config(a.config); sc=cfg["ssl"]; s=SSLConfig(sc["model_id"],sc.get("revision"),int(sc.get("sample_rate",16000)),sc.get("pooling","mean"),sc.get("device","auto")); ext=SSLExtractor(s)
    inv=pd.read_csv(a.inventory)
    if a.splits: inv=inv.merge(pd.read_csv(a.splits)[["file_id","split"]],on="file_id",how="inner")
    maxsec=float(sc.get("max_seconds",4)); bs=int(sc.get("batch_size",8)); meta=[c for c in ["file_id","generator","generator_family","checkpoint","speaker","voice","text_hash","raw_or_augmented","augmentation_type","source_dataset","split"] if c in inv.columns]; records=[]
    for start in tqdm(range(0,len(inv),bs),desc="SSL"):
        chunk=inv.iloc[start:start+bs]; aud=[]; rows=[]
        for _,r in chunk.iterrows():
            p=r.get("path")
            if pd.isna(p) or not Path(str(p)).exists(): continue
            y,sr=load_audio(p,s.sample_rate,peak_normalize=True); aud.append(fixed_length(y,sr,maxsec)); rows.append(r)
        if not aud: continue
        E=ext.encode_batch(aud)
        for r,e in zip(rows,E):
            rec={c:r.get(c) for c in meta}; rec.update({f"emb_{i:04d}":float(v) for i,v in enumerate(e)}); records.append(rec)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(records).to_parquet(out,index=False); print(out)
if __name__=="__main__": main()
