#!/usr/bin/env python
import argparse,traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from source_attribution.audio import load_audio
from source_attribution.acoustic_features import extract_acoustic_features,AcousticFeatureConfig

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--inventory",required=True); ap.add_argument("--splits",default=None); ap.add_argument("--output",required=True)
    ap.add_argument("--sample-rate",type=int,default=16000); ap.add_argument("--peak-normalize",action=argparse.BooleanOptionalAction,default=True); ap.add_argument("--exclude-duration",action="store_true"); ap.add_argument("--limit",type=int,default=None); a=ap.parse_args()
    inv=pd.read_csv(a.inventory)
    if a.splits:
        sp=pd.read_csv(a.splits)[["file_id","split"]]; inv=inv.merge(sp,on="file_id",how="inner")
    if a.limit: inv=inv.head(a.limit)
    meta=[c for c in ["file_id","path","generator","generator_family","checkpoint","speaker","voice","text_hash","duration","raw_or_augmented","augmentation_type","source_audio","source_dataset","split"] if c in inv.columns]
    rows=[]; errors=[]; cfg=AcousticFeatureConfig(sr=a.sample_rate,include_duration=not a.exclude_duration)
    for _,r in tqdm(inv.iterrows(),total=len(inv),desc="features"):
        path=r.get("path")
        if pd.isna(path) or not Path(str(path)).exists():
            errors.append({"file_id":r["file_id"],"error":f"Missing audio: {path}"}); continue
        try:
            y,sr=load_audio(path,a.sample_rate,peak_normalize=a.peak_normalize)
            feats=extract_acoustic_features(y,sr,float(r["duration"]) if pd.notna(r.get("duration")) else None,cfg)
            rec={c:r.get(c) for c in meta}; rec.update(feats); rows.append(rec)
        except Exception as exc:
            errors.append({"file_id":r["file_id"],"error":repr(exc),"traceback":traceback.format_exc(limit=2)})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_parquet(out,index=False)
    if errors: pd.DataFrame(errors).to_csv(out.with_suffix(".errors.csv"),index=False)
    print(f"features={len(rows)} errors={len(errors)} output={out}")
if __name__=="__main__": main()
