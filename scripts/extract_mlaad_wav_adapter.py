import argparse,json,sys
from pathlib import Path
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from scripts.extract_acoustic_features import load_settings,build_feature_config,feature_signature,sha256_file
from source_attribution.audio import preprocess_audio
from source_attribution.acoustic_features import extract_acoustic_features

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=None)
    a=ap.parse_args()
    split=Path('splits/mlaad_ru_three_family_split.csv')
    root=Path(os.environ.get("MLAAD_ROOT", "data/external/MLAAD_fake_ru"))
    config=Path('configs/closed_set_acoustic.yaml')
    settings=load_settings(config)
    cfg=build_feature_config(settings)
    sig,_=feature_signature(settings,cfg)
    ref=json.loads(Path('features/ruasd_closed_set_acoustic.meta.json').read_text(encoding='utf-8'))
    assert sig==ref['feature_signature'],'FEATURE SIGNATURE MISMATCH'
    names=[c for c in pq.read_schema('features/ruasd_closed_set_acoustic.parquet').names if c.startswith(('spectral_','prosody_'))]
    assert len(names)==790
    d=pd.read_csv(split,encoding='utf-8-sig',keep_default_na=False).sort_values('path').reset_index(drop=True)
    assert len(d)==3000 and d['path'].is_unique
    if a.limit is not None:
        assert a.limit>0
        d=d.head(a.limit).copy()
    out=Path('features/mlaad_three_family_acoustic.parquet' if a.limit is None else 'features/mlaad_smoke_'+str(a.limit)+'.parquet')
    assert not out.exists(), 'OUTPUT ALREADY EXISTS: '+str(out)
    cache=Path('features/cache/mlaad_wav')/sig/sha256_file(split)[:16]/('full' if a.limit is None else 'smoke_'+str(a.limit))
    cache.mkdir(parents=True,exist_ok=True)
    chunks=[]
    computed=0
    reused=0
    for i in tqdm(range(0,len(d),20),desc='MLAAD batches'):
        b=d.iloc[i:i+20]
        cp=cache/('batch_'+str(i).zfill(6)+'.parquet')
        valid=False
        if cp.exists():
            old=pd.read_parquet(cp)
            valid='path' in old.columns and old['path'].tolist()==b['path'].tolist() and all(n in old.columns for n in names)
        if valid:
            reused+=len(b)
        else:
            records=[]
            for row in b.to_dict('records'):
                wav=root/str(row['path'])
                if not wav.is_file():
                    raise FileNotFoundError(wav)
                y,sr,duration=preprocess_audio(str(wav),target_sr=cfg.sr,mono=True,peak_normalize=bool(settings['peak_normalize']),max_duration_sec=float(settings['max_duration_sec']),min_audio_sec=float(settings['min_audio_sec']))
                f=extract_acoustic_features(y,sr,original_duration=duration,cfg=cfg)
                if list(f)!=names:
                    raise RuntimeError('FEATURE COLUMN MISMATCH: '+str(wav))
                row.update(f)
                records.append(row)
                computed+=1
            part=pd.DataFrame(records)
            for n in names:
                part[n]=part[n].astype('float32')
            tmp=cp.with_suffix('.tmp.parquet')
            part.to_parquet(tmp,index=False,compression='zstd')
            tmp.replace(cp)
        chunks.append(cp)
    result=pd.concat([pd.read_parquet(cp) for cp in chunks],ignore_index=True)
    assert len(result)==len(d) and result['path'].tolist()==d['path'].tolist()
    assert [c for c in result.columns if c.startswith(('spectral_','prosody_'))]==names
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp.parquet')
    result.to_parquet(tmp,index=False,compression='zstd')
    tmp.replace(out)
    meta={'feature_signature':sig,'split_sha256':sha256_file(split),'config_sha256':sha256_file(config),'rows':len(result),'feature_columns':len(names),'newly_computed':computed,'cache_reused':reused}
    out.with_suffix('.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ROWS:',len(result))
    print('FEATURE COUNT:',len(names))
    print('NEWLY COMPUTED:',computed)
    print('CACHE REUSED:',reused)
    print('OUTPUT:',out)
    print('SHA256:',sha256_file(out))
    print('EXTRACTION PASSED')

if __name__=='__main__':
    main()
