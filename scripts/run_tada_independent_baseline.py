"""TADA (Stan et al., Interspeech 2025) on project frozen RuASD/MLAAD.

Direct reproduction of the REPRESENTATION and ATTRIBUTION MODEL:
  facebook/w2v-bert-2.0, hidden_states[4], temporal mean, unscaled
  Euclidean k-NN with uniform votes, k=21.
The original TADA OOD threshold (EER using held-out OOD validation models)
CANNOT be used on our original frozen folds: generator-level unknown labels
are unavailable for validation. We instead use the 95th percentile of
KNOWN-ONLY validation mean-kNN-distances, an explicitly adapted TADA score.

No production test data are inspected during hyperparameter selection.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd

RU_SPLIT=Path('splits/closed_set.csv')
ML_SPLIT=Path('splits/mlaad_ru_23class_split_v1.csv')
RU_FOLDS=Path('splits/open_set_folds.json')
ML_FOLDS=Path('splits/mlaad_23class_open_set_folds_v1.json')
RU_FROZEN_SHA='f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0'
MODEL='facebook/w2v-bert-2.0'
LAYER=4
K=21
DEFAULT_ROOT=Path('results/raw/tada_w2vbert2_layer4_knn21')
DEFAULT_MLAAD_ROOT=os.environ.get("MLAAD_ROOT", "data/external/MLAAD_fake_ru")


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def digest_ids(seq):
    return hashlib.sha256('\n'.join(map(str,seq)).encode('utf-8')).hexdigest()


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    os.replace(tmp,path)


def dataset_paths(name, a):
    if name=='ruasd':
        return RU_SPLIT,RU_FOLDS,Path(a.ruasd_root)
    return ML_SPLIT,ML_FOLDS,Path(a.mlaad_root)


def load_split(name):
    p=RU_SPLIT if name=='ruasd' else ML_SPLIT
    if not p.is_file(): raise FileNotFoundError(f'Missing frozen split: {p}')
    if name=='ruasd':
        sha=sha256_file(p)
        if sha!=RU_FROZEN_SHA: raise RuntimeError(f'FROZEN RuASD SHA mismatch: {sha}')
        d=pd.read_csv(p,usecols=['file_id','generator','split','shard_name','audio_member'],keep_default_na=False)
        d=d.sort_values(['shard_name','file_id'],kind='stable').reset_index(drop=True)
        counts={'train':44200,'validation':10200,'test':13600}
        assert len(d)==68000 and d.generator.nunique()==34
        assert d.groupby('generator').size().eq(2000).all()
        assert set(d['shard_name']) and d['audio_member'].ne('').all()
        key='file_id'
    else:
        sha=sha256_file(p)
        d=pd.read_csv(p,usecols=['path','generator','split','text_key','original_file'],keep_default_na=False,encoding='utf-8-sig')
        d=d.sort_values('path',kind='stable').reset_index(drop=True)
        counts={'train':14950,'validation':3450,'test':4600}
        assert len(d)==23000 and d.generator.nunique()==23
        assert d.groupby('generator').size().eq(1000).all()
        assert d.groupby('text_key')['split'].nunique().max()==1
        assert d.groupby('original_file')['split'].nunique().max()==1
        key='path'
    if d[key].duplicated().any(): raise RuntimeError(f'Duplicate {key}')
    if d['split'].value_counts().to_dict()!=counts: raise RuntimeError('Unexpected split sizes')
    if d['generator'].eq('').any(): raise RuntimeError('Empty generator')
    return d,key,sha


def load_folds(name, labels):
    f=RU_FOLDS if name=='ruasd' else ML_FOLDS
    if not f.exists():raise FileNotFoundError(f)
    obj=json.loads(f.read_text(encoding='utf-8'))
    expected=set(labels);seen=[]
    if int(obj.get('n_folds',-1))!=5:raise RuntimeError('Expected five fixed folds')
    if name=='mlaad' and obj.get('source_split_sha256')!=sha256_file(ML_SPLIT):
        raise RuntimeError('MLAAD fold/split SHA mismatch')
    for ix in range(1,6):
        fobj=obj['folds'][str(ix)]
        known=set(fobj['known_generators']);unk=set(fobj['unknown_generators'])
        if known & unk or known | unk!=expected:raise RuntimeError(f'Invalid fold {ix}')
        seen.extend(unk)
    if set(seen)!=expected or len(seen)!=len(expected):raise RuntimeError('Each generator must be unknown exactly once')
    return obj['folds'],sha256_file(f)


def find_mlaad_path(root, rel):
    s=str(rel).replace('\\','/')
    while s.startswith('./'):
        s=s[2:]
    if '..' in Path(s).parts:raise ValueError(f'Path traversal: {rel}')
    return root/Path(s)


def validate_audio_sources(name, d, root):
    if name=='ruasd':
        shards=d['shard_name'].unique()
        missing=[str(root/x) for x in shards if not (root/x).is_file()]
        if missing:raise FileNotFoundError(f'Missing {len(missing)} RuASD TAR shards, example: {missing[0]}')
        print(f'RUASD TAR SHARDS VERIFIED: {len(shards)}')
    else:
        if not root.is_dir():raise FileNotFoundError(f'MLAAD WAV root absent: {root}')
        # Test representative files across all classes, not all 23k paths at preflight.
        reps=d.groupby('generator',sort=False).head(1)
        missing=[str(find_mlaad_path(root,p)) for p in reps['path'] if not find_mlaad_path(root,p).is_file()]
        if missing:raise FileNotFoundError(f'Missing MLAAD WAVs, e.g. {missing[0]}')
        print('MLAAD REPRESENTATIVE WAVS VERIFIED:',len(reps))


def manifest_spec(name,split_sha,args):
    return {'protocol':'TADA_Stan2025_w2vbert2_hidden_states_4_mean_kNN21_frozen_splits_v1',
        'dataset':name,'model_id':MODEL,'layer_index':LAYER,'pooling':'time_mean_unmasked',
        'normalization':'librosa.load(sr=16000,mono=True), no DC removal, no peak normalization',
        'crop':'first_n_seconds','max_seconds':float(args.max_seconds),'sample_rate':16000,
        'truncated_encoder_layers':4,'batch_size':1,'knn_metric':'euclidean',
        'knn_weights':'uniform','knn_neighbors':K,'split_sha256':split_sha}


def read_waveform(path_or_buffer,max_seconds):
    import librosa
    # Equivalent to author's librosa.load(...,sr=16000) with optional first-N-seconds cap.
    y,sr=librosa.load(path_or_buffer,sr=16000,mono=True,duration=max_seconds)
    if sr!=16000 or y.size==0 or not np.isfinite(y).all():
        raise RuntimeError('Invalid/empty input waveform')
    return y.astype(np.float32,copy=False)


class W2vBertEarlyLayer:
    def __init__(self,device):
        import torch
        import transformers
        from transformers import AutoFeatureExtractor,Wav2Vec2BertModel
        self.torch=torch
        self.device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        print(f'LOADING EXACT TADA SSL CHECKPOINT: {MODEL}; device={self.device}',flush=True)
        self.processor=AutoFeatureExtractor.from_pretrained(MODEL)
        # Loading model on CPU first avoids loading all 600M weights to an 8GB GPU.
        self.model=Wav2Vec2BertModel.from_pretrained(MODEL)
        layers=self.model.encoder.layers
        if len(layers)<=LAYER:raise RuntimeError('Unexpected w2v-BERT encoder structure')
        self.original_layers=len(layers)
        # Early exit after 4th encoder layer. Evaluate at hidden_states[4] exactly as in
        # author's extractor. No parameters after layer 4 participate in this embedding.
        self.model.encoder.layers=torch.nn.ModuleList(list(layers)[:LAYER])
        self.model.eval().to(self.device)
        self.model_revision=getattr(self.model.config,'_commit_hash',None)
        self.transformers_version=transformers.__version__
        self.torch_version=torch.__version__
        print(f'EARLY-EXIT: {self.original_layers} -> {LAYER} layers; HF={self.transformers_version}; revision={self.model_revision}',flush=True)

    def __call__(self,y):
        torch=self.torch
        inputs=self.processor(y,sampling_rate=16000,return_tensors='pt',padding=True)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode():
            h=self.model(**inputs,output_hidden_states=True,return_dict=True).hidden_states
            if h is None or len(h)!=LAYER+1:raise RuntimeError(f'Expected {LAYER+1} hidden states, got {len(h) if h else None}')
            result=h[LAYER].mean(dim=1).squeeze(0).float().cpu().numpy()
        if result.ndim!=1 or result.size!=1024 or not np.isfinite(result).all():raise RuntimeError('Bad SSL embedding: expected 1024 finite values')
        return result


def save_chunk(path,frame,spec,ids,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False,compression='zstd');os.replace(tmp,path)
    write_json(path.with_suffix('.json'),{'spec':spec,'ids_hash':digest_ids(ids),'rows':len(frame),**metadata})


def read_cached_chunk(path,spec,ids,embedding_dim=None,provenance=None):
    mp=path.with_suffix('.json')
    if not path.is_file() and not mp.is_file():return None
    if not path.is_file() or not mp.is_file():raise RuntimeError(f'Partial cache, manual inspection required: {path}')
    meta=json.loads(mp.read_text(encoding='utf-8'))
    if meta['spec']!=spec or meta['rows']!=len(ids) or meta['ids_hash']!=digest_ids(ids):
        raise RuntimeError(f'Incompatible chunk: {path}. No overwriting.')
    if provenance is not None and any(meta.get(k)!=v for k,v in provenance.items()):
        raise RuntimeError(f'Cached embeddings use a different model/library revision: {path}')
    data=pd.read_parquet(path)
    if len(data)!=len(ids) or data['id'].astype(str).tolist()!=list(map(str,ids)):
        raise RuntimeError(f'Cache identity mismatch: {path}')
    n=len([c for c in data.columns if c.startswith('emb_')]);
    if n<1 or (embedding_dim is not None and n!=embedding_dim):raise RuntimeError(f'Unexpected dimension: {path}')
    return data


def extract_dataset(name,args,smoke=False):
    d,key,sha=load_split(name)
    root=Path(args.ruasd_root if name=='ruasd' else args.mlaad_root)
    validate_audio_sources(name,d,root)
    spec=manifest_spec(name,sha,args)
    out=Path(args.output_root)/name
    mfile=out/'manifest.json'
    if mfile.exists():
        m=json.loads(mfile.read_text(encoding='utf-8'))
        if m.get('spec')!=spec:raise RuntimeError(f'Protocol differs from existing cache: {mfile}')
    elif not smoke:
        write_json(mfile,{'spec':spec,'expected_rows':len(d),'description':'One entry per frozen split file_id/path; partial cache is resumable'})
    encoder=W2vBertEarlyLayer(args.device)
    provenance={'model_revision':encoder.model_revision,'torch_version':encoder.torch_version,
                'transformers_version':encoder.transformers_version}
    # A manifest created before the checkpoint was loaded is enriched on first load.
    # If restarted in another environment, refuse to mix embeddings from different
    # model revisions/library versions in the same feature store.
    if not smoke:
        mm=json.loads(mfile.read_text(encoding='utf-8'))
        if mm.get('extractor') is not None and mm['extractor']!=provenance:
            raise RuntimeError('Model revision or library version changed; refusing to mix cached representations')
        if mm.get('extractor') is None:
            mm['extractor']=provenance;write_json(mfile,mm)
    print(f'EXTRACT {name}: {len(d)} expected, chunk={args.chunk_size}, smoke={smoke}',flush=True)
    processed=0;reused=0;started=perf_counter()
    groups=(d.groupby('shard_name',sort=True) if name=='ruasd' else [('wav',d)])
    for grp,rows in groups:
        rows=rows.reset_index(drop=True)
        tf=None
        try:
            for offset in range(0,len(rows),args.chunk_size):
                group=rows.iloc[offset:offset+args.chunk_size]
                ids=group[key].astype(str).tolist()
                # Smoke never modifies production cache.
                cp=out/'chunks'/str(Path(grp).stem)/f'part_{offset:06d}.parquet'
                if not smoke and read_cached_chunk(cp,spec,ids,provenance=provenance) is not None:
                    reused+=len(group);continue
                if name=='ruasd' and tf is None:
                    tf=tarfile.open(root/str(grp),'r:*')
                vectors=[]
                for row in group.itertuples(index=False):
                    if name=='ruasd':
                        handle=tf.extractfile(str(row.audio_member))
                        if handle is None:raise FileNotFoundError(row.audio_member)
                        y=read_waveform(io.BytesIO(handle.read()),args.max_seconds)
                    else:
                        audio_path=find_mlaad_path(root,str(row.path))
                        if not audio_path.is_file():raise FileNotFoundError(audio_path)
                        y=read_waveform(str(audio_path),args.max_seconds)
                    vectors.append(encoder(y))
                    processed+=1
                    if processed<=3 or processed%100==0:
                        print(f'{name} NEW {processed}, cached {reused}',flush=True)
                    if smoke and processed>=args.smoke_records:
                        print('SMOKE OK:',name,'vectors',processed,'dim',vectors[-1].size,'elapsed_sec',round(perf_counter()-started,2))
                        return
                x=np.stack(vectors).astype(np.float32)
                fr=pd.DataFrame(x,columns=[f'emb_{j:04d}' for j in range(x.shape[1])])
                fr.insert(0,'generator',group['generator'].astype(str).tolist())
                fr.insert(0,'split',group['split'].astype(str).tolist())
                fr.insert(0,'id',ids)
                save_chunk(cp,fr,spec,ids,{'model_revision':encoder.model_revision,'torch_version':encoder.torch_version,
                    'transformers_version':encoder.transformers_version})
        finally:
            if tf is not None:tf.close()
    print(f'EXTRACTION COMPLETE dataset={name} new={processed} cached={reused} elapsed_sec={round(perf_counter()-started,1)}; run --mode assemble',flush=True)


def assemble(name,args):
    d,key,sha=load_split(name)
    out=Path(args.output_root)/name
    spec=manifest_spec(name,sha,args)
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest['spec']!=spec:raise RuntimeError('Protocol mismatch in assemble')
    parts=[]
    groups=(d.groupby('shard_name',sort=True) if name=='ruasd' else [('wav',d)])
    for grp,rows in groups:
        rows=rows.reset_index(drop=True)
        for offset in range(0,len(rows),args.chunk_size):
            group=rows.iloc[offset:offset+args.chunk_size]
            cp=out/'chunks'/str(Path(grp).stem)/f'part_{offset:06d}.parquet'
            data=read_cached_chunk(cp,spec,group[key].astype(str).tolist(),provenance=manifest.get('extractor'))
            if data is None:raise RuntimeError(f'Missing extracted chunk: {cp}')
            parts.append(data)
    result=pd.concat(parts,ignore_index=True)
    if len(result)!=len(d) or not result['id'].is_unique or result['id'].tolist()!=d[key].astype(str).tolist():
        raise RuntimeError('Assembled embeddings misaligned with frozen split')
    if result['split'].tolist()!=d['split'].tolist() or result['generator'].tolist()!=d['generator'].tolist():
        raise RuntimeError('Labels/splits misaligned')
    cols=[x for x in result if x.startswith('emb_')]
    arr=result[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():raise RuntimeError('NaN/Inf features')
    dest=out/'all_embeddings.parquet';tmp=dest.with_suffix('.tmp.parquet')
    result.to_parquet(tmp,index=False,compression='zstd');os.replace(tmp,dest)
    write_json(out/'assembled_metadata.json',{'rows':len(result),'dimension':len(cols),'split_sha256':sha,
        'embedding_data_sha256':sha256_file(dest),'spec':spec})
    print(f'ASSEMBLE OK {dest}: {len(result)} x {len(cols)}; SHA {sha256_file(dest)}',flush=True)


def load_emb(name,args):
    d,key,sha=load_split(name)
    path=Path(args.output_root)/name/'all_embeddings.parquet'
    if not path.is_file():raise FileNotFoundError(f'Run --mode assemble first: {path}')
    meta=json.loads((path.parent/'assembled_metadata.json').read_text(encoding='utf-8'))
    if meta['split_sha256']!=sha or meta['spec']!=manifest_spec(name,sha,args):raise RuntimeError('Embedding protocol mismatch')
    e=pd.read_parquet(path)
    if len(e)!=len(d) or e['id'].tolist()!=d[key].astype(str).tolist() or e['generator'].tolist()!=d.generator.tolist() or e['split'].tolist()!=d.split.tolist():
        raise RuntimeError('Saved embeddings not aligned to frozen labels')
    cols=[c for c in e.columns if c.startswith('emb_')]
    if not cols:raise RuntimeError('No embedding columns')
    x=e[cols].to_numpy(np.float32)
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite embeddings')
    return d,x,sha


def neighbors_cpu(ref,q,k,batch=256):
    from sklearn.neighbors import NearestNeighbors
    m=NearestNeighbors(n_neighbors=k,metric='euclidean',algorithm='brute',n_jobs=-1).fit(ref)
    dist=[];idx=[]
    for i in range(0,len(q),batch):
        a,b=m.kneighbors(q[i:i+batch],return_distance=True)
        dist.append(a.astype(np.float32));idx.append(b.astype(np.int32))
    return np.vstack(dist),np.vstack(idx)


def neighbors_gpu(ref,q,k,batch=128,ref_chunk=16384):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available; use --engine cpu')
    # Exact brute-force Euclidean kNN by chunked torch float32 matrix products.
    # Not approximate ANN. Distances are squared during top-k, sqrt afterwards.
    refs=torch.as_tensor(np.ascontiguousarray(ref),dtype=torch.float32,device='cuda')
    rn=(refs*refs).sum(dim=1)
    all_d=[];all_i=[]
    torch.backends.cuda.matmul.allow_tf32=False
    for s in range(0,len(q),batch):
        query=torch.as_tensor(np.ascontiguousarray(q[s:s+batch]),dtype=torch.float32,device='cuda')
        qn=(query*query).sum(dim=1,keepdim=True)
        dkeep=None;ikeep=None
        for pos in range(0,len(ref),ref_chunk):
            cur=refs[pos:pos+ref_chunk]
            sq=torch.clamp(qn+rn[pos:pos+len(cur)][None,:]-2*(query@cur.T),min=0)
            kk=min(k,len(cur));v,j=torch.topk(sq,kk,dim=1,largest=False,sorted=False);j+=pos
            if dkeep is not None:
                v=torch.cat([dkeep,v],dim=1);j=torch.cat([ikeep,j],dim=1)
            kk=min(k,v.shape[1]);ix=torch.topk(v,kk,dim=1,largest=False,sorted=True).indices
            dkeep=torch.gather(v,1,ix);ikeep=torch.gather(j,1,ix)
        all_d.append(torch.sqrt(dkeep).cpu().numpy());all_i.append(ikeep.cpu().numpy())
    return np.vstack(all_d),np.vstack(all_i)


def knn_predict(labels,neighbor_index):
    """Uniform voting, ties broken by lexicographic class order like sklearn KNN."""
    classes=np.unique(np.asarray(labels).astype(str));codes=np.searchsorted(classes,np.asarray(labels).astype(str))
    neighbor_codes=codes[neighbor_index]
    vote=np.zeros((len(neighbor_index),len(classes)),dtype=np.int32)
    rows=np.broadcast_to(np.arange(len(vote))[:,None],neighbor_codes.shape)
    np.add.at(vote,(rows.ravel(),neighbor_codes.ravel()),1)
    ranks=np.argsort(-vote,axis=1,kind='stable')
    return classes[ranks[:,0]],classes[ranks[:,:min(3,len(classes))]]


def make_predictor(train_x,train_y,engine):
    if len(train_x)<K:raise RuntimeError('Not enough train rows for k=21')
    def predict(q):
        dist,idx=(neighbors_gpu(train_x,q,K) if engine=='gpu' else neighbors_cpu(train_x,q,K))
        pred,top3=knn_predict(train_y,idx)
        return pred,top3,dist.mean(axis=1).astype(np.float64)
    return predict


def subset_metrics(y,pred,top3):
    from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
    return {'macro_f1':float(f1_score(y,pred,average='macro')),'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'top3':float(np.mean((np.asarray(y)[:,None]==top3).any(axis=1)))}


def calc_oscr(known_truth,known_pred,known_score,unknown_score):
    sk=np.asarray(known_score);su=np.asarray(unknown_score)
    ks=np.argsort(sk);sc=sk[ks];correct=(np.asarray(known_truth)==np.asarray(known_pred))[ks].astype(int)
    cumul=np.r_[0,np.cumsum(correct)];u=np.sort(su)
    thresholds=np.r_[np.inf,np.unique(np.r_[sk,su])[::-1],-np.inf]
    nk=np.searchsorted(sc,thresholds,side='left');nu=np.searchsorted(u,thresholds,side='left')
    ccr=(cumul[-1]-cumul[nk])/len(sk);fpr=(len(su)-nu)/len(su)
    order=np.argsort(fpr)
    # Do not evaluate np.trapz as getattr's default: NumPy >= 2.4 removed it.
    integrate=getattr(np,'trapezoid',None)
    if integrate is None:  # compatibility with NumPy < 2.0
        integrate=np.trapz
    return float(integrate(ccr[order],fpr[order]))


def open_scores_metrics(true_known,pred_known,score_known,score_unknown):
    from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
    yt=np.r_[np.ones(len(score_known),int),np.zeros(len(score_unknown),int)]
    s=np.r_[score_known,score_unknown]
    fpr,tpr,_=roc_curve(yt,s)
    ix=np.flatnonzero(tpr>=.95)
    fpr95=float(fpr[ix[0]]) if len(ix) else 1.0  # identical to source_attribution.metrics.fpr95
    return {'auroc':float(roc_auc_score(yt,s)), 'aupr_ood':float(average_precision_score(1-yt,-s)),
        'fpr95':fpr95,
        'oscr':calc_oscr(true_known,pred_known,score_known,score_unknown)}


def evaluate_closed(name,args):
    d,x,sha=load_emb(name,args)
    split=d['split'].to_numpy();y=d['generator'].astype(str).to_numpy()
    train=split=='train';val=split=='validation';test=split=='test'
    assert int(test.sum()) in (13600,4600)
    model=make_predictor(x[train],y[train],args.engine)
    pv,tv,sv=model(x[val]);pt,tt,st=model(x[test]);m=subset_metrics(y[test],pt,tt)
    from sklearn.metrics import classification_report,confusion_matrix
    out=Path(args.output_root)/name/'closed_set';out.mkdir(parents=True,exist_ok=True)
    pred=pd.DataFrame({'id':d.loc[test,'file_id' if name=='ruasd' else 'path'].astype(str).to_numpy(),
        'true_generator':y[test],'pred_generator':pt,'knn_mean_distance':st})
    pred.to_csv(out/'test_predictions.csv',index=False)
    classes=sorted(set(y));pd.DataFrame(confusion_matrix(y[test],pt,labels=classes),index=classes,columns=classes).to_csv(out/'confusion_matrix.csv')
    pd.DataFrame(classification_report(y[test],pt,labels=classes,output_dict=True,zero_division=0)).transpose().to_csv(out/'per_generator.csv')
    report={'protocol':'TADA_exact_representation_kNN21_closed_set','dataset':name,'split_sha256':sha,'knn_engine':args.engine,
        'n_classes':len(classes),'n_train':int(train.sum()),'n_val':int(val.sum()),'n_test':int(test.sum()),
        'metrics':m,'validation_metrics':subset_metrics(y[val],pv,tv),'top3_method':'kNN neighbor-vote counts'}
    write_json(out/'metrics.json',report)
    print('CLOSED-SET COMPLETE',name,json.dumps(m),out,flush=True)


def evaluate_open(name,args):
    d,x,sha=load_emb(name,args);folds,foldsha=load_folds(name,d.generator.unique())
    y=d.generator.astype(str).to_numpy();split=d.split.to_numpy()
    out=Path(args.output_root)/name/'open_set';out.mkdir(parents=True,exist_ok=True)
    from sklearn.metrics import f1_score
    results=[]
    for fi in range(1,6):
        file=out/f'fold{fi}_metrics.json'
        if file.exists() and not args.force:
            previous=json.loads(file.read_text(encoding='utf-8'))
            if previous['split_sha256']!=sha or previous['folds_sha256']!=foldsha or previous['engine']!=args.engine:
                raise RuntimeError(f'Mismatch with saved fold: {file}')
            results.append(previous);print('FOLD CACHED',fi,flush=True);continue
        f=folds[str(fi)];known=set(f['known_generators']);unknown=set(f['unknown_generators'])
        in_k=np.isin(y,list(known));in_u=np.isin(y,list(unknown))
        tr=(split=='train')&in_k;va=(split=='validation')&in_k;te=(split=='test')&in_k;tu=(split=='test')&in_u
        assert not tr[in_u].any() and not va[in_u].any() and tr.sum()>K
        infer=make_predictor(x[tr],y[tr],args.engine)
        _,_,dv=infer(x[va]);pk,top3,dk=infer(x[te]);pu,_,du=infer(x[tu])
        # Original TADA kNN mean-distance OOD score, but threshold only known val.
        threshold=float(np.quantile(dv,.95))
        truth=np.r_[y[te],np.repeat('unknown',tu.sum())]
        pred=np.r_[np.where(dk>threshold,'unknown',pk),np.where(du>threshold,'unknown',pu)]
        classes=sorted(known)+['unknown']
        scores=open_scores_metrics(y[te],pk,-dk,-du)
        scores.update({'macro_f1_unknown':float(f1_score(truth,pred,labels=classes,average='macro',zero_division=0)),
                       'known_accept':float(np.mean(dk<=threshold)), 'unknown_reject':float(np.mean(du>threshold)),
                       'known_macro_f1_before_rejection':float(f1_score(y[te],pk,labels=sorted(known),average='macro',zero_division=0))})
        report={'protocol':'TADA_w2vbert2_layer4_knn21_known_val_95pct_OOD_adaptation','dataset':name,'fold':fi,
            'split_sha256':sha,'folds_sha256':foldsha,'engine':args.engine,'known_generators':sorted(known),
            'unknown_generators':sorted(unknown),'n_train':int(tr.sum()),'n_val':int(va.sum()),
            'n_known_test':int(te.sum()),'n_unknown_test':int(tu.sum()),'distance_threshold_known_val_q95':threshold,
            'metrics':scores,'not_original_TADA_OOD_threshold':'Stan et al. use EER with OOD validation samples; this study forbids using withheld generators in validation'}
        write_json(file,report)
        pd.DataFrame({'id':pd.concat([d.loc[te,'file_id' if name=='ruasd' else 'path'],d.loc[tu,'file_id' if name=='ruasd' else 'path']]).astype(str).to_numpy(),
                      'true_label':truth,'prediction':pred,
                      'known_id_score':np.r_[-dk,-du]}).to_csv(out/f'fold{fi}_test_predictions.csv',index=False)
        results.append(report)
        print('OPEN FOLD',fi,'AUROC',round(scores['auroc'],4),'FPR95',round(scores['fpr95'],4),'MacroF1+unknown',round(scores['macro_f1_unknown'],4),flush=True)
    fields=list(results[0]['metrics'])
    rows=[{'fold':r['fold'],**r['metrics']} for r in results];pd.DataFrame(rows).to_csv(out/'five_folds.csv',index=False)
    mean={k:float(np.mean([r['metrics'][k] for r in results])) for k in fields}
    sd={k:float(np.std([r['metrics'][k] for r in results],ddof=1)) for k in fields}
    write_json(out/'summary.json',{'dataset':name,'n_folds':5,'folds_sha256':foldsha,'mean':mean,'sd_across_folds':sd,
        'description':'Five fixed generator-level folds, one deterministic kNN configuration; NOT five random seeds; no unknown in threshold tuning'})
    print('OPEN-SET COMPLETE',name,json.dumps(mean,indent=2),out,flush=True)


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=['preflight','smoke','extract','assemble','closed','open'],required=True)
    p.add_argument('--dataset',choices=['ruasd','mlaad'],required=True)
    p.add_argument('--ruasd-root',default='data/raw/RuASD')
    p.add_argument('--mlaad-root',default=DEFAULT_MLAAD_ROOT)
    p.add_argument('--output-root',default=str(DEFAULT_ROOT))
    p.add_argument('--max-seconds',type=float,default=4.0,help='4 RuASD fixed4, 10 MLAAD canonical (state separately in article)')
    p.add_argument('--chunk-size',type=int,default=64)
    p.add_argument('--smoke-records',type=int,default=2)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--engine',choices=['gpu','cpu'],default='gpu',help='Exact Euclidean kNN via torch GPU or sklearn CPU')
    p.add_argument('--force',action='store_true',help='Only overwrite existing OOD fold results explicitly')
    return p.parse_args()


def main():
    args=parse()
    if args.chunk_size<1 or args.smoke_records<1 or args.max_seconds<=0:raise ValueError('Invalid positive argument')
    d,key,sha=load_split(args.dataset)
    folds,foldsha=load_folds(args.dataset,d.generator.unique())
    print(f'FROZEN VERIFIED dataset={args.dataset} n={len(d)} classes={d.generator.nunique()} splitSHA={sha} foldsSHA={foldsha}',flush=True)
    if args.mode=='preflight':
        validate_audio_sources(args.dataset,d,Path(args.ruasd_root if args.dataset=='ruasd' else args.mlaad_root))
        import importlib.util
        missing=[]
        for module in ['torch','transformers','librosa','soundfile','pyarrow','sklearn']:
            ok=bool(importlib.util.find_spec(module))
            print(f'IMPORT {module}: {ok}')
            if not ok: missing.append(module)
        if missing: raise RuntimeError('Missing modules: '+', '.join(missing)+'; see README install instructions')
        import torch
        print('CUDA AVAILABLE:',torch.cuda.is_available(),flush=True)
        print('PREFLIGHT OK (does not download/model load or predict).',flush=True)
    elif args.mode in ('smoke','extract'):extract_dataset(args.dataset,args,smoke=args.mode=='smoke')
    elif args.mode=='assemble':assemble(args.dataset,args)
    elif args.mode=='closed':evaluate_closed(args.dataset,args)
    elif args.mode=='open':evaluate_open(args.dataset,args)

if __name__=='__main__':main()
