#!/usr/bin/env python
"""RuASD 34-class, fixed 4s AASIST baseline (official architecture).

Separate from the project's original scripts/run_aasist.py. Does not
train on or select checkpoints with test data. The first 4 seconds of audio
are processed with the same source_attribution.audio utilities used for
WavLM; 600 right-side zeros reach official AASIST's 64600-sample input.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from source_attribution.audio import preprocess_audio, fixed_length
from source_attribution.aasist_adapter import load_official_aasist

SAMPLE_RATE = 16000
CONTENT_SAMPLES = 64000
AASIST_SAMPLES = 64600
EXPECTED_COUNTS = {'train': 44200, 'validation': 10200, 'test': 13600}
EXPECTED_PER_CLASS = {'train': 1300, 'validation': 300, 'test': 400}
OFFICIAL_COMMIT = 'a04c9863f63d44471dde8a6abcb3b082b07cd1d1'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def get_commit(repo_root: Path) -> str:
    return subprocess.check_output(['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], text=True).strip()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_split(split_path: Path, tar_root: Path, *, require_full=True):
    required = ['file_id', 'generator', 'split', 'shard_name', 'audio_member', 'text_hash']
    df = pd.read_csv(split_path, usecols=lambda c: c in required + ['raw_or_augmented'])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RuntimeError(f'Missing required split columns: {missing}')
    if df[required].isna().any().any():
        raise RuntimeError('Missing file ID, generator, split, shard, member or text hash')
    if df['file_id'].duplicated().any():
        raise RuntimeError('Duplicate file_id in split')
    if (df['split'].astype(str).isin(EXPECTED_COUNTS) == False).any():
        raise RuntimeError('Unexpected partition name')
    if 'raw_or_augmented' in df and not (df['raw_or_augmented'].astype(str) == 'raw').all():
        raise RuntimeError('Augmented sample found in frozen split')
    labels = sorted(df['generator'].astype(str).unique().tolist())
    if len(labels) != 34:
        raise RuntimeError(f'Expected 34 generators, found {len(labels)}')
    if require_full:
        if len(df) != 68000 or df['split'].value_counts().to_dict() != EXPECTED_COUNTS:
            raise RuntimeError('Not the expected frozen 68k split')
        counts = df.groupby(['generator', 'split']).size()
        for g in labels:
            for s, expected in EXPECTED_PER_CLASS.items():
                if counts.get((g, s), 0) != expected:
                    raise RuntimeError(f'Incorrect quota for {g}/{s}')
        texts = {s: set(df.loc[df['split'] == s, 'text_hash'].astype(str)) for s in EXPECTED_COUNTS}
        if any(texts[a] & texts[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise RuntimeError('Text leakage across train/validation/test!')
        shards = sorted(df['shard_name'].astype(str).unique())
        absent = [s for s in shards if not (tar_root / s).is_file()]
        if absent:
            raise FileNotFoundError(f'Missing TAR files, examples: {absent[:5]}')
    return df, labels


class RuASDTarFixed4(Dataset):
    """Lazy, TAR-aware WAV reader. TAR index/handles cached per process.

    `shard_name` is used instead of the machine-specific absolute `shard_path`.
    Windows DataLoader workers initialize their own handles; default workers=0.
    """
    def __init__(self, df: pd.DataFrame, labels: list[str], tar_root: Path, max_open_shards=128):
        d = df.reset_index(drop=True)
        self.ids = d['file_id'].astype(str).tolist()
        self.shards = d['shard_name'].astype(str).tolist()
        self.members = d['audio_member'].astype(str).tolist()
        lookup = {name: n for n,name in enumerate(labels)}
        self.targets = [lookup[g] for g in d['generator'].astype(str)]
        self.tar_root = str(tar_root)
        self.max_open_shards = int(max_open_shards)
        self._tars = collections.OrderedDict()

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_tars'] = collections.OrderedDict()
        return state

    def close(self):
        for handle in self._tars.values():
            try: handle.close()
            except Exception: pass
        self._tars.clear()

    def __del__(self):
        if hasattr(self, '_tars'):
            self.close()

    def _archive(self, shard: str):
        if shard in self._tars:
            self._tars.move_to_end(shard)
            return self._tars[shard]
        handle = tarfile.open(str(Path(self.tar_root) / shard), mode='r:')
        self._tars[shard] = handle
        if len(self._tars) > self.max_open_shards:
            _, old = self._tars.popitem(last=False)
            old.close()
        return handle

    def __getitem__(self, index):
        shard, member = self.shards[index], self.members[index]
        try:
            tf = self._archive(shard)
            entry = tf.getmember(member)
            fh = tf.extractfile(entry)
            if fh is None:
                raise OSError('TAR member has no readable contents')
            with fh:
                audio_bytes = fh.read()
            y, sr, _ = preprocess_audio(io.BytesIO(audio_bytes), target_sr=SAMPLE_RATE,
                 mono=True, peak_normalize=True, max_duration_sec=4.0, min_audio_sec=0.10)
            y = fixed_length(y, sr, 4.0, mode='start')
            if y.shape != (CONTENT_SAMPLES,):
                raise ValueError(f'Wrong content shape: {y.shape}')
            # IMPORTANT: no signal from beyond the first 4 seconds.
            # AASIST's *original* config requires 64600 samples; add silence only.
            y = np.pad(y, (0, AASIST_SAMPLES-CONTENT_SAMPLES), mode='constant').astype(np.float32)
            if not np.isfinite(y).all():
                raise ValueError('NaN/Inf audio')
            return torch.from_numpy(y), self.targets[index], self.ids[index]
        except Exception as exc:
            raise RuntimeError(f'Failed file_id={self.ids[index]}, TAR={shard}, member={member}: {exc}') from exc


def eval_net(model, loader, device, *, record_probs=False, amp=False):
    model.eval()
    ids, truths, preds, probs = [], [], [], []
    summed_loss = 0.0
    total = 0
    with torch.inference_mode():
        for x, y, fid in loader:
            x = x.to(device, non_blocking=True)
            target = y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(amp and device.type=='cuda')):
                _, logits = model(x, Freq_aug=False)
                loss = torch.nn.functional.cross_entropy(logits.float(), target)
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            summed_loss += float(loss.item()) * len(y)
            total += len(y)
            truths.extend(y.numpy().tolist())
            preds.extend(np.argmax(p, axis=1).tolist())
            ids.extend(list(fid))
            if record_probs:
                probs.append(p)
    truth = np.asarray(truths, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    result = {
        'loss': summed_loss / max(1, total),
        'accuracy': float(accuracy_score(truth, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
        'macro_f1': float(f1_score(truth, pred, average='macro')),
        'n_samples': int(total),
    }
    if record_probs:
        mat = np.concatenate(probs, axis=0)
        result['top1_accuracy'] = float(np.mean(np.argmax(mat, axis=1) == truth))
        result['top3_accuracy'] = float(np.mean(np.any(np.argsort(mat, axis=1)[:, -3:] == truth[:, None], axis=1)))
        return result, ids, truth, pred, mat
    return result


def make_loader(dataset, *, batch_size, shuffle, workers, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=g,
                      persistent_workers=(workers > 0))


def smoketest(args, df, labels, device):
    # Take known classes from both train and validation. Never read test for smoke.
    chosen = labels[:2]
    tr = pd.concat([df[(df['generator'] == c) & (df['split'] == 'train')].head(3)
                    for c in chosen], ignore_index=True)
    va = pd.concat([df[(df['generator'] == c) & (df['split'] == 'validation')].head(2)
                    for c in chosen], ignore_index=True)
    train_ds = RuASDTarFixed4(tr, labels, args.tar_root)
    valid_ds = RuASDTarFixed4(va, labels, args.tar_root)
    t0 = time.perf_counter()
    x, y, ids = next(iter(make_loader(train_ds, batch_size=2, shuffle=False, workers=0, seed=args.seed)))
    print(f'TAR/4s smoke: {len(x)} samples; shape={tuple(x.shape)}; '
          f'last_600_are_zero={bool(torch.count_nonzero(x[:, CONTENT_SAMPLES:]) == 0)}; '
          f'read_seconds={time.perf_counter()-t0:.2f}', flush=True)
    assert tuple(x.shape) == (2, AASIST_SAMPLES)
    assert torch.count_nonzero(x[:, CONTENT_SAMPLES:]).item() == 0
    model, cfg = load_official_aasist(args.repo_root, len(labels))
    assert cfg['nb_samp'] == AASIST_SAMPLES
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    t1 = time.perf_counter()
    _, logits = model(x.to(device), Freq_aug=False)
    assert tuple(logits.shape) == (2, 34)
    loss = torch.nn.functional.cross_entropy(logits, y.to(device))
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError('Non-finite smoke loss')
    loss.backward()
    opt.step()
    print(f'GPU smoke train: loss={loss.item():.5f}; seconds={time.perf_counter()-t1:.2f}; '
          f'max_allocated_MiB={torch.cuda.max_memory_allocated()/1024**2:.0f}'
          if device.type=='cuda' else f'CPU smoke train: loss={loss.item():.5f}', flush=True)
    val_loader = make_loader(valid_ds, batch_size=2, shuffle=False, workers=0, seed=args.seed)
    v = eval_net(model, val_loader, device)
    assert v['n_samples'] == 4
    print('SMOKE PASS: TAR reading, first-4s preprocessing, forward/backward, validation.', flush=True)
    print('Not a scientific performance estimate. No test examples accessed.', flush=True)
    train_ds.close(); valid_ds.close()


def build_run_meta(args, labels, split_hash, commit):
    return {'protocol': 'ruasd_34class_aasist_first4s_peaknorm_v1',
            'audio_input': '16k mono, first 4s (64000), DC-removal and peaknorm; 600 right zeros for original AASIST 64600',
            'architecture': 'official clovaai/aasist, full training from scratch; binary output head changed to 34',
            'aasist_commit': commit, 'frozen_split_sha256': split_hash,
            'n_classes': len(labels), 'classes_in_output_order': labels,
            'seed': args.seed, 'batch_size': args.batch_size, 'gradient_accumulation': args.accumulation,
            'optimizer': 'Adam', 'lr': args.lr, 'weight_decay': args.weight_decay,
            'scheduler': 'CosineAnnealingLR', 'epochs_max': args.epochs,
            'patience': args.patience, 'amp_bfloat16': args.amp,
            'torch_version': torch.__version__, 'device': str(args.device),
            'train_val_test': EXPECTED_COUNTS, 'num_workers': args.workers}


def save_predict(run_dir, part, result, ids, truth, pred, proba, labels):
    records = pd.DataFrame({'file_id': ids, 'generator_true': [labels[z] for z in truth],
                            'generator_pred': [labels[z] for z in pred]})
    # Full probability vectors are needed for top-k verification / follow-up analysis.
    for i in range(len(labels)):
        records[f'prob_{i:02d}'] = proba[:, i]
    records.to_parquet(run_dir / f'{part}_predictions.parquet', index=False)
    p,r,f,s = precision_recall_fscore_support(truth,pred,labels=np.arange(len(labels)),zero_division=0)
    pd.DataFrame({'generator':labels,'precision':p,'recall':r,'f1':f,'support':s}).to_csv(
        run_dir / f'{part}_per_generator.csv',index=False)
    pd.DataFrame(confusion_matrix(truth,pred,labels=np.arange(len(labels))),index=labels,columns=labels).to_csv(
        run_dir / f'{part}_confusion_matrix.csv')
    return result


def train_model(args, df, labels, device, commit):
    run_dir = args.output / f'aasist_fixed4_ruasd34_seed{args.seed}'
    run_dir.mkdir(parents=True,exist_ok=True)
    meta = build_run_meta(args, labels, sha256(args.splits), commit)
    metadata_file = run_dir / 'run_metadata.json'
    if metadata_file.exists():
        existing = json.loads(metadata_file.read_text(encoding='utf-8'))
        if existing != meta:
            raise RuntimeError('Output already contains an INCOMPATIBLE run. Use another --output or seed.')
    else:
        atomic_json(metadata_file, meta)
    best_path = run_dir/'best_model.pt'
    last_path = run_dir/'last_training_state.pt'
    if best_path.exists() and not args.resume:
        raise RuntimeError('Checkpoint exists; use --resume to continue or choose another run folder.')
    train_ds = RuASDTarFixed4(df[df['split']=='train'], labels, args.tar_root)
    val_ds = RuASDTarFixed4(df[df['split']=='validation'], labels, args.tar_root)
    test_ds = RuASDTarFixed4(df[df['split']=='test'], labels, args.tar_root)
    val_loader = make_loader(val_ds,batch_size=args.batch_size,shuffle=False,workers=args.workers,seed=args.seed)
    model,cfg = load_official_aasist(args.repo_root,len(labels))
    if cfg['nb_samp'] != AASIST_SAMPLES:
        raise RuntimeError('Official AASIST nb_samp mismatch')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=5e-6)
    start_epoch,best_f1,best_loss,best_epoch,bad_epochs = 1,-1.0,float('inf'),0,0
    history = []
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f'Resume requested; missing {last_path}')
        state = torch.load(last_path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['epoch']) + 1
        best_f1,best_loss,best_epoch,bad_epochs = state['best_f1'],state['best_loss'],state['best_epoch'],state['bad_epochs']
        history=state['history']
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        print(f'Resuming after epoch {start_epoch-1} / {args.epochs}', flush=True)
    for epoch in range(start_epoch,args.epochs+1):
        if bad_epochs >= args.patience:
            print('Patience exhausted; skipping remaining epochs.', flush=True)
            break
        epoch_start=time.perf_counter()
        train_loader=make_loader(train_ds,batch_size=args.batch_size,shuffle=True,workers=args.workers,seed=args.seed+epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum=0.0; seen=0
        for step,(x,y,_ids) in enumerate(train_loader,1):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(args.amp and device.type=='cuda')):
                _h,z=model(x,Freq_aug=False)
                loss=torch.nn.functional.cross_entropy(z.float(),y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss epoch={epoch}, step={step}')
            (loss / args.accumulation).backward()
            if (step%args.accumulation==0) or (step==len(train_loader)):
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            seen += len(y);loss_sum += float(loss.item())*len(y)
            if step%args.log_every==0:
                print(f'  epoch={epoch} step={step}/{len(train_loader)} train_seen={seen} '
                      f'loss={loss_sum/max(1,seen):.5f}',flush=True)
        scheduler.step()
        model.eval()
        val=eval_net(model,val_loader,device,amp=args.amp)
        improved = ((val['macro_f1'] > best_f1+1e-8) or
                    (abs(val['macro_f1']-best_f1)<=1e-8 and val['loss'] < best_loss))
        if improved:
            best_f1,best_loss,best_epoch,bad_epochs = val['macro_f1'],val['loss'],epoch,0
            tmp=best_path.with_name('best_model.partial.pt')
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'epoch':epoch,'validation_metrics':val,'metadata':meta},tmp)
            os.replace(tmp,best_path)
        else:
            bad_epochs+=1
        hist={'epoch':epoch,'train_loss':loss_sum/max(1,seen),'validation':val,
              'lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-epoch_start}
        history.append(hist)
        atomic_json(run_dir/'epoch_history.json',history)
        last={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
              'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
              'epoch':epoch,'best_f1':best_f1,'best_loss':best_loss,'best_epoch':best_epoch,
              'bad_epochs':bad_epochs,'history':history, 'torch_rng':torch.get_rng_state(),
              'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
              'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        tmp=last_path.with_name('last_training_state.partial.pt')
        torch.save(last,tmp);os.replace(tmp,last_path)
        print(f'EPOCH {epoch}: train_loss={hist["train_loss"]:.5f} '
              f'val_macro_f1={val["macro_f1"]:.5f} val_loss={val["loss"]:.5f} '
              f'best_epoch={best_epoch} elapsed_s={hist["seconds"]:.0f}',flush=True)
    if not best_path.is_file():
        raise RuntimeError('No selected best checkpoint')
    chosen = torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(chosen['model']);model.eval()
    final = {'protocol':meta, 'checkpoint_epoch':int(chosen['epoch']),
             'validation_selected_metrics':chosen['validation_metrics']}
    for partition, dataset in [('validation',val_ds),('test',test_ds)]:
        loader=make_loader(dataset,batch_size=args.batch_size,shuffle=False,workers=args.workers,seed=args.seed)
        met,ids,truth,pred,proba=eval_net(model,loader,device,record_probs=True,amp=args.amp)
        save_predict(run_dir,partition,met,ids,truth,pred,proba,labels)
        final[partition]=met
        print(f'FINAL {partition} macro_f1={met["macro_f1"]:.5f} '
              f'top1={met["top1_accuracy"]:.5f} top3={met["top3_accuracy"]:.5f}', flush=True)
    atomic_json(run_dir/'final_metrics.json',final)
    print('Saved to',run_dir,flush=True)
    train_ds.close();val_ds.close();test_ds.close()


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['smoke','train'],default='smoke')
    ap.add_argument('--splits',type=Path,default=Path('splits/closed_set.csv'))
    ap.add_argument('--tar-root',type=Path,default=Path(os.environ.get("RUASD_ROOT", "data/raw/RuASD")))
    ap.add_argument('--repo-root',type=Path,default=Path('third_party/aasist'))
    ap.add_argument('--output',type=Path,default=Path('results/raw'))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--accumulation',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--log-every',type=int,default=500)
    ap.add_argument('--amp',action='store_true',help='Use CUDA BF16 autocast. Smoke defaults to full FP32.')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--confirm-full',action='store_true',help='Mandatory guard for full 68k experiment')
    return ap.parse_args()


def main():
    args=parse_args()
    args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.batch_size < 1 or args.accumulation < 1 or args.workers < 0:
        raise ValueError('Invalid batch/accumulation/workers')
    if args.mode=='train' and not args.confirm_full:
        raise RuntimeError('Full experiment requires explicit --confirm-full')
    if args.mode=='train' and args.device!='cuda':
        raise RuntimeError('GPU unavailable; refusing very slow full training')
    if args.epochs<1 or args.patience<1:
        raise ValueError('epochs/patience must be >= 1')
    df,labels=load_split(args.splits,args.tar_root)
    commit=get_commit(args.repo_root)
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f'AASIST repo commit {commit}; expected {OFFICIAL_COMMIT}')
    seed_everything(args.seed)
    device=torch.device(args.device)
    print(f'Protocol AASIST first4s / RuASD34 / {args.mode} / GPU={args.device}',flush=True)
    print(f'Frozen splits SHA256={sha256(args.splits)}; official AASIST commit={commit}',flush=True)
    if args.mode=='smoke':
        smoketest(args,df,labels,device)
    else:
        train_model(args,df,labels,device,commit)


if __name__=='__main__':
    # Needed for Windows DataLoader spawn.
    main()
