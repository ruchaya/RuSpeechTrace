#!/usr/bin/env python
"""Resume-safe, exact-float32 TAR audio cache for TRAIN and VALIDATION only.

Reads each TAR group together to avoid repeated cold archive indexing. Preserves
frozen CSV ordering in the cache, and never accesses test audio.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np

from run_aasist_fixed4_ruasd import RuASDTarFixed4, load_split, sha256
from aasist_fixed4_cache_utils import (CONTENT_SAMPLES, Fixed4CacheDataset,
    expected_meta, paths, validate_meta)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def build_one(df_all, labels, partition, *, tar_root, cache_root, split_hash, checkpoint_every, verify_n, seed):
    df = df_all.loc[df_all['split'] == partition].reset_index(drop=True)
    expected = expected_meta(df, partition, split_hash)
    dest, meta_path, working, progress_path = paths(cache_root, partition)
    cache_root.mkdir(parents=True, exist_ok=True)
    if dest.exists() and meta_path.exists():
        validate_meta(json.loads(meta_path.read_text(encoding='utf-8')), expected)
        cached = Fixed4CacheDataset(df, labels, cache_root, partition, split_hash)
        print(f'CACHE HIT {partition}: {len(cached)} rows; verified metadata', flush=True)
        cached.close()
    elif dest.exists() or meta_path.exists():
        raise RuntimeError(f'Incomplete finalized cache for {partition}: {dest}, {meta_path}. Inspect manually.')
    else:
        required = len(df) * CONTENT_SAMPLES * 4
        free_bytes = shutil.disk_usage(cache_root).free
        if not working.exists() and free_bytes < required + (1 << 30):
            raise RuntimeError(f'Insufficient disk space for {partition} float32 cache: requires {required/2**30:.1f} GiB +1GiB overhead, free {free_bytes/2**30:.1f} GiB')
        ds = RuASDTarFixed4(df, labels, tar_root, max_open_shards=2)
        # Stable order avoids repeated cold indexing across many TAR shards.
        order = sorted(range(len(df)), key=lambda i: (ds.shards[i], ds.members[i]))
        if working.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
            validate_meta(progress['expected'], expected)
            cursor = int(progress['cursor'])
            if not (0 <= cursor <= len(ds)):
                raise RuntimeError('Invalid cache resume cursor')
            arr = np.load(working, mmap_mode='r+', allow_pickle=False)
            if arr.shape != (len(ds), CONTENT_SAMPLES) or arr.dtype != np.float32:
                raise RuntimeError('Wrong resume array format')
            print(f'RESUME {partition} from {cursor}/{len(ds)} rows', flush=True)
        elif working.exists() or progress_path.exists():
            raise RuntimeError(f'Broken cache checkpoint for {partition}; do not delete without inspection')
        else:
            arr = np.lib.format.open_memmap(working, mode='w+', dtype=np.float32,
                                            shape=(len(ds), CONTENT_SAMPLES))
            cursor = 0
            atomic_json(progress_path, {'expected': expected, 'cursor': cursor})
            print(f'BUILD {partition}: {len(ds)} rows, {required/2**30:.2f} GiB float32', flush=True)
        started = time.perf_counter()
        try:
            for pos in range(cursor, len(order)):
                idx = order[pos]
                audio, _, fid = ds[idx]
                if audio.shape != (64600,) or bool(audio[64000:].count_nonzero()):
                    raise RuntimeError(f'Invalid model audio for {fid}')
                arr[idx, :] = audio[:64000].numpy()
                if (pos + 1) % checkpoint_every == 0 or pos + 1 == len(order):
                    arr.flush()
                    atomic_json(progress_path, {'expected': expected, 'cursor': pos+1})
                    print(f'  {partition}: {pos+1}/{len(order)}, elapsed={time.perf_counter()-started:.0f}s', flush=True)
            arr.flush()
            del arr
            os.replace(working, dest)
            atomic_json(meta_path, expected)
            progress_path.unlink(missing_ok=True)
        finally:
            ds.close()
        print(f'BUILT {partition} cache: {dest}', flush=True)
    # Check sampled elements byte-for-byte against original audio preprocessing.
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(df), size=min(verify_n, len(df)), replace=False))
    original = RuASDTarFixed4(df, labels, tar_root, max_open_shards=2)
    cached = Fixed4CacheDataset(df, labels, cache_root, partition, split_hash)
    try:
        for idx in chosen:
            a, y, fid = original[int(idx)]
            b, y2, fid2 = cached[int(idx)]
            if not np.array_equal(a.numpy(), b.numpy()) or y != y2 or fid != fid2:
                raise RuntimeError(f'Exact cache verification failed at {partition} {idx} {fid}')
    finally:
        original.close()
        cached.close()
    print(f'CACHE VERIFIED {partition}: exact float32 values on {len(chosen)} sampled source WAV', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--splits',type=Path,default=Path('splits/closed_set.csv'))
    p.add_argument('--tar-root',type=Path,default=Path(os.environ.get("RUASD_ROOT", "data/raw/RuASD")))
    p.add_argument('--cache-root',type=Path,default=Path('data/cache/aasist_fixed4_float32'))
    p.add_argument('--partitions',nargs='+',choices=['train','validation'],default=['train','validation'])
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--verify-n',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.checkpoint_every<1 or args.verify_n<1:
        p.error('checkpoint-every and verify-n must be >=1')
    df, labels = load_split(args.splits, args.tar_root)
    split_hash = sha256(args.splits)
    print('AASIST FLOAT32 CACHE | frozen split sha256=',split_hash,flush=True)
    for part in args.partitions:
        build_one(df,labels,part,tar_root=args.tar_root,cache_root=args.cache_root,
                  split_hash=split_hash,checkpoint_every=args.checkpoint_every,
                  verify_n=args.verify_n,seed=args.seed)
    print('CACHE COMPLETE: train/validation only; test WAV never touched',flush=True)

if __name__ == '__main__':
    main()
