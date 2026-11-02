#!/usr/bin/env python
"""Read-only throughput and GPU-memory preflight for AASIST fixed-4s RuASD.

Reads ONLY training audio. Trains on a tiny fixed subset in RAM for timing;
does not update or save experimental checkpoints and does not touch test audio.
The scientific trainer remains scripts/run_aasist_fixed4_ruasd.py unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_aasist_fixed4_ruasd import (
    AASIST_SAMPLES,
    OFFICIAL_COMMIT,
    RuASDTarFixed4,
    get_commit,
    load_split,
    load_official_aasist,
    seed_everything,
    sha256,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits', type=Path, default=Path('splits/closed_set.csv'))
    ap.add_argument('--tar-root', type=Path, default=Path(os.environ.get("RUASD_ROOT", "data/raw/RuASD")))
    ap.add_argument('--repo-root', type=Path, default=Path('third_party/aasist'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--items-per-class', type=int, default=2,
                    help='Read this many real TRAIN records for each of 34 generators; default 68 total')
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[2, 4, 8])
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--amp', action='store_true', help='Benchmark optional BF16 autocast, same as trainer --amp')
    ap.add_argument('--output', type=Path, default=Path('results/raw/aasist_fixed4_preflight_benchmark.json'))
    return ap.parse_args()


def load_real_training_audio(args, df, labels):
    train_df = df.loc[df['split'] == 'train']
    sub = (train_df.groupby('generator', group_keys=False)
           .sample(n=args.items_per_class, random_state=args.seed)
           .sample(frac=1, random_state=args.seed)
           .reset_index(drop=True))
    expected = args.items_per_class * len(labels)
    if len(sub) != expected or not (sub['split'] == 'train').all():
        raise RuntimeError('Sample selection unexpectedly mixed partitions or dropped classes')
    sample_ids_hash = hashlib.sha256('\n'.join(sorted(sub.file_id.astype(str))).encode('utf8')).hexdigest()
    ds = RuASDTarFixed4(sub, labels, args.tar_root)
    xs, ys = [], []
    try:
        print(f'Reading {expected} real TRAIN WAV members from TAR archives, without accessing validation/test...',
              flush=True)
        t0 = time.perf_counter()
        for j in range(len(ds)):
            x, y, _ = ds[j]
            xs.append(x)
            ys.append(y)
            if (j + 1) % 17 == 0 or j + 1 == len(ds):
                print(f'  cold TAR read: {j + 1}/{len(ds)}; elapsed={time.perf_counter()-t0:.1f}s',
                      flush=True)
        cold_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        for j in range(len(ds)):
            x, y, _ = ds[j]
            if not torch.equal(x, xs[j]) or int(y) != ys[j]:
                raise RuntimeError(f'Non-reproducible audio loading for TRAIN sample {j}')
        warm_s = time.perf_counter() - t1
    finally:
        ds.close()
    print(f'Audio read: cold={cold_s:.2f}s, repeated={warm_s:.2f}s for {len(xs)} samples', flush=True)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long), {
        'n_train_samples_read': len(xs), 'sample_ids_sha256': sample_ids_hash,
        'tar_read_cold_s': round(cold_s, 4), 'tar_read_repeat_s': round(warm_s, 4),
        'tar_read_repeat_s_per_sample': round(warm_s / len(xs), 6),
    }


def bench_batch(args, batch_size, all_x, all_y, n_classes):
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(args.seed)
    model, cfg = load_official_aasist(args.repo_root, n_classes)
    if cfg['nb_samp'] != AASIST_SAMPLES:
        raise ValueError(f'Official AASIST nb_samp expected {AASIST_SAMPLES}, got {cfg["nb_samp"]}')
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    train_times = []
    eval_times = []
    total_steps = args.warmup + args.steps
    try:
        print(f'  GPU batch={batch_size}; amp_bfloat16={args.amp}; '
              f'warmup={args.warmup}; timed_train_steps={args.steps}', flush=True)
        for step in range(total_steps):
            offset = (step * batch_size) % (len(all_x) - batch_size + 1)
            xb = all_x[offset:offset + batch_size].cuda(non_blocking=False)
            yb = all_y[offset:offset + batch_size].cuda(non_blocking=False)
            model.train()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
                _, logits = model(xb, Freq_aug=False)
                loss = torch.nn.functional.cross_entropy(logits.float(), yb)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss in GPU benchmark')
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= args.warmup:
                train_times.append(dt)
        model.eval()
        for step in range(total_steps):
            offset = (step * batch_size) % (len(all_x) - batch_size + 1)
            xb = all_x[offset:offset + batch_size].cuda(non_blocking=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
                    _, logits = model(xb, Freq_aug=False)
            torch.cuda.synchronize()
            if step >= args.warmup:
                eval_times.append(time.perf_counter() - t0)
        return {
            'batch_size': batch_size, 'status': 'ok',
            'train_step_seconds_median': round(float(np.median(train_times)), 5),
            'train_step_seconds_mean': round(float(np.mean(train_times)), 5),
            'eval_step_seconds_median': round(float(np.median(eval_times)), 5),
            'max_gpu_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
            'max_gpu_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
            'approx_train_gpu_minutes_per_epoch': round(
                math.ceil(44200 / batch_size) * float(np.median(train_times)) / 60.0, 1),
            'approx_validation_gpu_minutes_per_epoch': round(
                math.ceil(10200 / batch_size) * float(np.median(eval_times)) / 60.0, 1),
        }
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.items_per_class < 1 or args.steps < 1 or args.warmup < 0:
        raise ValueError('items-per-class and steps must be positive, warmup nonnegative')
    if not args.batch_sizes or any(bs < 1 or bs > args.items_per_class * 34 for bs in args.batch_sizes):
        raise ValueError('Batch sizes must be between 1 and the number of selected training records')
    commit = get_commit(args.repo_root)
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f'Unexpected official AASIST commit: {commit}')
    df, labels = load_split(args.splits, args.tar_root, require_full=True)
    print(f'Split SHA256={sha256(args.splits)}, official AASIST commit={commit}', flush=True)
    print('CUDA:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
    if args.amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 autocast not supported on this GPU; rerun without --amp')
    all_x, all_y, read_metrics = load_real_training_audio(args, df, labels)
    result = {
        'purpose': 'benchmark only; NOT scientific evaluation or completed training',
        'train_only': True, 'n_frozen_train': 44200, 'n_frozen_validation': 10200,
        'split_sha256': sha256(args.splits), 'aasist_commit': commit,
        'torch': str(torch.__version__), 'cuda_version': str(torch.version.cuda),
        'gpu': torch.cuda.get_device_name(0), 'amp_bfloat16': args.amp,
        'seed': args.seed, 'samples_per_generator': args.items_per_class,
        'batch_sizes': args.batch_sizes, **read_metrics, 'gpu_benchmarks': [],
        'notes': ('GPU step times exclude TAR I/O, scheduling, shuffle, checkpoint saving and '
                  'the validation metrics computation. First 68 (or configured number) '
                  'training samples cannot predict full-epoch disk throughput reliably. '
                  'Approximate epoch numbers are lower-bound indicators, not promises.'),
    }
    for bs in args.batch_sizes:
        try:
            m = bench_batch(args, bs, all_x, all_y, len(labels))
        except torch.cuda.OutOfMemoryError as exc:
            m = {'batch_size': bs, 'status': 'cuda_oom', 'error': str(exc).splitlines()[0][:250]}
            gc.collect(); torch.cuda.empty_cache()
        result['gpu_benchmarks'].append(m)
        print('RESULT', json.dumps(m), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.json.partial')
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, args.output)
    print(f'BENCHMARK COMPLETE (no test audio, no scientific checkpoint); report={args.output}', flush=True)


if __name__ == '__main__':
    main()
