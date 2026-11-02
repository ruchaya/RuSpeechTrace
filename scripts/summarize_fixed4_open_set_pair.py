"""Summarize two completed fixed-4s open-set runs, seed 42 (descriptive only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('results/raw')
NAMES = {
    'acoustic_lr_C1': 'open_set_acoustic_fixed4_lr_C1',
    'wavlm_lr_C1': 'open_set_wavlm_fixed4_lr_C1',
}
METRICS = [
    'auroc_id_vs_ood', 'aupr_ood', 'fpr95', 'oscr',
    'macro_f1_with_unknown', 'known_accept_rate', 'unknown_reject_rate',
]


def newest_metrics(experiment):
    candidates = sorted((ROOT / experiment).glob('*/metrics.json'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'Missing metrics.json for {experiment}. First finish the full run.')
    return candidates[-1]


def main():
    frames = []
    paths = {}
    for name, directory in NAMES.items():
        p = newest_metrics(directory)
        paths[name] = str(p)
        obj = json.loads(p.read_text(encoding='utf-8'))
        if len(obj) != 5:
            raise ValueError(f'{name}: expected 5 folds, received {len(obj)}')
        for fold, fd in obj.items():
            for method, d in fd['methods'].items():
                frames.append({'representation': name, 'fold': str(fold), 'method': method,
                               **{key: float(d[key]) for key in METRICS}})
    per_fold = pd.DataFrame(frames)
    methods = set(per_fold.loc[per_fold.representation == 'acoustic_lr_C1','method'])
    assert methods == set(per_fold.loc[per_fold.representation == 'wavlm_lr_C1','method'])
    for method in sorted(methods):
        left = set(per_fold.loc[(per_fold.representation == 'acoustic_lr_C1') & (per_fold.method == method), 'fold'])
        right = set(per_fold.loc[(per_fold.representation == 'wavlm_lr_C1') & (per_fold.method == method), 'fold'])
        if left != right or len(left) != 5:
            raise ValueError(f'Fold mismatch for {method}')
    summary = per_fold.groupby(['representation', 'method'],sort=True)[METRICS].agg(['mean','std'])
    summary.columns = [f'{m}_{stat}' for m,stat in summary.columns]
    summary = summary.reset_index()

    out = Path('results/summaries')
    out.mkdir(parents=True, exist_ok=True)
    path1 = out / 'open_set_fixed4_lr_C1_seed42_per_fold.csv'
    path2 = out / 'open_set_fixed4_lr_C1_seed42_summary.csv'
    path3 = out / 'open_set_fixed4_lr_C1_seed42_sources.json'
    per_fold.to_csv(path1,index=False)
    summary.to_csv(path2,index=False)
    path3.write_text(json.dumps({'source_metrics': paths, 'seed': 42,
        'notes': 'Five folds, single seed; descriptive statistics. Different test known/unknown subsets by fold. No inferential p-values.'},indent=2),encoding='utf-8')
    print('\nCompleted runs:',json.dumps(paths,indent=2))
    print('\nFive-fold means (single seed=42, NO significance claim):')
    show=['representation','method','auroc_id_vs_ood_mean','fpr95_mean','oscr_mean', 'macro_f1_with_unknown_mean', 'unknown_reject_rate_mean']
    print(summary[show].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f'\nWrote {path1} and {path2} and {path3}')


if __name__=='__main__':
    main()
