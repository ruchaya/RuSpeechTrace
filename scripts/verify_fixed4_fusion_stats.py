#!/usr/bin/env python
"""Paired stratified-bootstrap and exact McNemar audit of fixed-4s predictions.

Only reads saved test predictions; never trains or selects a fusion weight.
Checks identical frozen test IDs, 34-class labels and class counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


RUNS = {
    'acoustic_lr': Path('results/raw/fixed4_acoustic_vs_wavlm_lr/acoustic_fixed4/predictions.parquet'),
    'wavlm_lr': Path('results/raw/fixed4_acoustic_vs_wavlm_lr/wavlm_fixed4/predictions.parquet'),
    'fusion': Path('results/raw/fixed4_acoustic_wavlm_late_fusion_saved/predictions.parquet'),
}
PAIRS = [('fusion', 'acoustic_lr'), ('fusion', 'wavlm_lr'), ('acoustic_lr', 'wavlm_lr')]


def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [0.0] * len(pvals)
    cumulative = 0.0
    for rank, i in enumerate(order):
        cumulative = max(cumulative, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = cumulative
    return out


def load_aligned(splits_path: Path, files: dict[str, Path]):
    base = pd.read_csv(splits_path, usecols=['file_id', 'generator', 'split'], encoding='utf-8-sig')
    if base.file_id.duplicated().any():
        raise ValueError('Duplicate file_id in frozen split')
    te = base.loc[base.split == 'test', ['file_id', 'generator']].copy()
    te['file_id'] = te.file_id.astype(str)
    te['generator'] = te.generator.astype(str)
    if len(te) != 13600 or te.generator.nunique() != 34 or not te.generator.value_counts().eq(400).all():
        raise ValueError('Expected frozen test: 13,600 objects, 34 classes, 400 per class')
    merged = te.rename(columns={'generator': 'true'}).copy()
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing predictions for {name}: {path}')
        df = pd.read_parquet(path, columns=['file_id', 'generator_true', 'generator_pred']).copy()
        df.file_id = df.file_id.astype(str)
        if df.file_id.duplicated().any() or len(df) != len(te) or set(df.file_id) != set(te.file_id):
            raise ValueError(f'{name}: predictions do not equal frozen test file IDs')
        df.generator_true = df.generator_true.astype(str)
        df.generator_pred = df.generator_pred.astype(str)
        merged = merged.merge(df.rename(columns={'generator_true': f'true_{name}', 'generator_pred': name}), on='file_id', how='left', validate='one_to_one')
        if not merged.true.eq(merged[f'true_{name}']).all():
            raise ValueError(f'{name}: true labels differ from frozen split')
        merged.drop(columns=f'true_{name}', inplace=True)
    classes = sorted(te.generator.unique())
    if any(~merged[name].isin(classes).all() for name in files):
        raise ValueError('An unexpected predicted class was found')
    return merged, classes


def bootstrap(merged, classes, pairs, repetitions: int, seed: int):
    label_id = {name: i for i, name in enumerate(classes)}
    y = merged.true.map(label_id).to_numpy(dtype=np.int64)
    preds = {name: merged[name].map(label_id).to_numpy(dtype=np.int64) for name in RUNS}
    c = len(classes)
    groups = [np.flatnonzero(y == j) for j in range(c)]
    n = len(y)
    rng = np.random.default_rng(seed)
    result = {name: {'macro_f1': np.empty(repetitions), 'accuracy': np.empty(repetitions)} for name in RUNS}
    for b in range(repetitions):
        sample = np.concatenate([rng.choice(ind, size=len(ind), replace=True) for ind in groups])
        for name, pr in preds.items():
            cm = np.bincount((y[sample] * c + pr[sample]), minlength=c * c).reshape(c, c)
            result[name]['macro_f1'][b] = macro_f1_from_cm(cm)
            result[name]['accuracy'][b] = np.trace(cm) / n
    base_scores = {
        name: {
            'accuracy': float(accuracy_score(merged.true, merged[name])),
            'balanced_accuracy': float(balanced_accuracy_score(merged.true, merged[name])),
            'macro_f1': float(f1_score(merged.true, merged[name], labels=classes, average='macro', zero_division=0)),
        }
        for name in RUNS
    }
    interval = lambda arr: [float(v) for v in np.quantile(arr, [.025, .975])]
    for name in RUNS:
        for metric in ('macro_f1', 'accuracy'):
            base_scores[name][f'{metric}_95ci'] = interval(result[name][metric])
    pair_rows = []
    for one, two in pairs:
        a = merged[one].eq(merged.true).to_numpy()
        b = merged[two].eq(merged.true).to_numpy()
        n_first_only = int(np.sum(a & ~b))
        n_second_only = int(np.sum(~a & b))
        discordants = n_first_only + n_second_only
        pval = float(binomtest(n_first_only, discordants, .5).pvalue) if discordants else 1.0
        row = {'first': one, 'second': two, 'only_first_correct': n_first_only,
               'only_second_correct': n_second_only, 'mcnemar_exact_p': pval}
        for metric in ('macro_f1','accuracy'):
            d = result[one][metric] - result[two][metric]
            row[f'delta_{metric}'] = base_scores[one][metric] - base_scores[two][metric]
            row[f'delta_{metric}_95ci'] = interval(d)
        pair_rows.append(row)
    adjusted = holm([row['mcnemar_exact_p'] for row in pair_rows])
    for row, pval in zip(pair_rows, adjusted):
        row['mcnemar_holm_p'] = pval
    return base_scores, pair_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits', default='splits/closed_set.csv')
    ap.add_argument('--n-bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/statistical_tests/fixed4_fusion_comparison_20261021')
    args = ap.parse_args()
    if args.n_bootstrap < 100:
        ap.error('At least 100 bootstrap repetitions needed')
    aligned, classes = load_aligned(Path(args.splits), RUNS)
    scores, comparisons = bootstrap(aligned, classes, PAIRS, args.n_bootstrap, args.seed)
    dest = Path(args.out)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing statistical output: {dest}')
    dest.mkdir(parents=True,exist_ok=False)
    document = {'protocol': 'fixed4_RuASD_34generators_paired_test_bootstrap',
                'n_test':len(aligned), 'n_classes':len(classes), 'bootstrap_repetitions':args.n_bootstrap,
                'seed':args.seed,
                'scope': 'Fixed pretrained/fitted models, same test samples; no train or corpus uncertainty',
                'base_metrics':scores, 'paired_comparisons':comparisons}
    (dest/'metrics.json').write_text(json.dumps(document, ensure_ascii=False, indent=2),encoding='utf-8')
    pd.DataFrame(comparisons).to_csv(dest/'paired_comparisons.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
