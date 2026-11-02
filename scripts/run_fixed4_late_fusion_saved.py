#!/usr/bin/env python
"""Leakage-safe late fusion of *already trained* Acoustic and WavLM LR models.

Run from the project's repository root after merging this script into scripts/.
Does not re-tune the base classifiers and never selects alpha on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from source_attribution.metrics import closed_set_metrics, per_class_metrics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_models(path: Path):
    files = [path / name / 'model.joblib' for name in ('acoustic_fixed4', 'wavlm_fixed4')]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Already trained LR model(s) not found: {missing}')
    return (joblib.load(files[0]), joblib.load(files[1]), files)


def load_features(acoustic_path: Path, wavlm_chunks: Path, expected_ids: set[str]):
    acoustic_columns = [c for c in pd.read_parquet(acoustic_path).columns if c.startswith(('spectral_', 'prosody_'))]
    if len(acoustic_columns) != 789 or 'prosody_duration_sec' in acoustic_columns:
        raise ValueError(f'Expected exactly 789 acoustic columns without explicit duration; got {len(acoustic_columns)}')
    aa = pd.read_parquet(acoustic_path, columns=['file_id'] + acoustic_columns)
    chunk_paths = sorted(wavlm_chunks.glob('*.parquet'))
    if len(chunk_paths) != 120:
        raise ValueError(f'Expected 120 WavLM Parquet chunks; found {len(chunk_paths)} in {wavlm_chunks}')
    emb_cols = [f'emb_{j:04d}' for j in range(768)]
    bb = pd.concat([pd.read_parquet(p, columns=['file_id']+emb_cols) for p in chunk_paths], ignore_index=True)
    for name, frame in [('acoustic', aa), ('wavlm', bb)]:
        frame['file_id'] = frame['file_id'].astype(str)
        if frame.file_id.duplicated().any() or set(frame.file_id) != expected_ids:
            raise RuntimeError(f'{name}: duplicate or missing/extra file IDs vs frozen split')
        if not np.isfinite(frame.drop(columns='file_id').to_numpy(dtype=np.float64)).all():
            # The original acoustic representation has a few legitimate F0 NaNs, which its
            # fitted LR pipeline imputes. Do not conflate NaNs with infinite values.
            if np.isinf(frame.drop(columns='file_id').to_numpy(dtype=np.float64)).any():
                raise RuntimeError(f'{name}: infinite features')
    return aa, acoustic_columns, bb, emb_cols


def align_probabilities(model, features: pd.DataFrame, expected_classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_).astype(str)
    if len(set(classes)) != len(classes) or set(classes) != set(expected_classes):
        raise RuntimeError('Model class set differs from frozen split')
    p = np.asarray(model.predict_proba(features), dtype=np.float64)
    if p.shape != (len(features), len(classes)) or not np.isfinite(p).all():
        raise RuntimeError('Invalid classifier probability matrix')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise RuntimeError('Probabilities do not sum to 1')
    by_class = {v: j for j,v in enumerate(classes)}
    return p[:, [by_class[v] for v in expected_classes]]


def choose_alpha(y_val, p_acoustic, p_wavlm, classes, grid):
    """Select WavLM weight strictly by validation Macro-F1 (accuracy tie-break)."""
    records = []
    best_key = (-np.inf, -np.inf)
    selected = None
    for alpha in grid:
        fused = (1 - alpha) * p_acoustic + alpha * p_wavlm
        pred = classes[fused.argmax(axis=1)]
        f1 = float(f1_score(y_val, pred, labels=classes, average='macro', zero_division=0))
        acc = float(np.mean(y_val == pred))
        records.append({'alpha_wavlm': float(alpha), 'validation_macro_f1': f1, 'validation_accuracy': acc})
        # First maximizer in ascending alpha grid wins -> simpler acoustic-heavy model on ties.
        if (f1, acc) > best_key:
            best_key, selected = (f1, acc), float(alpha)
    return selected, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', default='splits/closed_set.csv')
    p.add_argument('--acoustic', default='features/ruasd_closed_set_acoustic_fixed4.parquet')
    p.add_argument('--wavlm-chunks', default='embeddings/wavlm_base_ruasd_4s/chunks')
    p.add_argument('--models', default='results/raw/fixed4_acoustic_vs_wavlm_lr')
    p.add_argument('--output', default='results/raw/fixed4_acoustic_wavlm_late_fusion_saved')
    p.add_argument('--alpha-step', type=float, default=0.05)
    a = p.parse_args()
    if a.alpha_step <= 0 or 1 / a.alpha_step > 1000:
        p.error('--alpha-step must be positive and reasonable')
    split_path, acoustic_path = Path(a.splits), Path(a.acoustic)
    sp = pd.read_csv(split_path, usecols=['file_id','split','generator'],encoding='utf-8-sig')
    sp['file_id'] = sp.file_id.astype(str)
    sp['generator'] = sp.generator.astype(str)
    if sp.file_id.duplicated().any() or len(sp) != 68000:
        raise RuntimeError('Frozen split must have 68,000 unique file IDs')
    if sp.split.value_counts().to_dict() != {'train':44200, 'validation':10200, 'test':13600}:
        raise RuntimeError('Unexpected RuASD frozen split counts')
    classes = np.asarray(sorted(sp.generator.unique()), dtype=str)
    if len(classes) != 34:
        raise RuntimeError('Expected 34 generators in frozen split')
    acoustic_model, wavlm_model, model_files = get_models(Path(a.models))
    af, acols, wf, wcols = load_features(acoustic_path, Path(a.wavlm_chunks), set(sp.file_id))
    # One-to-one merge with strict validation; no intersection-only silently dropping rows.
    df = sp.merge(af, on='file_id', validate='one_to_one').merge(wf, on='file_id', validate='one_to_one',suffixes=('','_wavlm'))
    if len(df) != len(sp):
        raise RuntimeError('Feature merge changed the number of records')
    va, te = df.split.eq('validation'), df.split.eq('test')
    p_av = align_probabilities(acoustic_model, df.loc[va,acols], classes)
    p_wv = align_probabilities(wavlm_model, df.loc[va,wcols], classes)
    y_val = df.loc[va,'generator'].to_numpy(dtype=str)
    grid = np.unique(np.r_[np.arange(0,1+1e-9,a.alpha_step), 1.0])
    alpha, grid_rows = choose_alpha(y_val,p_av,p_wv,classes,grid)
    p_at = align_probabilities(acoustic_model, df.loc[te,acols], classes)
    p_wt = align_probabilities(wavlm_model, df.loc[te,wcols], classes)
    pt = (1-alpha)*p_at + alpha*p_wt
    y_test = df.loc[te,'generator'].to_numpy(dtype=str)
    pred = classes[pt.argmax(axis=1)]
    test_metrics = closed_set_metrics(y_test,pred,pt,classes)
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(grid_rows).to_csv(out/'validation_alpha_grid.csv', index=False)
    pd.DataFrame(per_class_metrics(y_test,pred)).to_csv(out/'class_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'file_id':df.loc[te,'file_id'].to_numpy(), 'generator_true':y_test, 'generator_pred':pred}).to_parquet(out/'predictions.parquet',index=False)
    np.save(out/'test_probabilities.npy', pt.astype(np.float32))
    (out/'classes.txt').write_text('\n'.join(classes),encoding='utf-8')
    meta = {'method':'late_fusion_lr_fixed4_existing_models', 'alpha_wavlm':alpha,
            'alpha_acoustic':1-alpha,'validation_metric':'macro_f1_with_accuracy_tiebreak',
            'validation_best_macro_f1':max(x['validation_macro_f1'] for x in grid_rows),
            'test':test_metrics,'split_sha256':sha256(split_path),
            'acoustic_features_sha256':sha256(acoustic_path),
            'base_model_sha256':[sha256(file) for file in model_files],
            'class_order':classes.tolist(),'n_val':int(va.sum()),'n_test':int(te.sum()),
            'base_model_paths':[str(file) for file in model_files]}
    (out/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'alpha_wavlm':alpha,'test':test_metrics,'output':str(out)},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
