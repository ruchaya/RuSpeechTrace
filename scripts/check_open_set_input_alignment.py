#!/usr/bin/env python
"""Strict preflight test before long paired 4s open-set baselines."""
from pathlib import Path
import numpy as np
import pandas as pd
import json

split = pd.read_csv('splits/closed_set.csv', encoding='utf-8-sig')
assert len(split) == 68000 and split.file_id.is_unique
assert split.generator.nunique() == 34
assert split.split.value_counts().to_dict() == {'train':44200, 'validation':10200, 'test':13600}
if 'text_hash' in split:
    groups = {s:set(split.loc[split.split.eq(s),'text_hash'].dropna().astype(str)) for s in ('train','validation','test')}
    assert not (groups['train'] & groups['validation'])
    assert not (groups['train'] & groups['test'])
    assert not (groups['validation'] & groups['test'])
folds = json.loads(Path('splits/open_set_folds.json').read_text(encoding='utf-8'))['folds']
classes = set(split.generator.astype(str))
unknown_list = []
for fid, part in folds.items():
    known, unknown = set(part['known_generators']), set(part['unknown_generators'])
    assert known.isdisjoint(unknown) and known | unknown == classes, fid
    unknown_list.extend(unknown)
assert len(unknown_list) == 34 and set(unknown_list) == classes
for name,path,prefix,count in [
    ('acoustic4s','features/ruasd_closed_set_acoustic_fixed4.parquet','spectral_',789),
    ('wavlm4s','embeddings/wavlm_base_ruasd_4s/all_68000_for_open_set.parquet','emb_',768)]:
    frame = pd.read_parquet(path)
    assert len(frame) == 68000 and frame.file_id.is_unique, name
    assert set(frame.file_id.astype(str)) == set(split.file_id.astype(str)), name
    if name == 'acoustic4s':
        cols = [c for c in frame.columns if c.startswith(('spectral_','prosody_')) and c != 'prosody_duration_sec']
    else:
        cols = [c for c in frame.columns if c.startswith('emb_')]
    assert len(cols) == count, (name, len(cols))
    assert not np.isinf(frame[cols].to_numpy(dtype=np.float64)).any(), name
    print(f'{name}: {len(frame)} rows, {len(cols)} model features; all IDs match')
print('Preflight OK: same frozen split, 5 generator-level folds, no file-ID loss')
