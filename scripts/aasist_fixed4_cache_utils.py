"""Lossless, versioned float32 cache for fixed-4s RuASD AASIST.

The cache stores exactly the 64,000 post-preprocessing samples in original
split CSV row order. AASIST's required 600 right-padding zeros are inserted
when reading. No test material is cached by this module.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

CONTENT_SAMPLES = 64000
NETWORK_SAMPLES = 64600
VERSION = 'aasist_fixed4_cache_float32_exact_v1'


def ids_digest(ids):
    h = hashlib.sha256()
    for fid in ids:
        h.update(fid.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()


def paths(root: Path, split_name: str):
    root = Path(root)
    return (root / f'{split_name}_content_float32.npy',
            root / f'{split_name}_metadata.json',
            root / f'{split_name}_content_float32.building.npy',
            root / f'{split_name}_progress.json')


def expected_meta(df, partition, split_hash):
    ids = df['file_id'].astype(str).tolist()
    return {'format': VERSION, 'partition': partition,
            'split_sha256': split_hash,
            'file_ids_sha256': ids_digest(ids),
            'shape': [len(ids), CONTENT_SAMPLES],
            'dtype': 'float32',
            'preprocessing': 'source_attribution.audio.preprocess_audio mono 16k peak_normalize max_duration_sec=4.0 min_audio_sec=.10 + fixed_length start4s',
            'model_padding': '600 exact zeros on right, added at load time'}


def validate_meta(meta, expected):
    for key, val in expected.items():
        if meta.get(key) != val:
            raise RuntimeError(f'AASIST cache mismatch for {key}: found {meta.get(key)!r}, expected {val!r}')


class Fixed4CacheDataset(Dataset):
    def __init__(self, df, labels, cache_root, partition, split_hash):
        if not len(df) or not (df['split'].astype(str) == partition).all():
            raise RuntimeError(f'Cached {partition}: bad or empty partition')
        path, meta_path, _, _ = paths(cache_root, partition)
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f'Cache for {partition} incomplete. Run scripts/build_aasist_fixed4_cache.py first')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        validate_meta(meta, expected_meta(df, partition, split_hash))
        self.array_path = str(path)
        self.expected_shape = (len(df), CONTENT_SAMPLES)
        self._arr = None
        self.ids = df['file_id'].astype(str).tolist()
        lookup = {name: i for i, name in enumerate(labels)}
        self.targets = [lookup[g] for g in df['generator'].astype(str)]
        self._open()

    def _open(self):
        self._arr = np.load(self.array_path, mmap_mode='r', allow_pickle=False)
        if self._arr.shape != self.expected_shape or self._arr.dtype != np.dtype('float32'):
            raise RuntimeError('Cached numpy array has wrong shape or dtype')

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arr'] = None   # each DataLoader worker reopens its own handle
        return state

    def __getitem__(self, idx):
        if self._arr is None:
            self._open()
        content = np.asarray(self._arr[idx])
        if not np.isfinite(content).all():
            raise RuntimeError(f'Nonfinite cached audio for file_id={self.ids[idx]}')
        out = np.zeros(NETWORK_SAMPLES, dtype=np.float32)
        out[:CONTENT_SAMPLES] = content
        return torch.from_numpy(out), self.targets[idx], self.ids[idx]

    def close(self):
        self._arr = None
