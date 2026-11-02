#!/usr/bin/env python
"""Lightweight AASIST TAR/4s preprocessing test; no GPU or RuASD required."""
import io
import sys
import tarfile
import tempfile
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from run_aasist_fixed4_ruasd import RuASDTarFixed4, CONTENT_SAMPLES, AASIST_SAMPLES
from source_attribution.audio import preprocess_audio, fixed_length


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tar_path = root / 'fixture.tar'
        x = np.linspace(-0.3, 0.3, 5 * 16000, dtype=np.float32)
        short = np.sin(np.arange(3200) * .08).astype(np.float32) * 0.2
        wavs = {}
        for filename, arr in [('long.wav', x), ('short.wav', short)]:
            b = io.BytesIO(); sf.write(b, arr, 16000, format='WAV', subtype='PCM_16')
            wavs[filename] = b.getvalue()
        with tarfile.open(tar_path, 'w') as tf:
            for name, raw in wavs.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(raw)
                tf.addfile(info, io.BytesIO(raw))
        dframe = pd.DataFrame([
            {'file_id': 'long_id', 'generator': 'genA', 'shard_name':tar_path.name,'audio_member':'long.wav'},
            {'file_id': 'short_id','generator':'genB','shard_name':tar_path.name,'audio_member':'short.wav'},
        ])
        ds = RuASDTarFixed4(dframe, ['genA', 'genB'], root)
        for i, name in enumerate(['long.wav', 'short.wav']):
            y,label,fid = ds[i]
            assert tuple(y.shape) == (AASIST_SAMPLES,)
            assert label == i
            assert fid == dframe.iloc[i]['file_id']
            assert np.count_nonzero(y.numpy()[CONTENT_SAMPLES:]) == 0
            yy, sr, _ = preprocess_audio(io.BytesIO(wavs[name]), target_sr=16000,
                  mono=True, peak_normalize=True, max_duration_sec=4.0, min_audio_sec=0.1)
            expected = fixed_length(yy,sr,4.0,mode='start')
            np.testing.assert_array_equal(y.numpy()[:CONTENT_SAMPLES],expected)
        assert len(pickle.loads(pickle.dumps(ds))) == 2
        ds.close()
    print('PASS: TAR reader, exact project preprocessing, 64000 content + 600 right zeros, Windows pickling')


if __name__ == '__main__':
    main()
