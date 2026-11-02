import io

import numpy as np
import soundfile as sf

from source_attribution.acoustic_features import (
    AcousticFeatureConfig,
    extract_acoustic_features,
)
from source_attribution.audio import preprocess_audio


def test_acoustic_feature_count_790():
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    y = 0.4 * np.sin(2 * np.pi * 220.0 * t)

    feats = extract_acoustic_features(
        y,
        sr,
        original_duration=2.0,
        cfg=AcousticFeatureConfig(sr=sr, include_duration=True),
    )

    spectral = [k for k in feats if k.startswith("spectral_")]
    prosody = [k for k in feats if k.startswith("prosody_")]

    assert len(spectral) == 777
    assert len(prosody) == 13
    assert len(feats) == 790


def test_audio_file_object_preprocessing():
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    y = 0.2 + 0.4 * np.sin(2 * np.pi * 220.0 * t)

    bio = io.BytesIO()
    sf.write(bio, y, sr, format="WAV", subtype="PCM_16")
    bio.seek(0)

    out, out_sr, original_duration = preprocess_audio(
        bio,
        target_sr=16000,
        peak_normalize=True,
        max_duration_sec=10.0,
    )

    assert out_sr == 16000
    assert abs(original_duration - 1.0) < 0.01
    assert abs(float(np.mean(out))) < 1e-4
    assert np.max(np.abs(out)) <= 1.0001
