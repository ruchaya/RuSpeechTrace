from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from scipy import stats
from scipy.fft import dct


# Bump this whenever the scientific definition of the features changes.
FEATURE_DEFINITION_VERSION = "legacy_790_v1"

EPS = 1e-10
SUMMARY_STATS = ("mean", "std", "min", "max", "median", "skew", "kurtosis")


def _safe_scalar(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else np.nan


def _summarize(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    """
    Seven statistics per coefficient, matching the previous-study notebook.
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    output: dict[str, float] = {}

    for index, row in enumerate(matrix):
        finite = row[np.isfinite(row)]
        values = {
            "mean": np.mean(finite) if finite.size else np.nan,
            "std": np.std(finite) if finite.size else np.nan,
            "min": np.min(finite) if finite.size else np.nan,
            "max": np.max(finite) if finite.size else np.nan,
            "median": np.median(finite) if finite.size else np.nan,
            "skew": stats.skew(finite, bias=False) if finite.size > 2 else np.nan,
            "kurtosis": (
                stats.kurtosis(finite, fisher=True, bias=False)
                if finite.size > 3
                else np.nan
            ),
        }
        for stat_name, value in values.items():
            output[f"{prefix}_{index:02d}_{stat_name}"] = _safe_scalar(value)

    return output


def _linear_filterbank(
    sr: int,
    n_fft: int = 512,
    n_filters: int = 40,
) -> np.ndarray:
    frequencies = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    bank = np.zeros((n_filters, len(frequencies)), dtype=np.float64)

    for index in range(n_filters):
        left, center, right = edges[index:index + 3]
        bank[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-8),
                (right - frequencies) / max(right - center, 1e-8),
            ),
        )
    return bank


@dataclass(frozen=True)
class AcousticFeatureConfig:
    sr: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160

    n_mfcc: int = 20
    n_lfcc: int = 20
    n_cqcc: int = 20
    n_lfcc_filters: int = 40

    cqcc_hop_length: int = 256
    cqcc_n_bins: int = 72
    cqcc_bins_per_octave: int = 12

    f0_min_hz: float = 60.0
    f0_max_hz: float = 500.0
    f0_frame_length: int = 1024

    energy_frame_length: int = 400
    pause_percentile: float = 25.0
    pause_threshold_multiplier: float = 1.5
    pause_threshold_floor: float = 1e-4
    pause_min_duration_sec: float = 0.10

    include_duration: bool = True


def _mfcc_features(
    y: np.ndarray,
    sr: int,
    cfg: AcousticFeatureConfig,
) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )

    # Previous-study notebook used a safe adaptive width for short signals.
    width = min(
        9,
        mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1,
    )
    if width < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        delta = librosa.feature.delta(
            mfcc, width=width, order=1, mode="nearest"
        )
        delta2 = librosa.feature.delta(
            mfcc, width=width, order=2, mode="nearest"
        )

    return (
        _summarize(mfcc, "spectral_mfcc")
        | _summarize(delta, "spectral_mfcc_delta")
        | _summarize(delta2, "spectral_mfcc_delta2")
    )


def _lfcc_features(
    y: np.ndarray,
    sr: int,
    cfg: AcousticFeatureConfig,
) -> dict[str, float]:
    power = np.abs(
        librosa.stft(
            y,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
        )
    ) ** 2

    energies = (
        _linear_filterbank(
            sr,
            n_fft=cfg.n_fft,
            n_filters=cfg.n_lfcc_filters,
        )
        @ power
    )
    lfcc = dct(
        np.log(np.maximum(energies, EPS)),
        type=2,
        axis=0,
        norm="ortho",
    )[:cfg.n_lfcc]

    return _summarize(lfcc, "spectral_lfcc")


def _cqcc_features(
    y: np.ndarray,
    sr: int,
    cfg: AcousticFeatureConfig,
) -> dict[str, float]:
    # This deliberately reproduces the previous project's practical
    # CQT-derived approximation; it is not claimed as a strict reference CQCC.
    cqt = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=cfg.cqcc_hop_length,
            n_bins=cfg.cqcc_n_bins,
            bins_per_octave=cfg.cqcc_bins_per_octave,
        )
    )
    cqcc = dct(
        np.log(np.maximum(cqt, EPS)),
        type=2,
        axis=0,
        norm="ortho",
    )[:cfg.n_cqcc]

    return _summarize(cqcc, "spectral_cqcc")


def _stft_features(
    y: np.ndarray,
    sr: int,
    cfg: AcousticFeatureConfig,
) -> dict[str, float]:
    kwargs = dict(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )
    matrices = {
        "spectral_stft_centroid": librosa.feature.spectral_centroid(**kwargs),
        "spectral_stft_bandwidth": librosa.feature.spectral_bandwidth(**kwargs),
        "spectral_stft_rolloff": librosa.feature.spectral_rolloff(**kwargs),
        "spectral_stft_flatness": librosa.feature.spectral_flatness(
            y=y,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
        ),
        "spectral_stft_contrast": librosa.feature.spectral_contrast(**kwargs),
    }

    out: dict[str, float] = {}
    for prefix, matrix in matrices.items():
        out.update(_summarize(matrix, prefix))
    return out


def _f0_features(
    y: np.ndarray,
    sr: int,
    cfg: AcousticFeatureConfig,
) -> dict[str, float]:
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=cfg.f0_min_hz,
            fmax=cfg.f0_max_hz,
            sr=sr,
            frame_length=cfg.f0_frame_length,
            hop_length=cfg.hop_length,
        )
        valid = f0[np.isfinite(f0)]
        return {
            "prosody_f0_mean": (
                _safe_scalar(np.mean(valid)) if valid.size else np.nan
            ),
            "prosody_f0_std": (
                _safe_scalar(np.std(valid)) if valid.size else np.nan
            ),
            "prosody_f0_min": (
                _safe_scalar(np.min(valid)) if valid.size else np.nan
            ),
            "prosody_f0_max": (
                _safe_scalar(np.max(valid)) if valid.size else np.nan
            ),
            "prosody_voiced_ratio": (
                _safe_scalar(np.mean(voiced_flag.astype(float)))
                if voiced_flag is not None
                else 0.0
            ),
        }
    except Exception:
        return {
            "prosody_f0_mean": np.nan,
            "prosody_f0_std": np.nan,
            "prosody_f0_min": np.nan,
            "prosody_f0_max": np.nan,
            "prosody_voiced_ratio": np.nan,
        }


def _energy_features(
    y: np.ndarray,
    cfg: AcousticFeatureConfig,
) -> tuple[dict[str, float], np.ndarray]:
    rms = librosa.feature.rms(
        y=y,
        frame_length=cfg.energy_frame_length,
        hop_length=cfg.hop_length,
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=cfg.energy_frame_length,
        hop_length=cfg.hop_length,
    )[0]

    features = {
        "prosody_rms_mean": _safe_scalar(np.mean(rms)),
        "prosody_rms_std": _safe_scalar(np.std(rms)),
        "prosody_zcr_mean": _safe_scalar(np.mean(zcr)),
        "prosody_zcr_std": _safe_scalar(np.std(zcr)),
    }
    return features, rms


def _pause_features(
    y: np.ndarray,
    sr: int,
    rms: np.ndarray,
    cfg: AcousticFeatureConfig,
) -> dict[str, float]:
    if len(rms) == 0:
        return {
            "prosody_pause_count": 0.0,
            "prosody_pause_mean_sec": 0.0,
            "prosody_pause_ratio": 0.0,
        }

    threshold = max(
        float(np.percentile(rms, cfg.pause_percentile))
        * cfg.pause_threshold_multiplier,
        cfg.pause_threshold_floor,
    )
    silent = rms < threshold

    transitions = np.diff(np.pad(silent.astype(int), (1, 1)))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    durations = (ends - starts) * cfg.hop_length / sr
    durations = durations[durations >= cfg.pause_min_duration_sec]

    total_duration = max(len(y) / sr, 1e-8)
    return {
        "prosody_pause_count": float(len(durations)),
        "prosody_pause_mean_sec": (
            _safe_scalar(np.mean(durations)) if durations.size else 0.0
        ),
        "prosody_pause_ratio": _safe_scalar(
            np.sum(durations) / total_duration
        ),
    }


def extract_acoustic_features(
    y: np.ndarray,
    sr: int,
    original_duration: float | None = None,
    cfg: AcousticFeatureConfig | None = None,
) -> dict[str, float]:
    """
    Extract the primary handcrafted representation:
      777 spectral + 13 prosodic = 790 features.
    """
    cfg = cfg or AcousticFeatureConfig(sr=sr)

    out: dict[str, float] = {}
    out.update(_mfcc_features(y, sr, cfg))
    out.update(_lfcc_features(y, sr, cfg))
    out.update(_cqcc_features(y, sr, cfg))
    out.update(_stft_features(y, sr, cfg))
    out.update(_f0_features(y, sr, cfg))

    energy, rms = _energy_features(y, cfg)
    out.update(energy)
    out.update(_pause_features(y, sr, rms, cfg))

    if cfg.include_duration:
        if original_duration is not None and np.isfinite(original_duration):
            out["prosody_duration_sec"] = float(original_duration)
        else:
            out["prosody_duration_sec"] = float(len(y) / sr)

    return out
