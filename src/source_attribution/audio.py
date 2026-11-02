from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import librosa
import numpy as np
import soundfile as sf


AudioSource = Union[str, Path, BinaryIO]


def _read_audio(source: AudioSource, mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Read an audio path or a seekable binary file object.

    This intentionally supports tarfile.ExFileObject / BytesIO so RuASD WAV files
    can be processed directly from TAR shards without unpacking them.
    """
    y, sr = sf.read(source, always_2d=False)
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 2 and mono:
        y = y.mean(axis=1)

    if y.ndim != 1 or y.size == 0:
        raise ValueError(f"Expected non-empty mono-compatible audio, got shape={y.shape}")

    return y, int(sr)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    y = librosa.resample(
        np.asarray(y, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    )
    return np.asarray(y, dtype=np.float32)


def trim_or_pad_audio(
    y: np.ndarray,
    sr: int,
    max_duration_sec: float | None = 10.0,
    min_audio_sec: float = 0.10,
) -> np.ndarray:
    """
    Match the previous-study preprocessing:
    - truncate from the start to max_duration_sec;
    - right-pad very short clips for stable framing.
    """
    y = np.asarray(y, dtype=np.float32)

    if max_duration_sec is not None:
        maximum = max(1, int(round(sr * float(max_duration_sec))))
        y = y[:maximum]

    minimum = max(2048, int(round(sr * float(min_audio_sec))))
    if len(y) < minimum:
        y = np.pad(y, (0, minimum - len(y)), mode="constant")

    return np.asarray(y, dtype=np.float32)


def normalize_audio(
    y: np.ndarray,
    peak_normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Remove DC bias in all conditions and optionally apply peak normalization.

    Keeping DC removal constant lets the peak-normalization ablation isolate the
    peak scaling itself rather than changing two preprocessing operations at once.
    """
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if y.size == 0:
        return y

    y = y - float(np.mean(y))

    if peak_normalize:
        peak = float(np.max(np.abs(y)))
        if peak > eps:
            y = y / peak

    return np.asarray(y, dtype=np.float32)


def preprocess_audio(
    source: AudioSource,
    target_sr: int = 16000,
    mono: bool = True,
    peak_normalize: bool = True,
    max_duration_sec: float | None = 10.0,
    min_audio_sec: float = 0.10,
) -> tuple[np.ndarray, int, float]:
    """
    Load -> mono -> resample -> record original duration -> trim/pad ->
    DC removal -> optional peak normalization.

    Returns
    -------
    y_processed, target_sr, original_duration_sec
    """
    y, native_sr = _read_audio(source, mono=mono)
    y = _resample(y, native_sr, int(target_sr))
    original_duration = float(len(y) / int(target_sr))

    y = trim_or_pad_audio(
        y,
        int(target_sr),
        max_duration_sec=max_duration_sec,
        min_audio_sec=min_audio_sec,
    )
    y = normalize_audio(y, peak_normalize=peak_normalize)

    return y, int(target_sr), original_duration


def load_audio(
    path: AudioSource,
    target_sr: int = 16000,
    mono: bool = True,
    peak_normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Backward-compatible loader used by existing project code.

    Unlike preprocess_audio(), this does not truncate/pad audio.
    """
    y, native_sr = _read_audio(path, mono=mono)
    y = _resample(y, native_sr, int(target_sr))
    y = normalize_audio(y, peak_normalize=peak_normalize)
    return y, int(target_sr)


def fixed_length(
    y: np.ndarray,
    sr: int,
    seconds: float,
    mode: str = "center",
) -> np.ndarray:
    n = int(round(sr * seconds))
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    if len(y) == n:
        return y
    if mode == "start":
        return y[:n]
    start = max(0, (len(y) - n) // 2)
    return y[start:start + n]
