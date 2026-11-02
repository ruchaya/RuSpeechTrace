# Fixed-4-second acoustic feature definitions

**Scope.** This document describes the handcrafted **789-feature fixed4, no-duration representation** implemented in `src/source_attribution/acoustic_features.py`, configured in `configs/closed_set_acoustic_fixed4.yaml`, and ordered in `configs/acoustic_fixed4_feature_columns.json`. This is **not** a specification for all other acoustic/SSL representations or every ablation in the study.

## Audio preprocessing (fixed4 configuration)

`src/source_attribution/audio.py` and `scripts/extract_acoustic_features_fixed4.py` implement the sequence: load the audio → convert multichannel audio to mono by channel averaging → resample with librosa (`kaiser_best`) to **16,000 Hz** → preserve original duration as metadata → truncate from the beginning to at most **4.0 s** → right-pad short audio to the minimum 0.10 s or 2,048 samples (whichever is larger) → remove DC offset → optionally normalize to peak magnitude 1 (enabled) → right-pad or crop from the beginning to an **exact 4.0 s** fixed-length waveform. Numerical audio is processed as float32; feature summary computations use float64 within `_summarize`, and the extracted feature stores save features in float32.

The frozen feature configuration uses `n_fft=512`, `win_length=400`, `hop_length=160` for the relevant STFT descriptors (frame/hop durations at 16 kHz: 25 ms window, 10 ms hop). See the YAML for the complete parameter record and each extractor for details.

## Feature groups and counts

| Prefix / feature family | Method used by source code | Columns |
| --- | --- | ---: |
| `spectral_mfcc_*` | 20 `librosa.feature.mfcc` coefficients, seven summary statistics each | 140 |
| `spectral_mfcc_delta_*` | First temporal difference of the MFCC sequence; adaptive delta width up to 9, `mode="nearest"` | 140 |
| `spectral_mfcc_delta2_*` | Second temporal difference of the MFCC sequence; same adaptive width | 140 |
| `spectral_lfcc_*` | 20 LFCC coefficients: 40-band linear triangular filterbank on power STFT → log energies → orthonormal type-II DCT | 140 |
| `spectral_cqcc_*` | 20 CQT-derived coefficients: CQT magnitude → log → orthonormal type-II DCT; **practical approximation, not reference-standard CQCC** | 140 |
| `spectral_stft_centroid_*` | Spectral centroid (1 descriptor × 7 statistics) | 7 |
| `spectral_stft_bandwidth_*` | Spectral bandwidth (1 × 7) | 7 |
| `spectral_stft_rolloff_*` | Spectral roll-off (1 × 7) | 7 |
| `spectral_stft_flatness_*` | Spectral flatness (1 × 7) | 7 |
| `spectral_stft_contrast_*` | Seven spectral-contrast bands, seven statistics each | 49 |
| `prosody_f0_{mean,std,min,max}` | Fundamental-frequency summaries via `librosa.pyin` (60–500 Hz); NaN if no finite f0 estimate | 4 |
| `prosody_voiced_ratio` | Fraction of frames estimated voiced by pYIN | 1 |
| `prosody_rms_{mean,std}` | RMS frame-energy summary | 2 |
| `prosody_zcr_{mean,std}` | Zero-crossing-rate summary | 2 |
| `prosody_pause_{count,mean_sec,ratio}` | Duration/count/fraction of low-energy runs; see below | 3 |
| **Total** | **777 spectral + 12 prosodic, duration excluded** | **789** |

For every framewise spectral coefficient, the seven summary statistics are `mean`, `std`, `min`, `max`, `median`, `skew`, and `kurtosis` in that order. `std` is NumPy population standard deviation; `skew` uses SciPy `stats.skew(..., bias=False)` and `kurtosis` uses `stats.kurtosis(..., fisher=True, bias=False)`. Invalid/insufficient frame samples may produce NaNs rather than arbitrary replacement values.

**CQCC-approximation parameters:** `cqcc_hop_length=256`, `cqcc_n_bins=72`, `cqcc_bins_per_octave=12`. **Pitch parameters:** `f0_frame_length=1024`, `f0_min_hz=60`, `f0_max_hz=500`. **Energy parameters:** `energy_frame_length=400`, `hop_length=160`.

**Pause definition:** The low-energy threshold is `max(percentile_25(RMS) × 1.5, 0.0001)`. Consecutive frames with `RMS < threshold` form candidate pauses; only runs lasting at least 0.10 s are counted. Pause ratio is total duration of retained pauses divided by waveform duration.

### Why 789, rather than 790?

`src/source_attribution/acoustic_features.py` can extract one extra field, `prosody_duration_sec`, when `include_duration=True`. The fixed4 configuration sets **`include_duration: false`**, so that field is deliberately omitted. The ordered list of the exact **789** model inputs lives in `configs/acoustic_fixed4_feature_columns.json`; **the order is part of the published representation** and must not be alphabetically regenerated or inferred from a Parquet column order.

## Verification and provenance

- Configuration: `configs/closed_set_acoustic_fixed4.yaml`
- Ordered 789 names: `configs/acoustic_fixed4_feature_columns.json`
- Schema provenance: `reproducibility/acoustic_fixed4_feature_schema_provenance.json`
- Feature source: `src/source_attribution/acoustic_features.py`
- Audio preprocessing: `src/source_attribution/audio.py`
- Extraction entrypoint: `scripts/extract_acoustic_features_fixed4.py`
- Feature quality control: `features/ruasd_closed_set_acoustic_fixed4_feature_qc.csv` and corresponding `.meta.json`

**Publication note:** In addition to this fixed4 representation, experiments using legacy/nopeak/SSL/WavLM features need their own configuration-specific references; do not extrapolate the above settings to those experiments. Full precomputed feature Parquet stores are not included in the selected export.

## Distinguishing canonical 10-second and fixed four-second variants

The canonical closed-set model uses `configs/closed_set_acoustic.yaml`: an at-most-10-second waveform and **790** features, including `prosody_duration_sec`. Removing this duration input yields the **789**-feature representation used in the principal open-set, robustness and cross-corpus analyses. For the four-second acoustic extraction, `configs/closed_set_acoustic_fixed4.yaml` instead sets both `max_duration_sec: 4.0` and `fixed_length_sec: 4.0`, so recordings shorter than four seconds **are right-padded to an exact four-second feature-extraction window**. The WavLM 4 s extraction also pads short inputs. Do not describe the fixed4 extraction as using an unpadded variable-length 4-second maximum. The ten-second canonical settings, the no-peak variant, and the TADA preprocessing are distinct and have their own run metadata.
