# Shortcut ablation report

Frozen clean RuASD closed-set protocol, seed 42.

- Feature store SHA-256: `a21d537c5133a7e6daa852138f96ee7f8bf1491036cbf940da6451f688adaad8`
- Split SHA-256: `f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0`

## Results

- canonical_790: n_features=790, test Macro-F1=0.976635, BA=0.976691, Top-1=0.976691, Top-3=0.998897, ΔMacro-F1=0.000000
- acoustic_no_duration: n_features=789, test Macro-F1=0.976722, BA=0.976765, Top-1=0.976765, Top-3=0.999044, ΔMacro-F1=0.000088
- duration_only: n_features=1, test Macro-F1=0.073251, BA=0.107353, Top-1=0.107353, Top-3=0.228309, ΔMacro-F1=-0.903384
- sample_rate_only: n_features=1, test Macro-F1=0.106443, BA=0.176471, Top-1=0.176471, Top-3=0.352941, ΔMacro-F1=-0.870192
- metadata_only: n_features=3, test Macro-F1=0.274717, BA=0.295809, Top-1=0.295809, Top-3=0.510735, ΔMacro-F1=-0.701918

## Interpretation rule

Metadata-only and sample-rate-only results are diagnostics of corpus-level shortcut risk, not evidence of acoustic generator fingerprints.

Peak-normalization and crop/full-duration ablations require separate feature extraction and are intentionally not part of this cheap pass.
