# Manuscript V9: result-to-artifact crosswalk

This file maps the 13 tables of the *Applied Acoustics* manuscript to the archived outputs. It enables **metric and artifact verification**, not bitwise independent training reproduction.

| Table | Topic | Data files |
|---:|---|---|
| 1 | Closed-set and shortcut controls | `results/summaries/shortcut_ablations_seed42.csv; results/summaries/same_sr_*_seed42.csv; results/raw/closed_set_acoustic_{fixed4,nopeak}_lightgbm/*/metrics.json` |
| 2 | RuASD open-set | `results/summaries/open_set_multiseed_summary.csv; results/summaries/open_set_multiseed_per_fold.csv` |
| 3 | MLAAD open-set | `results/summary/mlaad_23class_open_set_v1/overall_summary.csv; results/summary/mlaad_23class_open_set_v1/all_25_runs.csv` |
| 4 | Robustness and noise augmentation | `results/raw/robustness_statistics_v1/macro_f1_bootstrap_ci.csv; results/raw/robustness_training_noise30_noise20_n50_seed*/condition_summary.csv` |
| 5 | Cross-corpus preprocessing | `results/tables/cross_dataset_preprocessing_comparison.csv` |
| 6 | Cross-corpus partial-overlap open-set | `results/raw/cross_dataset_open_set_multiseed_5seeds/metrics_mean_sd.csv (subset=full_external)` |
| 7 | Shared-family recall | `results/raw/cross_dataset_open_set_multiseed_5seeds/family_metrics_mean_sd.csv (subset=full_external, method=sme)` |
| 8 | SHAP group importance | `results/raw/shap_ruasd_5seeds_100perclass/feature_groups_mean_sd.csv (divide nonrounded share_mean by n_features)` |
| 9 | Four-second baselines, CIs | `results/raw/fixed4_acoustic_vs_wavlm_lr/paired_statistics/model_metrics_ci.csv; results/statistical_tests/fixed4_aasist_vs_baselines_seed42/metrics.json; results/raw/tada_w2vbert2_layer4_knn21/ruasd/closed_set/metrics.json` |
| 10 | Paired model contrasts | `results/raw/fixed4_acoustic_vs_wavlm_lr/paired_statistics/paired_comparisons.csv; results/statistical_tests/fixed4_fusion_comparison_20261021/paired_comparisons.csv` |
| 11 | AASIST comparisons | `results/statistical_tests/fixed4_aasist_vs_baselines_seed42/paired_comparisons.csv` |
| 12 | Paired 80% Acoustic/WavLM open-set | `results/summaries/fixed4_lr_80pct_5seeds_summary.csv` |
| 13 | TADA comparison | `results/raw/tada_w2vbert2_layer4_knn21/{ruasd,mlaad}/open_set/summary.json; Table 2/3 summary sources` |

## Primary scripts and integrity

- Frozen identities: `splits/closed_set.csv`, `splits/open_set_folds.json`, `splits/mlaad_ru_23class_split_v1.csv`, `splits/mlaad_23class_open_set_folds_v1.json`. Check with `python scripts/release_run.py --check` and compare the SHA-256 values in `reproducibility/FROZEN_SPLIT_SHA256_VERIFIED.txt`.
- Score definitions: `src/source_attribution/metrics.py` and `scripts/fast_open_set_metrics.py`. Execute `python -m pytest -q` for tests.
- Independent verification: read `docs/reproducibility.md` and run the four commands in `reproducibility/tools/` with `--project-root .` and `--output-dir` **outside** the repository. The closed-set recomputation is in `scripts/verify_closedset_predictions.py`.
- TADA baseline implementation: `scripts/run_tada_independent_baseline.py`. Its existing external verification only recomputes core open-set indicators, not OSCR or Top-3 confidence intervals.
- Figures: SHAP Fig. 2 underlying image files `results/raw/shap_ruasd_5seeds_100perclass/global_top20_shap.png` and `shap_feature_groups.png`; Fig. 3 `shap_generator_panels_top5.png`; Fig. 1 data derive from `results/raw/robustness_*`, `results/raw/vosk_loso_speaker*_seed42/`, and `results/raw/cross_dataset_open_set_multiseed_5seeds/`. These sources do not assert full pixel-level regeneration of the final manuscript figure.
- The original audio, full extracted acoustic/SSL feature matrices and certain model weights are not bundled. Dataset access is documented in `docs/dataset_access.md`. Run reproduction and archive verification are distinct levels.

## Public availability

The permanent public repository or archived release link is pending author approval. Until then, use the review-copy archive and the exact relative file paths above. Do not claim public access before the artifact is uploaded and the authors have verified its visibility.
