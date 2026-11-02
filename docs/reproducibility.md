# Reproducibility and verification guide

## Levels of reproduction

1. **Artifact integrity:** verify frozen splits and syntax of released Python code.
2. **Metric reproduction:** independently recompute metrics from saved predictions,
   without touching audio, training, or the frozen manifests.
3. **Feature reproduction:** obtain the datasets lawfully, set local roots, and
   re-extract features with exactly the stated audio preprocessing parameters.
4. **Training reproduction:** rerun configurations with corresponding dependencies,
   audio, features, and model weights; GPU/runtime numerical differences may occur.

This release has been numerically checked at level 2 for the listed scopes; **the
complete release has not been certified at levels 3–4.**

## Frozen split references

- `splits/closed_set.csv` (RuASD):
  `f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0`
- `splits/open_set_folds.json` (RuASD):
  `cac3bf00c0e8f77a1a2a37a4b3b005c1ce65ba3dbaf364d3b028dd5dde20171d`
- `splits/mlaad_ru_23class_split_v1.csv` (MLAAD):
  `f4b66a0c8cbce8e7629d85838eb371d5f7381a9d2eafb63917f8ccf1fe5fbd66`
- `splits/mlaad_23class_open_set_folds_v1.json` (MLAAD):
  `eac341ab00a413ff1360ed4324316ff7822a056c4d02717e62b6c9691a66cde0`

Run `python scripts/release_run.py --check` from the project root. The original
frozen CSVs include legacy Windows paths; their hashes must remain unchanged.

## Python environment and metrics definitions

Install the dependencies from `pyproject.toml`, record Python and package versions,
and run `python -m pytest -q`. See `docs/metrics.md` and
`src/source_attribution/metrics.py`: AUROC uses known/ID as the positive class
with an ID-oriented score; AUPR-OOD uses the opposite orientation; FPR95 and
OSCR follow the source implementation. For OSCR the faster equivalent is
`scripts/fast_open_set_metrics.py`; compare with `scripts/test_fast_open_set_metrics.py`.

## Independent prediction-level verification

Run from a command prompt in the repository root. **Set the report directory
outside the repository**, e.g. `../source-attribution-verification` (absolute
paths also work). The command does not retrain or change source artifacts.

```bash
# Existing four main closed-set results (writes to reproducibility/verification)
python scripts/verify_closedset_predictions.py

# Ten TADA open-set folds (AUROC/threshold metrics; AUPR and FPR95 diagnostics)
python reproducibility/tools/verify_tada_openset_external_v2.py --project-root . --output-dir ../source-attribution-verification/tada

# RuASD/MLAAD acoustic open-set, 220 fold-method groups
python reproducibility/tools/verify_acoustic_openset_external_v3.py --project-root . --output-dir ../source-attribution-verification/acoustic

# Cross-dataset transfer, 60 groups and 4 aggregate CSV tables
python reproducibility/tools/verify_cross_dataset_external.py --project-root . --output-dir ../source-attribution-verification/cross_dataset

# Robustness and stratified 2000-repeat bootstrap results
python reproducibility/tools/verify_robustness_external.py --project-root . --output-dir ../source-attribution-verification/robustness
```

**Verification evidence from the authors' local run on 2026-11-01** (reports
retained separately until reviewed for inclusion):

| Scope | Reported result | What it establishes |
| --- | --- | --- |
| Closed-set TADA RuASD/MLAAD, AASIST seed42, Fusion | 4/4 PASS | Accuracy/Macro-F1 consistent with saved predictions and paper rounding |
| TADA open-set RuASD/MLAAD | 10/10 core PASS | AUROC, known acceptance, unknown rejection; AP/FPR95 candidate values also match |
| Acoustic open-set RuASD/MLAAD | 220/220 PASS | Eight saved metrics, including OSCR, per fold and method |
| Cross-dataset transfer | 60/60 PASS, 4/4 aggregate CSV PASS | Saved per-record and aggregate metrics agree |
| Robustness | 6/6 scopes, 1341/1341 checks PASS | Point metrics and saved bootstrap results agree |

**Scope boundaries:** the TADA verifier does not yet independently recompute
OSCR or Macro-F1 with `unknown`; the closed-set verifier does not reconstruct
Top-3/CI. SHAP attributions and every other paper result are not covered by
these particular scripts. Do not use these PASS counts to imply full model
retraining reproducibility or verified access rights.

## How to reproduce features and train

Obtain audio under original licenses, set `RUASD_ROOT`/`MLAAD_ROOT`, and follow
configuration-specific scripts. The 789 fixed4 feature names are frozen in
`configs/acoustic_fixed4_feature_columns.json`; their ordering provenance is
in `reproducibility/acoustic_fixed4_feature_schema_provenance.json`.
Full feature parquet stores and training caches are **not distributed** with
this archive. See `reproducibility/LOCAL_SETUP.md`, `docs/protocol.md`, and
`docs/dataset_access.md`. Scripts that require the absent feature stores or
model checkpoints cannot run end-to-end from the selected artifact archive.

## Permissions and final paper linkage

Obtain clearance for model weights, third-party code and dataset-derived
transcript/metadata before publishing. Link every reported table/figure to
its configuration, predictions, metric script and result file after the final
manuscript has been compared with this release. Obtain coauthor approval.
