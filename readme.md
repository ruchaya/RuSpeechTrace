# Open-Set Source Attribution of Russian Synthetic Speech

Research code, fixed data splits, configurations, selected per-record predictions, and
metric-verification tools for Russian synthetic-speech generator attribution.
This repository is a **release candidate under internal review**, not a redistribution
of RuASD or MLAAD audio and not a guarantee of bitwise retraining reproducibility.

## What is included

- `splits/`: frozen RuASD and MLAAD train/validation/test identifiers and open-set
  generator folds; SHA-256 identifiers are documented in
  `reproducibility/FROZEN_SPLIT_SHA256_VERIFIED.txt`.
- `configs/`: experiment configuration files and the ordered 789-feature schema
  (`configs/acoustic_fixed4_feature_columns.json`).
- `scripts/` and `src/`: feature extraction, model training/evaluation, and metrics.
- `results/raw/`: selected recorded metrics, predictions, and aggregate results.
  **Presence of an artifact does not imply every experiment can be retrained from
  this archive without external audio, full feature stores, or model checkpoints.**
- `reproducibility/tools/`: independent per-record prediction verification scripts.

## Local setup

Python >=3.10 is required. For the Python components (CPU-only workflows may require
different PyTorch installation steps):

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
python scripts/release_run.py --check
```

**Audio files are not included.** Obtain RuASD/MLAAD from their authorized sources
under the applicable licenses and configure:

- `RUASD_ROOT`: directory containing `ruasd-000000.tar` and other RuASD TAR shards.
- `MLAAD_ROOT`: MLAAD dataset root containing `fake/ru/`.

Either set these environment variables, or copy
`configs/local_paths.example.json` to the ignored `configs/local_paths.json` and
update the paths. **Do not commit `local_paths.json`.**

The portable wrapper can run compatible scripts with the configured roots:

```bash
python scripts/release_run.py --check
python scripts/release_run.py --script scripts/extract_mlaad_cap4.py -- --limit 3
```

Re-extraction and retraining can be expensive. To **verify already stored results**
without audio or training, use the read-only checks documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Integrity of frozen splits

**Never edit the frozen split files to fix local filesystem paths.** Some RuASD
split CSVs retain a legacy absolute `shard_path` that was present at the time
of freezing. These paths are *not* valid on other machines. Code should resolve
the TAR basename under `RUASD_ROOT` without changing split bytes or SHA-256.
See [`docs/dataset_access.md`](docs/dataset_access.md).

## Manuscript traceability

- [V9 article table-by-table source map](docs/manuscript_artifact_map.md)
- [Definitions of the 789 fixed-four-second acoustic features](docs/feature_definitions.md)

These documents distinguish saved-result verification from full re-extraction or
retraining. The manuscript and its author confirmations remain subject to review.

## Scope and limitations

- The main closed-set task is generator attribution; open-set evaluations hold out
  generators at the fold level. Some tables also aggregate generator families.
- All ID-oriented scores use **higher score = more likely known**. See
  [`docs/metrics.md`](docs/metrics.md) and [`docs/protocol.md`](docs/protocol.md).
- External components such as the original AASIST implementation and datasets
  are subject to their own licenses. The project does not include original audio.
  The publication status of any distributed model weights must be reviewed before release.
- Verification of predictions against saved metrics is not proof that a complete
  training run can be reproduced on another machine.

## Research and contact

Publication bibliographic details, dataset provenance/permissions, funding,
contributor affiliations, and a permanent artifact URL will be added after
coauthor review. The repository is not yet cleared for public distribution.
