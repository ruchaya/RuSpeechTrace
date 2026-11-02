# Final release gate (internal checklist)

Status: **NOT APPROVED FOR PUBLICATION** until all gates are complete.

- [x] Core four closed-set per-record checks: PASS in local runs.
- [x] TADA open-set core checks: 10/10 PASS in local runs.
- [x] Acoustic open-set: 220/220 PASS in local runs.
- [x] Cross-dataset: 60/60 and 4 CSV aggregates PASS in local runs.
- [x] Robustness: 6 scopes, 1341 checks PASS in local runs.
- [ ] Re-run these checks on an independently copied release tree and attach the report files.
- [ ] Validate archived `data_inventory` or publish a documented generator of inventories.
- [ ] Reconcile *final manuscript* tables/figures including SHAP with saved artifacts.
- [ ] Review all distributed datasets/metadata/transcripts, model checkpoints and third-party code licenses.
- [ ] Remove ignored `configs/local_paths.json` from published ZIP/RAR; Git ignored files can still be in archives.
- [ ] Ensure backup, smoke, cache and editor files are excluded from published ZIP/RAR/Git.
- [ ] Rebuild `reproducibility/RELEASE_FILE_INVENTORY.csv` after all modifications.
- [ ] Run `python -m pytest -q` and `python scripts/release_run.py --check` in `palkin-aasist`.
- [ ] Obtain contributor approval and only then change `RELEASE_STATUS.md`.

Frozen splits and numerical prediction/metric artifacts are not to be modified to make checks pass.
