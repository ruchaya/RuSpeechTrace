# Reproducibility: local setup

This repository does not distribute the original audio datasets.

1. Install the documented Python dependencies.
2. Copy configs/local_paths.example.json to configs/local_paths.json.
3. Set the local RuASD and MLAAD dataset paths.
4. From the repository root, run:

    python scripts/release_run.py --check

Example experiment invocation:

    python scripts/release_run.py --script scripts/extract_mlaad_cap4.py -- --limit 3

Alternative: override paths without editing a config file:

    python scripts/release_run.py --mlaad-root "D:\Datasets\MLAAD_fake_ru" --script scripts/extract_mlaad_cap4.py -- --limit 3

Important:
- Run scripts from the repository root via the wrapper.
- The wrapper exports RUASD_ROOT and MLAAD_ROOT.
- The target script must support these variables to use their values.
- The wrapper does not install dependencies or download datasets.
- This setup is under verification; full paper reproduction is not yet certified.
