"""Resolve frozen RuASD TAR shard references on another computer.

The original frozen manifests embed the author's Windows shard_path values.
Those values are kept byte-for-byte for their published SHA-256 hashes; only
the TAR basename is used when opening recordings.  Set RUASD_ROOT to the
directory containing ruasd-000000.tar etc. (release_run.py exports it).
"""
from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath


_SHARD_NAME = re.compile(r"ruasd-\d{6}\.tar\Z", re.IGNORECASE)


def resolve_ruasd_shard(legacy_shard_path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> Path:
    if not legacy_shard_path:
        raise ValueError("Missing RuASD shard path in frozen manifest")
    # PureWindowsPath handles the original S:\... path even on Linux.
    name = PureWindowsPath(str(legacy_shard_path)).name
    if not _SHARD_NAME.fullmatch(name):
        raise ValueError(f"Unexpected RuASD TAR shard name: {name!r}")
    dataset_root = Path(root if root is not None else os.environ.get("RUASD_ROOT", "data/raw/RuASD")).expanduser()
    resolved = dataset_root / name
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing RuASD archive {resolved}. Set RUASD_ROOT to the folder containing {name}."
        )
    return resolved
