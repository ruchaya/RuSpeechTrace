"""Portable TAR path mapping; does not touch experiment results."""
import tarfile
from pathlib import Path
import pytest
from source_attribution.ruasd_shards import resolve_ruasd_shard

def test_windows_legacy_path_maps_to_new_root(tmp_path):
    shard = tmp_path / "ruasd-000000.tar"
    with tarfile.open(shard, "w"):
        pass
    result = resolve_ruasd_shard(r"S:\old\dataset\ruasd-000000.tar", tmp_path)
    assert result == shard

def test_posix_and_simple_names(tmp_path):
    shard = tmp_path / "ruasd-000012.tar"
    shard.touch()
    assert resolve_ruasd_shard("/old/location/ruasd-000012.tar", tmp_path) == shard
    assert resolve_ruasd_shard("ruasd-000012.tar", tmp_path) == shard

def test_invalid_shard_name_rejected(tmp_path):
    (tmp_path / "secret.txt").touch()
    with pytest.raises(ValueError):
        resolve_ruasd_shard("../../secret.txt", tmp_path)

def test_missing_tar_yields_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="RUASD_ROOT"):
        resolve_ruasd_shard(r"S:\old\ruasd-000099.tar", tmp_path)
