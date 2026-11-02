# Dataset access, identifiers, and local path resolution

Audio from RuASD and MLAAD is not bundled. It must be acquired from authorized
sources under the original conditions of use. Dataset versions, licenses and
provenance must be recorded by the releasing authors before publication.

## RuASD

Set `RUASD_ROOT` to the directory containing the tar archives (e.g.
`ruasd-000000.tar`). **Keep** the original `splits/closed_set.csv` untouched:
some `shard_path` fields contain a Windows path from the original study.
Use its `shard_name` or the TAR basename resolved under `RUASD_ROOT`, not the
original drive prefix. `source_attribution.ruasd_shards.resolve_ruasd_shard`
provides an explicitly validated lookup for scripts using legacy `shard_path`.

This operation changes only where audio is opened; it does not change the
chosen `file_id`, generator, split, TAR member or frozen SHA-256. It will
fail loudly if the expected local TAR is absent.

## MLAAD

Set `MLAAD_ROOT` to the MLAAD root containing `fake/ru/`; exact download,
version and license information must come from the source dataset provider.
The frozen 23-class subset identifiers are in
`splits/mlaad_ru_23class_split_v1.csv` and the generator-level folds in
`splits/mlaad_23class_open_set_folds_v1.json`.

## Important limitations

- Do not publish local `configs/local_paths.json` or raw audio caches.
- Published CSVs may contain dataset metadata and transcripts; assess licensing,
  confidentiality, or personal-data obligations before release.
- Generated features, raw audio and selected model checkpoints were not all
  included in the supplied release snapshot. The availability of identifiers
  does not imply external datasets can be redistributed.
