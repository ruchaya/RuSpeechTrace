
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import confusion_matrix
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics


SEEDS = [42, 123, 456, 789, 2026]

CORE = {
    "Bark": (
        "bark_small_tts",
        "tts_models_multilingual_multi-dataset_bark",
    ),
    "XTTS-v2": (
        "coquiTTS",
        "tts_models_multilingual_multi-dataset_xtts_v2",
    ),
    "MMS-TTS": (
        "mms_tts_rus",
        "facebook_mms-tts-rus",
    ),
}

FISH = (
    "fishsTTS",
    "Fish-S2-Pro",
)

F5_MLAAD = "LEMAS-TTS"

SCENARIOS = {
    "3base": dict(CORE),

    "4fish": {
        **CORE,
        "FishSpeech": FISH,
    },

    "5f5_sf": {
        **CORE,
        "FishSpeech": FISH,
        "F5TTS_candidate": ("sf5TTS", F5_MLAAD),
    },

    "5f5_misha": {
        **CORE,
        "FishSpeech": FISH,
        "F5TTS_candidate": ("mishaf5TTS", F5_MLAAD),
    },

    "3xtts_ipa": {
        **CORE,
        "XTTS-v2": (
            "xtts_ru_ipa",
            "tts_models_multilingual_multi-dataset_xtts_v2",
        ),
    },

    "3xtts_rux": {
        **CORE,
        "XTTS-v2": (
            "ruxTTS",
            "tts_models_multilingual_multi-dataset_xtts_v2",
        ),
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_features(split_path, feature_path, key, mapping):
    meta_path = feature_path.with_suffix(".meta.json")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["split_sha256"] == sha256(split_path), (
        f"Split hash mismatch: {split_path}"
    )

    split = pd.read_csv(
        split_path,
        usecols=[key, "generator", "split"],
        low_memory=False,
    )
    assert split[key].is_unique
    assert split["split"].isin(
        ["train", "validation", "test"]
    ).all()

    split = split.loc[
        split["generator"].isin(mapping)
    ].copy()

    split["family"] = split["generator"].map(mapping)

    features = pd.read_parquet(feature_path)
    assert features[key].is_unique

    features = features.drop(
        columns=["generator", "split", "family"],
        errors="ignore",
    )

    data = split.merge(
        features,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    assert len(data) == len(split), "Missing feature rows"
    assert data["family"].notna().all()

    return data, metadata


def check_counts(data, families, quotas, dataset):
    tab = pd.crosstab(
        data["family"],
        data["split"],
    ).reindex(
        index=families,
        columns=["train", "validation", "test"],
        fill_value=0,
    )

    expected = np.tile(
        np.asarray(quotas),
        (len(families), 1),
    )

    if not np.array_equal(tab.to_numpy(), expected):
        raise RuntimeError(
            f"Unexpected {dataset} counts:\n{tab}"
        )

    print(f"\n{dataset} COUNTS")
    print(tab.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    scenario = args.scenario
    seed = args.seed
    pairs = SCENARIOS[scenario]

    output = Path(
        f"results/raw/extended_family_"
        f"{scenario}_seed{seed}"
    )

    if output.exists():
        raise FileExistsError(
            f"Existing result directory: {output}"
        )

    r_split = Path("splits/closed_set.csv")
    m_split = Path(
        "splits/mlaad_ru_23class_split_v1.csv"
    )

    r_feat = Path(
        "features/ruasd_closed_set_acoustic.parquet"
    )
    m_feat = Path(
        "features/mlaad_23class_acoustic.parquet"
    )

    r_map = {
        r: family
        for family, (r, m) in pairs.items()
    }
    m_map = {
        m: family
        for family, (r, m) in pairs.items()
    }

    assert len(r_map) == len(pairs)
    assert len(m_map) == len(pairs)

    families = sorted(pairs)

    r, r_meta = load_features(
        r_split, r_feat, "file_id", r_map
    )
    m, m_meta = load_features(
        m_split, m_feat, "path", m_map
    )

    assert (
        r_meta["feature_signature"]
        == m_meta["feature_signature"]
    ), "Feature extraction definitions differ"

    r_cols = [
        c for c in r.columns
        if c.startswith(("spectral_", "prosody_"))
        and c != "prosody_duration_sec"
    ]
    m_cols = [
        c for c in m.columns
        if c.startswith(("spectral_", "prosody_"))
        and c != "prosody_duration_sec"
    ]

    assert r_cols == m_cols
    assert len(r_cols) == 789

    cols = r_cols

    assert not np.isinf(
        r[cols].to_numpy(dtype=float)
    ).any()

    assert not np.isinf(
        m[cols].to_numpy(dtype=float)
    ).any()

    check_counts(
        r, families, [1300, 300, 400], "RUASD"
    )
    check_counts(
        m, families, [650, 150, 200], "MLAAD"
    )

    print("\nSCENARIO:", scenario)
    print("SEED:", seed)
    print("FAMILIES:", families)
    print("FEATURES:", len(cols))
    print("MATCHED GENERATORS:", pairs)

    if args.dry_run:
        print("\nDRY RUN PASSED")
        return

    cfg_path = Path("configs/cross_dataset.yaml")
    cfg = yaml.safe_load(
        cfg_path.read_text(encoding="utf-8")
    )["model"]

    results = {}
    predictions = []

    for source, train_data in [
        ("ruasd", r),
        ("mlaad", m),
    ]:
        tr = train_data.loc[
            train_data["split"] == "train"
        ]
        va = train_data.loc[
            train_data["split"] == "validation"
        ]

        print(
            f"\nTRAINING {source.upper()}",
            flush=True,
        )

        model = build_and_train(
            tr[cols],
            tr["family"],
            va[cols],
            va["family"],
            cfg["type"],
            cfg["params"],
            80,
            seed,
        )

        assert sorted(model.classes_.tolist()) == families

        for target, test_data in [
            ("ruasd", r),
            ("mlaad", m),
        ]:
            te = test_data.loc[
                test_data["split"] == "test"
            ]

            actual = te["family"].astype(str).to_numpy()
            predicted = model.predict(te[cols])
            proba = model.predict_proba(te[cols])

            tag = source + "_to_" + target

            metrics = closed_set_metrics(
                actual,
                predicted,
                proba,
                model.classes_,
            )

            matrix = confusion_matrix(
                actual,
                predicted,
                labels=families,
            )

            results[tag] = {
                "train_n": len(tr),
                "validation_n": len(va),
                "test_n": len(te),
                "metrics": metrics,
                "confusion_matrix": matrix.tolist(),
            }

            sample_key = (
                "file_id"
                if target == "ruasd"
                else "path"
            )

            frame = pd.DataFrame({
                "scenario": scenario,
                "seed": seed,
                "experiment": tag,
                "sample_key": te[sample_key].to_numpy(),
                "true_family": actual,
                "predicted_family": predicted,
            })

            for j, label in enumerate(model.classes_):
                frame["proba_" + str(label)] = proba[:, j]

            predictions.append(frame)

            print(
                tag,
                "N:", len(te),
                "MACRO_F1:",
                round(metrics["macro_f1"], 5),
                "BAL_ACC:",
                round(metrics["balanced_accuracy"], 5),
                flush=True,
            )

    # Only create output after all calculations succeed.
    output.mkdir(parents=True, exist_ok=False)

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_parquet(
        output / "predictions.parquet",
        index=False,
    )

    report = {
        "scenario": scenario,
        "seed": seed,
        "families": families,
        "mapping": {
            k: list(v) for k, v in pairs.items()
        },
        "mapping_level": (
            "family-level; Fish cross-version; "
            "F5 exploratory when present; "
            "XTTS alternatives are checkpoint variants"
        ),
        "n_features": len(cols),
        "feature_signature": r_meta["feature_signature"],
        "ruasd_split_sha256": sha256(r_split),
        "mlaad_split_sha256": sha256(m_split),
        "ruasd_features_sha256": sha256(r_feat),
        "mlaad_features_sha256": sha256(m_feat),
        "config_sha256": sha256(cfg_path),
        "script_sha256": sha256(Path(__file__)),
        "results": results,
    }

    (output / "metrics.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSAVED:", output)
    print("EXTENDED FAMILY TRANSFER PASSED")


if __name__ == "__main__":
    main()
