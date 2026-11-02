
from pathlib import Path
import hashlib
import json
import pandas as pd

SOURCE = Path("splits/mlaad_ru_23class_split_v1.csv")
OUTPUT = Path("splits/mlaad_23class_open_set_folds_v1.json")

UNKNOWN_GROUPS = [
    [
        "dots.tts-base",
        "dots.tts-mf",
        "dots.tts-soar",
        "Edge-TTS",
        "RVC",
    ],
    [
        "MOSS-TTS-1.7B",
        "MOSS-TTS-8B",
        "Chatterbox Multilingual",
        "griffin_lim",
        "Higgs-Audio-V3",
    ],
    [
        "Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen3-TTS-12Hz-1.7B-Base",
        "Fish-S2-Pro",
        "LEMAS-TTS",
        "OpenAI TTS-1 HD",
    ],
    [
        "tts_models_multilingual_multi-dataset_xtts_v1.1",
        "tts_models_multilingual_multi-dataset_xtts_v2",
        "OuteTTS",
        "OmniVoice",
    ],
    [
        "facebook_mms-tts-rus",
        "supertonic-3",
        "tts_models_multilingual_multi-dataset_bark",
        "VoxCPM2",
    ],
]


def main():
    if OUTPUT.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing file: {OUTPUT}"
        )

    d = pd.read_csv(
        SOURCE,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    assert len(d) == 23000
    assert d["path"].is_unique

    generators = set(d["generator"])
    assert len(generators) == 23

    balance = pd.crosstab(d["generator"], d["split"])
    assert set(balance.columns) == {
        "train", "validation", "test"
    }
    assert (balance["train"] == 650).all()
    assert (balance["validation"] == 150).all()
    assert (balance["test"] == 200).all()

    for key in ["text_key", "original_file"]:
        assert d.groupby(key)["split"].nunique().max() == 1

    all_unknown = [
        name
        for group in UNKNOWN_GROUPS
        for name in group
    ]

    assert len(all_unknown) == 23
    assert len(set(all_unknown)) == 23
    assert set(all_unknown) == generators

    assert list(map(len, UNKNOWN_GROUPS)) == [
        5, 5, 5, 4, 4
    ]

    folds = {}

    for number, group in enumerate(UNKNOWN_GROUPS, 1):
        unknown = sorted(group)
        known = sorted(generators - set(unknown))

        assert not (set(known) & set(unknown))
        assert set(known) | set(unknown) == generators

        folds[str(number)] = {
            "known_generators": known,
            "unknown_generators": unknown,
            "rule": (
                "Unknown generators are absent from "
                "training, validation, and threshold tuning."
            ),
        }

        print(
            f"FOLD {number}: "
            f"KNOWN={len(known)}, UNKNOWN={len(unknown)}"
        )
        print("  UNKNOWN:", ", ".join(unknown))

    digest = hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()

    result = {
        "n_folds": 5,
        "assignment_strategy": "predefined_family_grouped_v1",
        "source_split": str(SOURCE),
        "source_split_sha256": digest,
        "generators": 23,
        "folds": folds,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("SAVED:", OUTPUT)
    print("SOURCE SPLIT SHA256:", digest)
    print("ALL GENERATORS HELD OUT EXACTLY ONCE")
    print("FOLD VALIDATION PASSED")


if __name__ == "__main__":
    main()
