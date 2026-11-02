
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

from source_attribution.config import load_config
from source_attribution.experiment import feature_columns
from source_attribution.leakage import assert_no_leakage
from source_attribution.models import build_and_train
from source_attribution.utils import file_sha256, json_dump, set_seed


def select_partition(meta, partition, speaker, mode, seed):
    part = meta.loc[meta["split"].eq(partition)]

    other_generators = part.loc[
        part["generator"].ne("voskTTS")
    ]

    vosk = part.loc[part["generator"].eq("voskTTS")]

    target = vosk.loc[vosk["speaker_id"].eq(speaker)]
    others = vosk.loc[
        vosk["speaker_id"].notna()
        & vosk["speaker_id"].ne(speaker)
    ]

    if mode == "heldout":
        selected = others
    elif mode == "control":
        n = len(others) - len(target)
        if n < 0:
            raise RuntimeError("Insufficient control samples.")

        offset = 0 if partition == "train" else 1
        sampled = others.sample(
            n=n,
            random_state=seed + speaker * 100 + offset,
        )
        selected = pd.concat([target, sampled])
    else:
        raise ValueError(mode)

    return pd.concat(
        [other_generators, selected],
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=int, default=0,
                        choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    split_path = Path("splits/closed_set.csv")
    feature_path = Path(
        "features/ruasd_closed_set_acoustic.parquet"
    )
    config_path = Path("configs/closed_set_acoustic.yaml")

    cfg = load_config(config_path)

    meta = pd.read_csv(
        split_path,
        dtype={"file_id": str, "text_hash": str},
        low_memory=False,
    )
    assert_no_leakage(meta)

    assert len(meta) == 68000
    assert meta["file_id"].is_unique
    assert meta["generator"].nunique() == 34

    meta["speaker_id"] = pd.to_numeric(
        meta["speaker"], errors="coerce"
    )

    vosk = meta.loc[meta["generator"].eq("voskTTS")]
    known = vosk["speaker_id"].dropna()

    assert set(known.unique()) == set(range(5))

    datasets = {}
    for mode in ("heldout", "control"):
        tr = select_partition(
            meta, "train", args.speaker, mode, args.seed
        )
        va = select_partition(
            meta, "validation", args.speaker, mode, args.seed
        )
        datasets[mode] = (tr, va)

        assert tr["generator"].nunique() == 34
        assert va["generator"].nunique() == 34
        assert tr["file_id"].is_unique
        assert va["file_id"].is_unique

        assert not (
            set(tr["file_id"]) & set(va["file_id"])
        )

    held_tr, held_va = datasets["heldout"]
    ctrl_tr, ctrl_va = datasets["control"]

    assert len(held_tr) == len(ctrl_tr)
    assert len(held_va) == len(ctrl_va)

    assert not held_tr.loc[
        held_tr["generator"].eq("voskTTS"),
        "speaker_id",
    ].eq(args.speaker).any()

    assert not held_va.loc[
        held_va["generator"].eq("voskTTS"),
        "speaker_id",
    ].eq(args.speaker).any()

    test = meta.loc[meta["split"].eq("test")].copy()
    target_test = (
        test["generator"].eq("voskTTS")
        & test["speaker_id"].eq(args.speaker)
    )

    assert len(test) == 13600
    assert int(target_test.sum()) > 0

    for tr, va in datasets.values():
        assert set(tr["text_hash"]).isdisjoint(
            set(test["text_hash"])
        )
        assert set(va["text_hash"]).isdisjoint(
            set(test["text_hash"])
        )

    print("LOSO AUDIT PASSED")
    print("Speaker:", args.speaker)
    print("Held-out train:", len(held_tr))
    print("Control train:", len(ctrl_tr))
    print("Held-out validation:", len(held_va))
    print("Control validation:", len(ctrl_va))
    print("Target test samples:", int(target_test.sum()))
    print("Text leakage: NONE")
    print("Unknown voskTTS speakers excluded from train/val")

    if args.dry_run:
        print("DRY RUN PASSED - NO TRAINING")
        return

    features = pd.read_parquet(feature_path)
    features["file_id"] = features["file_id"].astype(str)

    fs = cfg.get("feature_selection", {})
    cols = feature_columns(
        features,
        fs.get("include_prefixes"),
        fs.get("exclude_columns"),
        True,
    )

    assert len(cols) == 789
    assert features["file_id"].is_unique
    assert set(features["file_id"]) == set(meta["file_id"])

    X = features.set_index("file_id")[cols]
    X = X.replace([np.inf, -np.inf], np.nan)

    out = Path(
        f"results/raw/vosk_loso_speaker"
        f"{args.speaker}_seed{args.seed}"
    )

    if out.exists():
        raise RuntimeError(f"Output already exists: {out}")

    out.mkdir(parents=True)

    json_dump(
        {
            "speaker": args.speaker,
            "seed": args.seed,
            "feature_count": len(cols),
            "feature_sha256": file_sha256(feature_path),
            "split_sha256": file_sha256(split_path),
            "config_sha256": file_sha256(config_path),
            "duration_feature_excluded": True,
            "control_strategy": "equal_size_random_replacement",
        },
        out / "experiment_context.json",
    )

    mc = cfg["model"]

    for mode, (tr, va) in datasets.items():
        print("TRAINING:", mode)

        tr[["file_id", "generator", "speaker_id", "split"]].to_csv(
            out / f"{mode}_train_manifest.csv", index=False
        )
        va[["file_id", "generator", "speaker_id", "split"]].to_csv(
            out / f"{mode}_validation_manifest.csv", index=False
        )

        model = build_and_train(
            X.loc[tr["file_id"]],
            tr["generator"],
            X.loc[va["file_id"]],
            va["generator"],
            mc["type"],
            mc.get("params", {}),
            int(cfg["training"]["early_stopping_rounds"]),
            args.seed,
        )

        pred = model.predict(X.loc[test["file_id"]])
        truth = test["generator"].astype(str).to_numpy()
        mask = target_test.to_numpy()

        target_correct = int(
            np.sum(pred[mask] == "voskTTS")
        )
        target_total = int(mask.sum())

        result = {
            "mode": mode,
            "speaker": args.speaker,
            "seed": args.seed,
            "n_train": len(tr),
            "n_validation": len(va),
            "n_test": len(test),
            "target_test_n": target_total,
            "target_correct": target_correct,
            "target_recall": target_correct / target_total,
            "macro_f1": float(
                f1_score(truth, pred, average="macro")
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(truth, pred)
            ),
        }

        json_dump(result, out / f"{mode}_metrics.json")

        predictions = test[
            ["file_id", "generator", "speaker_id"]
        ].copy()
        predictions["predicted_generator"] = pred
        predictions["mode"] = mode
        predictions.to_parquet(
            out / f"{mode}_predictions.parquet",
            index=False,
        )

        joblib.dump(model, out / f"{mode}_model.joblib")

        print(
            mode,
            "TARGET_RECALL:",
            round(result["target_recall"], 4),
            "MACRO_F1:",
            round(result["macro_f1"], 4),
        )

    print("SAVED:", out)


if __name__ == "__main__":
    main()
