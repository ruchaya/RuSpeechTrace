
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


SOURCE = Path("splits/mlaad_ru_manifest.csv")
OUTPUT = Path("splits/mlaad_ru_23class_split_v1.csv")
SEED = 2026
PARTS = ["train", "validation", "test"]
QUOTAS = [650, 150, 200]


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"Output already exists: {OUTPUT}")

    d = pd.read_csv(
        SOURCE, low_memory=False, keep_default_na=False
    )

    assert len(d) == 23000
    assert d["path"].is_unique
    assert d["generator"].nunique() == 23
    assert d.groupby("generator").size().eq(1000).all()

    for key in ["text_key", "original_file"]:
        assert d[key].astype(str).str.strip().ne("").all(), key

    # Connected components across text and original audio.
    text_ids, text_values = pd.factorize(d["text_key"])
    audio_ids, audio_values = pd.factorize(d["original_file"])

    n_text = len(text_values)
    n_audio = len(audio_values)

    left = text_ids
    right = n_text + audio_ids

    rows = np.concatenate([left, right])
    cols = np.concatenate([right, left])

    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_text + n_audio, n_text + n_audio),
    ).tocsr()

    _, labels = connected_components(
        graph, directed=False
    )

    d["component"] = labels[text_ids]

    classes = sorted(d["generator"].unique())
    components = pd.crosstab(
        d["component"], d["generator"]
    ).reindex(columns=classes, fill_value=0)

    matrix = components.to_numpy(dtype=np.int32)
    n_components, n_classes = matrix.shape

    target = np.tile(
        np.array(QUOTAS, dtype=np.int32)[:, None],
        (1, n_classes),
    )

    sizes = matrix.sum(axis=1)
    best = None

    # Multiple reproducible starts.
    for trial in range(8):
        rng = np.random.default_rng(SEED + trial)
        order = np.lexsort(
            (rng.random(n_components), -sizes)
        )

        assignment = np.full(n_components, -1, dtype=int)
        current = np.zeros_like(target)

        for i in order:
            v = matrix[i]

            feasible = [
                p for p in range(3)
                if np.all(current[p] + v <= target[p])
            ]

            options = feasible if feasible else list(range(3))

            scores = [
                (
                    float(
                        np.dot(
                            2 * (current[p] - target[p]) + v,
                            v,
                        )
                    ),
                    rng.random(),
                    p,
                )
                for p in options
            ]

            p = min(scores)[2]
            assignment[i] = p
            current[p] += v

        # Improve balance by moving whole components.
        for _ in range(3):
            improvements = 0

            for i in rng.permutation(n_components):
                v = matrix[i]
                old = assignment[i]

                for new in rng.permutation(3):
                    if new == old:
                        continue

                    before = (
                        np.square(current[old] - target[old]).sum()
                        + np.square(current[new] - target[new]).sum()
                    )

                    after = (
                        np.square(current[old] - v - target[old]).sum()
                        + np.square(current[new] + v - target[new]).sum()
                    )

                    if after < before:
                        current[old] -= v
                        current[new] += v
                        assignment[i] = new
                        improvements += 1
                        break

            if improvements == 0:
                break

        loss = int(np.square(current - target).sum())

        if best is None or loss < best[0]:
            best = (
                loss,
                assignment.copy(),
                current.copy(),
                trial,
            )

        if loss == 0:
            break

    loss, assignment, counts, trial = best

    mapping = dict(
        zip(
            components.index,
            [PARTS[p] for p in assignment],
        )
    )

    d["split"] = d["component"].map(mapping)

    assert d["split"].notna().all()
    assert d["path"].is_unique

    # Strict leakage checks.
    for key in ["component", "text_key", "original_file"]:
        overlap = d.groupby(key)["split"].nunique().max()
        assert overlap == 1, f"LEAKAGE: {key}"

    balance = pd.crosstab(
        d["generator"], d["split"]
    ).reindex(index=classes, columns=PARTS, fill_value=0)

    assert (balance.to_numpy() > 0).all()
    assert balance.to_numpy().sum() == len(d)

    max_deviation = int(
        np.abs(counts - target).max()
    )

    print("COMPONENTS:", n_components)
    print("BEST TRIAL:", trial)
    print("BALANCE LOSS:", loss)
    print("MAX DEVIATION FROM QUOTA:", max_deviation)
    print("CLASS BALANCE:")
    print(balance.to_string())
    print("TOTAL COUNTS:")
    print(balance.sum().to_string())
    print("TEXT AND ORIGINAL-FILE LEAKAGE: NONE")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

    metadata = {
        "seed": SEED,
        "trial": trial,
        "source": str(SOURCE),
        "rows": len(d),
        "generators": n_classes,
        "components": n_components,
        "target_per_generator": dict(zip(PARTS, QUOTAS)),
        "balance_loss": loss,
        "max_deviation": max_deviation,
        "leakage_keys": ["text_key", "original_file"],
        "sha256": digest,
    }

    OUTPUT.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("SAVED:", OUTPUT)
    print("SHA256:", digest)
    print("SPLIT VALIDATION PASSED")


if __name__ == "__main__":
    main()
