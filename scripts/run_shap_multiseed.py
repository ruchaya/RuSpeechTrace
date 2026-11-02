
#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS = [42, 123, 456, 789, 2026]
SAMPLES_PER_CLASS = 100
BATCH_SIZE = 100
SELECTION_SEED = 42
TOP_FEATURES = 25

REFERENCE = Path(
    "results/raw/shortcut_acoustic_no_duration/"
    "20260928T091631Z_seed42"
)
FEATURES = Path("features/ruasd_closed_set_acoustic.parquet")
SPLITS = Path("splits/closed_set.csv")

OUT = Path("results/raw/shap_ruasd_5seeds_100perclass")


def model_dir(seed):
    if seed == 42:
        return REFERENCE
    return Path(
        f"results/raw/ruasd_34class_no_duration_seed{seed}"
    )


def get_shap_array(values, batch_size, n_features, n_classes):
    if isinstance(values, list):
        arr = np.stack(values, axis=1)
    else:
        arr = np.asarray(values)

    if arr.shape == (batch_size, n_features, n_classes):
        arr = np.transpose(arr, (0, 2, 1))

    expected = (batch_size, n_classes, n_features)
    if arr.shape != expected:
        raise RuntimeError(
            f"Unexpected SHAP shape {arr.shape}; "
            f"expected {expected}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite SHAP values detected")

    return np.abs(arr).astype(np.float64, copy=False)


def feature_group(name):
    if name.startswith("spectral_mfcc_"):
        return "MFCC"
    if name.startswith("spectral_lfcc_"):
        return "LFCC"
    if name.startswith("spectral_cqcc_"):
        return "CQCC"
    if name.startswith("spectral_stft_"):
        return "STFT"
    if name.startswith("prosody_"):
        return "Prosody"
    if name.startswith("spectral_"):
        return "Other spectral"
    return "Other"


def write_matrix(path, matrix, classes, features):
    pd.DataFrame(
        matrix,
        index=classes,
        columns=features,
    ).rename_axis("generator").to_csv(
        path, encoding="utf-8-sig"
    )


def main():
    start_time = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    print("VERIFYING FIVE FROZEN MODELS", flush=True)

    reference_bundle = joblib.load(
        REFERENCE / "model.joblib"
    )
    if reference_bundle.kind != "lightgbm":
        raise RuntimeError("Expected LightGBM bundle")

    columns = (
        REFERENCE / "feature_columns.txt"
    ).read_text(encoding="utf-8").splitlines()

    classes = [str(c) for c in reference_bundle.classes_]

    assert len(columns) == 789
    assert len(set(columns)) == 789
    assert len(classes) == 34
    assert len(set(classes)) == 34

    for seed in SEEDS:
        run = model_dir(seed)
        assert (run / "model.joblib").is_file()
        assert (
            run / "feature_columns.txt"
        ).read_text(encoding="utf-8").splitlines() == columns

    del reference_bundle

    print("LOADING FIXED TEST DATA", flush=True)

    split = pd.read_csv(SPLITS)[
        ["file_id", "generator", "split"]
    ]

    assert len(split) == 68000
    assert not split["file_id"].duplicated().any()

    test = split[split["split"] == "test"].copy()

    assert len(test) == 13600
    assert test["generator"].nunique() == 34
    assert (test["generator"].value_counts() == 400).all()
    assert set(test["generator"].astype(str)) == set(classes)

    # Deterministic, balanced sample shared by all models.
    selected = (
        test.sort_values(["generator", "file_id"])
        .groupby("generator", group_keys=False)
        .sample(
            n=SAMPLES_PER_CLASS,
            random_state=SELECTION_SEED,
        )
        .sort_values(["generator", "file_id"])
        .reset_index(drop=True)
    )

    assert len(selected) == 34 * SAMPLES_PER_CLASS

    features = pd.read_parquet(
        FEATURES, columns=["file_id"] + columns
    )
    assert len(features) == 68000
    assert not features["file_id"].duplicated().any()

    data = selected.merge(
        features,
        on="file_id",
        how="inner",
        validate="one_to_one",
    )
    assert len(data) == len(selected)

    X = data[columns].replace(
        [np.inf, -np.inf], np.nan
    )

    class_to_index = {
        name: index for index, name in enumerate(classes)
    }
    true_class_indices = np.array(
        [class_to_index[str(x)] for x in data["generator"]],
        dtype=np.int64,
    )

    chosen_ids = "\n".join(
        selected["file_id"].astype(str)
    )
    selection_hash = hashlib.sha256(
        chosen_ids.encode("utf-8")
    ).hexdigest()

    selected[["file_id", "generator"]].to_csv(
        OUT / "selected_test_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_samples = len(X)
    n_features = len(columns)
    n_classes = len(classes)

    print(f"FEATURES: {n_features}")
    print(f"GENERATORS: {n_classes}")
    print(f"TEST RECORDS PER GENERATOR: {SAMPLES_PER_CLASS}")
    print(f"RECORDS PER MODEL: {n_samples}")
    print(f"MODELS: {len(SEEDS)}")
    print(f"SELECTION SHA256: {selection_hash}")
    print(flush=True)

    global_results = []
    class_output_results = []
    class_conditional_results = []

    for seed in SEEDS:
        seed_start = perf_counter()

        print(
            f"========== SEED {seed} ==========",
            flush=True,
        )

        run = model_dir(seed)
        bundle = joblib.load(run / "model.joblib")

        if bundle.kind != "lightgbm":
            raise RuntimeError(f"Not LightGBM: {seed}")

        current_classes = [
            str(c) for c in bundle.classes_
        ]
        if current_classes != classes:
            raise RuntimeError(
                f"Class order differs for seed {seed}"
            )

        explainer = shap.TreeExplainer(
            bundle.estimator
        )

        # Each output class evaluated over all test records.
        output_sum = np.zeros(
            (n_classes, n_features),
            dtype=np.float64,
        )

        # Output class k evaluated only on true class-k records.
        conditional_sum = np.zeros_like(output_sum)
        conditional_counts = np.zeros(
            n_classes, dtype=np.int64
        )

        for batch_start in range(
            0, n_samples, BATCH_SIZE
        ):
            batch_end = min(
                batch_start + BATCH_SIZE, n_samples
            )

            xb = X.iloc[batch_start:batch_end]
            yb = true_class_indices[
                batch_start:batch_end
            ]

            values = explainer.shap_values(xb)

            abs_shap = get_shap_array(
                values,
                len(xb),
                n_features,
                n_classes,
            )

            output_sum += abs_shap.sum(axis=0)

            for class_index in np.unique(yb):
                mask = yb == class_index

                conditional_sum[class_index] += (
                    abs_shap[
                        mask, class_index, :
                    ].sum(axis=0)
                )

                conditional_counts[class_index] += (
                    mask.sum()
                )

            if (
                batch_end % 1000 == 0
                or batch_end == n_samples
            ):
                print(
                    f"  SHAP: {batch_end}/{n_samples}",
                    flush=True,
                )

            del values, abs_shap

        assert (
            conditional_counts == SAMPLES_PER_CLASS
        ).all()

        output_mean = output_sum / n_samples
        conditional_mean = (
            conditional_sum
            / conditional_counts[:, None]
        )

        global_mean = output_mean.mean(axis=0)

        global_results.append(global_mean)
        class_output_results.append(output_mean)
        class_conditional_results.append(
            conditional_mean
        )

        print(
            f"SEED {seed} DONE IN "
            f"{perf_counter() - seed_start:.1f}s",
            flush=True,
        )

        del explainer, bundle

    global_array = np.stack(global_results, axis=0)
    output_array = np.stack(
        class_output_results, axis=0
    )
    conditional_array = np.stack(
        class_conditional_results, axis=0
    )

    # Shapes:
    # global_array: (5 seeds, 789 features)
    # output_array: (5 seeds, 34 classes, 789 features)
    # conditional_array: same shape.

    global_mean = global_array.mean(axis=0)
    global_std = global_array.std(axis=0, ddof=1)

    output_mean = output_array.mean(axis=0)
    output_std = output_array.std(axis=0, ddof=1)

    conditional_mean = conditional_array.mean(axis=0)
    conditional_std = conditional_array.std(
        axis=0, ddof=1
    )

    global_table = pd.DataFrame({
        "feature": columns,
        "mean_abs_shap": global_mean,
        "sd_abs_shap": global_std,
    })

    for i, seed in enumerate(SEEDS):
        global_table[f"seed_{seed}"] = global_array[i]

    global_table = global_table.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    global_table.insert(
        0,
        "rank",
        np.arange(1, len(global_table) + 1),
    )

    global_table.to_csv(
        OUT / "global_shap_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_matrix(
        OUT / "class_output_mean_shap.csv",
        output_mean, classes, columns,
    )
    write_matrix(
        OUT / "class_output_sd_shap.csv",
        output_std, classes, columns,
    )
    write_matrix(
        OUT / "class_conditional_mean_shap.csv",
        conditional_mean, classes, columns,
    )
    write_matrix(
        OUT / "class_conditional_sd_shap.csv",
        conditional_std, classes, columns,
    )

    # Normalize each class within each seed.
    conditional_shares = (
        100.0 * conditional_array
        / np.maximum(
            conditional_array.sum(
                axis=2, keepdims=True
            ),
            1e-12,
        )
    )

    share_mean = conditional_shares.mean(axis=0)
    share_std = conditional_shares.std(
        axis=0, ddof=1
    )

    top_class_rows = []

    for class_index, generator in enumerate(classes):
        feature_order = np.argsort(
            -conditional_mean[class_index]
        )[:20]

        for rank, j in enumerate(feature_order, 1):
            top_class_rows.append({
                "generator": generator,
                "rank": rank,
                "feature": columns[j],
                "mean_abs_shap": float(
                    conditional_mean[class_index, j]
                ),
                "sd_abs_shap": float(
                    conditional_std[class_index, j]
                ),
                "share_percent_mean": float(
                    share_mean[class_index, j]
                ),
                "share_percent_sd": float(
                    share_std[class_index, j]
                ),
            })

    pd.DataFrame(top_class_rows).to_csv(
        OUT / "top20_features_by_generator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Acoustic feature groups.
    groups = sorted(set(
        feature_group(name) for name in columns
    ))

    group_records = []

    for seed_index, seed in enumerate(SEEDS):
        seed_total = max(
            global_array[seed_index].sum(),
            1e-12,
        )

        for group in groups:
            indices = [
                j for j, name in enumerate(columns)
                if feature_group(name) == group
            ]

            values = global_array[
                seed_index, indices
            ]

            group_records.append({
                "seed": seed,
                "group": group,
                "n_features": len(indices),
                "total_mean_abs_shap": float(
                    values.sum()
                ),
                "mean_per_feature": float(
                    values.mean()
                ),
                "share_percent": float(
                    100.0 * values.sum() / seed_total
                ),
            })

    group_df = pd.DataFrame(group_records)
    group_df.to_csv(
        OUT / "feature_groups_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_summary = (
        group_df.groupby("group")
        .agg(
            n_features=("n_features", "first"),
            share_mean=("share_percent", "mean"),
            share_sd=("share_percent", "std"),
            mean_per_feature=(
                "mean_per_feature", "mean"
            ),
            sd_per_feature=(
                "mean_per_feature", "std"
            ),
        )
        .reset_index()
        .sort_values(
            "share_mean", ascending=False
        )
    )

    group_summary.to_csv(
        OUT / "feature_groups_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("CREATING FIGURES...", flush=True)

    # Global top-20 feature importance.
    top20 = global_table.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(
        top20["feature"],
        top20["mean_abs_shap"],
        xerr=top20["sd_abs_shap"],
        capsize=2,
    )
    ax.set_xlabel(
        "Mean absolute SHAP across five seeds"
    )
    ax.set_title(
        "RuASD: global LightGBM feature importance"
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "global_top20_shap.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "global_top20_shap.svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Class-conditional SHAP heatmap.
    top_indices = np.argsort(
        -global_mean
    )[:TOP_FEATURES]

    heatmap = share_mean[:, top_indices]
    heatmap_features = [
        columns[j] for j in top_indices
    ]

    fig, ax = plt.subplots(figsize=(20, 14))
    im = ax.imshow(
        heatmap, aspect="auto"
    )

    ax.set_xticks(
        range(len(heatmap_features))
    )
    ax.set_xticklabels(
        heatmap_features,
        rotation=75,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)

    ax.set_xlabel("Acoustic feature")
    ax.set_ylabel("True generator")
    ax.set_title(
        "RuASD: class-conditional SHAP fingerprints "
        "(5 seeds, 100 test recordings per generator)"
    )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Share of class-conditional "
            "|SHAP| (%)"
        ),
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "shap_fingerprint_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "shap_fingerprint_heatmap.svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Feature-group contribution chart.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        group_summary["group"],
        group_summary["share_mean"],
        yerr=group_summary["share_sd"],
        capsize=3,
    )
    ax.set_ylabel(
        "Share of total absolute SHAP (%)"
    )
    ax.set_xlabel("Feature group")
    ax.set_title(
        "RuASD: feature group contributions"
    )
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(
        OUT / "shap_feature_groups.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "shap_feature_groups.svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    manifest = {
        "seeds": SEEDS,
        "selection_seed": SELECTION_SEED,
        "selection_sha256": selection_hash,
        "samples_per_class": SAMPLES_PER_CLASS,
        "samples_per_model": n_samples,
        "n_classes": n_classes,
        "n_features": n_features,
        "source_partition": "test",
        "model_paths": {
            str(seed): str(model_dir(seed))
            for seed in SEEDS
        },
        "global_shap_definition": (
            "Mean absolute SHAP over all records "
            "and all model output classes"
        ),
        "class_conditional_definition": (
            "Absolute SHAP for output class k, "
            "averaged only over recordings "
            "whose true generator is k"
        ),
        "standard_deviation": (
            "Sample SD across the five model seeds"
        ),
        "heatmap_scale": (
            "Percentage of total class-conditional "
            "absolute SHAP per class, averaged "
            "over seeds"
        ),
    }

    (
        OUT / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n========== GLOBAL TOP 20, FIVE SEEDS ==========",
        flush=True,
    )
    print(
        global_table[
            ["rank", "feature",
             "mean_abs_shap", "sd_abs_shap"]
        ].head(20).round(6).to_string(index=False),
        flush=True,
    )

    print(
        "\n========== FEATURE GROUPS ==========",
        flush=True,
    )
    print(
        group_summary.round(5).to_string(index=False),
        flush=True,
    )

    print(
        "\n========== ALL FIVE SEEDS COMPLETED ==========",
        flush=True,
    )
    print("SAMPLES PER MODEL:", n_samples)
    print("TOTAL RECORD EVALUATIONS:", n_samples * 5)
    print(
        f"TOTAL TIME: "
        f"{perf_counter() - start_time:.1f}s"
    )
    print("RESULTS:", OUT)


if __name__ == "__main__":
    main()
