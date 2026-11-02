
#!/usr/bin/env python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN = Path(
    "results/raw/shortcut_acoustic_no_duration/"
    "20260928T091631Z_seed42"
)
FEATURES = Path("features/ruasd_closed_set_acoustic.parquet")
SPLITS = Path("splits/closed_set.csv")
OUT = RUN / "shap_stratified_20perclass"

N_PER_CLASS = 20
SEED = 42
TOP_N = 25


def main():
    print("LOADING FROZEN MODEL", flush=True)
    bundle = joblib.load(RUN / "model.joblib")
    assert bundle.kind == "lightgbm"

    columns = (
        RUN / "feature_columns.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert len(columns) == 789
    assert len(set(columns)) == 789

    classes = [str(c) for c in bundle.classes_]
    assert len(classes) == 34

    print("LOADING FROZEN TEST DATA", flush=True)
    features = pd.read_parquet(
        FEATURES,
        columns=["file_id"] + columns,
    )
    splits = pd.read_csv(SPLITS)[
        ["file_id", "generator", "split"]
    ]

    df = splits.merge(
        features,
        on="file_id",
        how="inner",
        validate="one_to_one",
    )
    assert len(df) == 68000

    test = df[df["split"] == "test"].copy()
    assert len(test) == 13600
    assert test["generator"].nunique() == 34

    counts = test["generator"].value_counts()
    assert (counts == 400).all()

    print("STRATIFIED SAMPLING", flush=True)
    selected = (
        test.groupby("generator", group_keys=False)
        .sample(n=N_PER_CLASS, random_state=SEED)
        .sort_values(["generator", "file_id"])
        .reset_index(drop=True)
    )

    assert len(selected) == 34 * N_PER_CLASS
    assert (
        selected["generator"].value_counts() == N_PER_CLASS
    ).all()

    X = selected[columns].replace(
        [np.inf, -np.inf], np.nan
    )

    print(
        "SHAP INPUT:",
        X.shape,
        "CLASSES:",
        len(classes),
        flush=True,
    )

    print("CALCULATING SHAP VALUES...", flush=True)
    explainer = shap.TreeExplainer(bundle.estimator)
    values = explainer.shap_values(X)

    if isinstance(values, list):
        arr = np.stack(values, axis=1)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3 and arr.shape == (
            len(X), len(columns), len(classes)
        ):
            arr = np.transpose(arr, (0, 2, 1))

    expected_shape = (
        len(X), len(classes), len(columns)
    )
    assert arr.shape == expected_shape, (
        f"Unexpected SHAP shape: {arr.shape}"
    )

    assert np.isfinite(arr).all()

    classwise = np.abs(arr).mean(axis=0)
    global_importance = classwise.mean(axis=0)

    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "feature": columns,
        "mean_abs_shap": global_importance,
    }).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(
        OUT / "global_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        classwise,
        index=classes,
        columns=columns,
    ).to_csv(
        OUT / "classwise_mean_abs_shap.csv",
        encoding="utf-8-sig",
    )

    top_rows = []
    for class_index, class_name in enumerate(classes):
        indices = np.argsort(
            classwise[class_index]
        )[::-1][:20]

        for rank, feature_index in enumerate(indices, 1):
            top_rows.append({
                "generator": class_name,
                "rank": rank,
                "feature": columns[feature_index],
                "mean_abs_shap": float(
                    classwise[class_index, feature_index]
                ),
            })

    pd.DataFrame(top_rows).to_csv(
        OUT / "top20_by_generator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected[["file_id", "generator"]].to_csv(
        OUT / "selected_test_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("CREATING FINGERPRINT HEATMAP...", flush=True)

    top_indices = np.argsort(
        global_importance
    )[::-1][:TOP_N]

    relative = (
        100 * classwise
        / np.maximum(
            classwise.sum(axis=1, keepdims=True),
            1e-12,
        )
    )

    matrix = relative[:, top_indices]
    names = [columns[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(19, 13))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        names, rotation=75, ha="right", fontsize=8
    )

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)

    ax.set_xlabel("Acoustic feature")
    ax.set_ylabel("Generator")
    ax.set_title(
        "RuASD: class-wise acoustic fingerprint "
        "(LightGBM SHAP, 20 test records per generator)"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Share of total class-wise |SHAP| (%)",
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "generator_feature_shap_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "generator_feature_shap_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n========== TOP 20 GLOBAL FEATURES ==========")
    result = pd.DataFrame({
        "feature": columns,
        "mean_abs_shap": global_importance,
    }).sort_values(
        "mean_abs_shap", ascending=False
    )

    print(
        result.head(20).to_string(index=False),
        flush=True,
    )

    print("\nSHAP COMPLETED", flush=True)
    print("SAMPLES:", len(X), flush=True)
    print("GENERATORS:", len(classes), flush=True)
    print("FEATURES:", len(columns), flush=True)
    print("RESULTS:", OUT, flush=True)


if __name__ == "__main__":
    main()
