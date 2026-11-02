
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import run_cross_dataset_open_set as experiment


SEEDS = [42, 123, 456, 789, 2026]
DIRECTIONS = [
    ("RuASD", "MLAAD"),
    ("MLAAD", "RuASD"),
]
METHODS = ("msp", "energy", "sme")

ROOT = Path("results/raw/cross_dataset_open_set_multiseed_5seeds")
EXISTING_SEED42 = Path(
    "results/raw/cross_dataset_open_set_full_seed42"
)

REFERENCE_RUASD = Path(
    "results/raw/shortcut_acoustic_no_duration/"
    "20260928T091631Z_seed42/model.joblib"
)


def model_path(domain, seed):
    if domain == "RuASD":
        if seed == 42:
            return REFERENCE_RUASD
        return Path(
            f"results/raw/ruasd_34class_no_duration_seed{seed}/"
            "model.joblib"
        )

    return Path(
        f"results/raw/mlaad_23class_closedset_seed{seed}/"
        "model.joblib"
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verify_models():
    print("VERIFYING ALL TEN MODELS", flush=True)

    reference_features = None
    reference_classes = {}

    m_meta_42 = load_json(
        "results/raw/mlaad_23class_closedset_seed42/"
        "experiment_context.json"
    )

    for seed in SEEDS:
        for domain in ("RuASD", "MLAAD"):
            path = model_path(domain, seed)

            if not path.is_file():
                raise FileNotFoundError(path)

            model = joblib.load(path)
            names = list(model.estimator.booster_.feature_name())
            classes = list(map(str, model.classes_))

            assert len(names) == 789
            assert len(set(names)) == 789
            assert len(classes) == (
                34 if domain == "RuASD" else 23
            )

            if reference_features is None:
                reference_features = names
            else:
                assert names == reference_features, (
                    f"Feature mismatch: {domain}, seed {seed}"
                )

            if domain not in reference_classes:
                reference_classes[domain] = classes
            else:
                assert classes == reference_classes[domain], (
                    f"Class mismatch: {domain}, seed {seed}"
                )

            if domain == "MLAAD":
                meta = load_json(
                    f"results/raw/mlaad_23class_closedset_seed{seed}/"
                    "experiment_context.json"
                )
                assert meta["seed"] == seed
                assert meta["feature_count"] == 789
                assert meta["duration_excluded"] is True

                for key in (
                    "feature_sha256",
                    "split_sha256",
                    "config_sha256",
                ):
                    assert meta[key] == m_meta_42[key], (
                        f"MLAAD {key} mismatch, seed {seed}"
                    )

            elif seed != 42:
                meta = load_json(
                    f"results/raw/ruasd_34class_no_duration_seed{seed}/"
                    "model_metadata.json"
                )
                assert meta["seed"] == seed
                assert meta["n_features"] == 789
                assert meta["n_classes"] == 34
                assert meta["duration_excluded"] is True

            print(
                f"VERIFIED: {domain}, seed={seed}, "
                f"classes={len(classes)}",
                flush=True,
            )

            del model

    print("ALL MODELS VERIFIED", flush=True)


def validate_result(report, predictions_path):
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        assert direction in report

        methods = report[direction]["methods"]
        assert set(methods) == set(METHODS)

        expected_total = (
            23000 if target == "MLAAD" else 68000
        )
        expected_test = (
            4600 if target == "MLAAD" else 13600
        )

        for method in METHODS:
            result = methods[method]
            assert result["full_external"]["n_total"] == expected_total
            assert (
                result["frozen_test_external"]["n_total"]
                == expected_test
            )

    expected_rows = (23000 + 68000) * len(METHODS)
    actual_rows = pq.ParquetFile(
        predictions_path
    ).metadata.num_rows

    assert actual_rows == expected_rows, (
        f"Prediction count mismatch: {actual_rows}"
    )


def run_new_seed(seed):
    final_dir = ROOT / f"seed{seed}"
    temp_dir = ROOT / f"_seed{seed}_incomplete"

    if final_dir.exists():
        metrics_path = final_dir / "metrics.json"
        predictions_path = final_dir / "predictions.parquet"

        if not metrics_path.is_file() or not predictions_path.is_file():
            raise RuntimeError(
                f"Incomplete existing result: {final_dir}"
            )

        report = load_json(metrics_path)
        validate_result(report, predictions_path)
        print(f"REUSING COMPLETED SEED {seed}", flush=True)
        return report

    if temp_dir.exists():
        raise RuntimeError(
            f"Previous incomplete run detected: {temp_dir}. "
            "Do not delete it without checking its contents."
        )

    print(f"\n########## SEED {seed} ##########", flush=True)

    reports = {}
    predictions = []

    for source, target in DIRECTIONS:
        experiment.DOMAINS[source]["model"] = model_path(
            source, seed
        )

        report, pred = experiment.run_direction(
            source,
            target,
            experiment.load_mapping(),
        )

        direction = f"{source}_to_{target}"
        reports[direction] = report
        predictions.append(pred)

    combined = pd.concat(predictions, ignore_index=True)

    expected_rows = (23000 + 68000) * len(METHODS)
    assert len(combined) == expected_rows
    assert set(combined["method"]) == set(METHODS)

    temp_dir.mkdir(parents=True, exist_ok=False)

    save_json(temp_dir / "metrics.json", reports)
    combined.to_parquet(
        temp_dir / "predictions.parquet",
        index=False,
    )

    save_json(
        temp_dir / "context.json",
        {
            "seed": seed,
            "ruasd_model": str(model_path("RuASD", seed)),
            "mlaad_model": str(model_path("MLAAD", seed)),
            "features": 789,
            "source_validation_reject_quantile": 0.05,
            "methods": list(METHODS),
            "target_data_used_for_training": False,
            "mapping_file": str(experiment.MAPPING_PATH),
        },
    )

    validate_result(
        reports,
        temp_dir / "predictions.parquet",
    )

    temp_dir.rename(final_dir)

    print(
        f"SEED {seed} COMPLETED AND SAVED: {final_dir}",
        flush=True,
    )
    return reports


def flatten_results(all_results):
    rows = []
    family_rows = []

    for seed in SEEDS:
        reports = all_results[seed]

        for source, target in DIRECTIONS:
            direction = f"{source}_to_{target}"
            report = reports[direction]

            for method in METHODS:
                entry = report["methods"][method]

                for subset in (
                    "full_external",
                    "frozen_test_external",
                ):
                    metrics = entry[subset]

                    row = {
                        "seed": seed,
                        "direction": direction,
                        "method": method,
                        "subset": subset,
                        "source_test_macro_f1": report[
                            "source_in_domain_test_macro_f1"
                        ],
                        "threshold": entry["threshold"],
                        "source_validation_accept_rate": entry[
                            "source_validation_accept_rate"
                        ],
                        "source_test_accept_rate": entry[
                            "source_test_accept_rate"
                        ],
                    }

                    for key, value in metrics.items():
                        if isinstance(
                            value,
                            (int, float, np.integer, np.floating),
                        ):
                            row[key] = value

                    rows.append(row)

                    for family, fm in metrics[
                        "per_shared_family"
                    ].items():
                        family_rows.append({
                            "seed": seed,
                            "direction": direction,
                            "method": method,
                            "subset": subset,
                            "family": family,
                            **fm,
                        })

    return pd.DataFrame(rows), pd.DataFrame(family_rows)


def summarize(df, group_columns):
    numeric_columns = [
        col for col in df.select_dtypes(
            include=[np.number]
        ).columns
        if col not in ("seed", "n", "n_total", "n_shared", "n_unmapped")
    ]

    summary = (
        df.groupby(group_columns, sort=True)[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    experiment.check_feature_signature()
    verify_models()

    seed42_metrics = EXISTING_SEED42 / "metrics.json"
    seed42_predictions = EXISTING_SEED42 / "predictions.parquet"

    if not seed42_metrics.is_file():
        raise FileNotFoundError(seed42_metrics)
    if not seed42_predictions.is_file():
        raise FileNotFoundError(seed42_predictions)

    all_results = {}

    all_results[42] = load_json(seed42_metrics)
    validate_result(
        all_results[42],
        seed42_predictions,
    )

    print(
        "\nSEED 42: REUSING PREVIOUSLY VERIFIED EXPERIMENT",
        flush=True,
    )

    for seed in [123, 456, 789, 2026]:
        all_results[seed] = run_new_seed(seed)

    by_seed, family_by_seed = flatten_results(all_results)

    assert len(by_seed) == 5 * 2 * 3 * 2
    assert len(family_by_seed) == 5 * 2 * 3 * 2 * 3

    summary = summarize(
        by_seed,
        ["direction", "method", "subset"],
    )

    family_summary = summarize(
        family_by_seed,
        ["direction", "method", "subset", "family"],
    )

    by_seed.to_csv(
        ROOT / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        ROOT / "metrics_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_by_seed.to_csv(
        ROOT / "family_metrics_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_summary.to_csv(
        ROOT / "family_metrics_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        ROOT / "manifest.json",
        {
            "seeds": SEEDS,
            "methods": list(METHODS),
            "seed42_existing_results": str(EXISTING_SEED42),
            "other_seed_results": [
                str(ROOT / f"seed{s}")
                for s in SEEDS if s != 42
            ],
            "source_training_splits_frozen": True,
            "external_target_splits_evaluation_only": True,
            "number_of_external_model_sample_pairs": 455000,
            "number_of_method_prediction_rows": 1365000,
            "unknown_definition": (
                "No confirmed correspondence among "
                "three predefined shared families"
            ),
        },
    )

    selected = summary[
        summary["subset"].eq("full_external")
    ]

    columns = [
        "direction",
        "method",
        "auroc_id_vs_ood_mean",
        "auroc_id_vs_ood_std",
        "fpr95_mean",
        "fpr95_std",
        "macro_f1_four_labels_mean",
        "macro_f1_four_labels_std",
        "shared_correct_attribution_rate_mean",
        "shared_correct_attribution_rate_std",
        "unmapped_rejection_rate_mean",
        "unmapped_rejection_rate_std",
    ]

    print("\n========== FIVE-SEED MEAN AND SD ==========")
    print(
        selected[columns].round(5).to_string(index=False),
        flush=True,
    )

    print("\nALL FIVE SEEDS COMPLETED", flush=True)
    print("RESULTS:", ROOT, flush=True)


if __name__ == "__main__":
    main()
