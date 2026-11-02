
from pathlib import Path
import json
import numpy as np
import pandas as pd

SEEDS = [42, 123, 456, 789, 2026]
FOLDS = range(1, 6)
METHODS = ["msp", "energy", "sme", "prototype"]

METRICS = [
    "auroc_id_vs_ood",
    "aupr_ood",
    "aupr_id",
    "fpr95",
    "oscr",
    "known_macro_f1",
    "macro_f1_with_unknown",
    "known_accept_rate",
    "unknown_reject_rate",
]

OUTPUT = Path("results/summary/mlaad_23class_open_set_v1")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    records = []
    generator_records = []
    hashes = {}

    for seed in SEEDS:
        for fold in FOLDS:
            folder = Path(
                f"results/raw/mlaad_23class_openset_"
                f"fold{fold}_seed{seed}"
            )

            required = [
                "context.json",
                "metrics.json",
                "predictions.parquet",
                "model.joblib",
                "prototype.joblib",
                "feature_columns.txt",
            ]

            for filename in required:
                path = folder / filename
                if not path.is_file():
                    raise FileNotFoundError(path)

            context = read_json(folder / "context.json")
            metrics = read_json(folder / "metrics.json")
            predictions = pd.read_parquet(
                folder / "predictions.parquet"
            )

            assert context["seed"] == seed
            assert context["fold"] == fold
            assert context["features"] == 789
            assert set(metrics) == set(METHODS)

            known = set(context["known_generators"])
            unknown = set(context["unknown_generators"])

            assert not (known & unknown)
            assert len(known | unknown) == 23
            assert len(predictions) == 4600 * 4

            for key in [
                "split_sha256",
                "features_sha256",
                "folds_sha256",
                "config_sha256",
            ]:
                value = context[key]
                if key in hashes:
                    assert hashes[key] == value, key
                else:
                    hashes[key] = value

            for method in METHODS:
                d = predictions.loc[
                    predictions["method"] == method
                ]

                assert len(d) == 4600
                assert d["path"].is_unique

                k = d.loc[d["is_known_true"]]
                u = d.loc[~d["is_known_true"]]

                assert len(k) == len(known) * 200
                assert len(u) == len(unknown) * 200
                assert set(k["generator"]) == known
                assert set(u["generator"]) == unknown
                assert set(d["generator_pred"]).issubset(
                    known
                )

                assert u.groupby("generator").size().eq(
                    200
                ).all()

                m = metrics[method]

                assert np.isclose(
                    k["accepted_known"].mean(),
                    m["known_accept_rate"],
                )

                assert np.isclose(
                    1 - u["accepted_known"].mean(),
                    m["unknown_reject_rate"],
                )

                row = {
                    "seed": seed,
                    "fold": fold,
                    "method": method,
                }

                for metric in METRICS:
                    value = float(m[metric])
                    assert np.isfinite(value)
                    assert 0 <= value <= 1
                    row[metric] = value

                records.append(row)

                for generator, group in u.groupby(
                    "generator"
                ):
                    generator_records.append({
                        "seed": seed,
                        "fold": fold,
                        "method": method,
                        "generator": generator,
                        "n_test": len(group),
                        "reject_rate": float(
                            1 - group["accepted_known"].mean()
                        ),
                    })

            print(
                f"VERIFIED fold={fold} seed={seed}",
                flush=True,
            )

    runs = pd.DataFrame(records)
    generators = pd.DataFrame(generator_records)

    assert len(runs) == 25 * 4
    assert len(generators) == 25 * 23 * 4 // 5

    # Each generator is evaluated as unknown once per seed.
    coverage = generators.groupby(
        ["seed", "method", "generator"]
    ).size()

    assert coverage.eq(1).all()
    assert generators["generator"].nunique() == 23

    # Every seed uses the same five folds.
    by_seed = (
        runs.groupby(["method", "seed"], as_index=False)[
            METRICS
        ]
        .mean()
    )

    # Every fold is evaluated over all five seeds.
    by_fold = (
        runs.groupby(["method", "fold"], as_index=False)[
            METRICS
        ]
        .mean()
    )

    main_rows = []

    for method in METHODS:
        for metric in METRICS:
            seed_values = by_seed.loc[
                by_seed["method"] == method, metric
            ]

            fold_values = by_fold.loc[
                by_fold["method"] == method, metric
            ]

            main_rows.append({
                "method": method,
                "metric": metric,
                "mean": float(seed_values.mean()),
                "sd_across_seed_means": float(
                    seed_values.std(ddof=1)
                ),
                "sd_across_fold_means": float(
                    fold_values.std(ddof=1)
                ),
                "min_fold_mean": float(
                    fold_values.min()
                ),
                "max_fold_mean": float(
                    fold_values.max()
                ),
            })

    main_table = pd.DataFrame(main_rows)

    per_generator = (
        generators.groupby(
            ["method", "generator", "fold"]
        )["reject_rate"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "reject_rate_mean",
            "std": "reject_rate_sd_across_seeds",
            "min": "reject_rate_min",
            "max": "reject_rate_max",
        })
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    runs.to_csv(
        OUTPUT / "all_25_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_table.to_csv(
        OUTPUT / "overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_fold.to_csv(
        OUTPUT / "summary_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_seed.to_csv(
        OUTPUT / "summary_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_generator.to_csv(
        OUTPUT / "unknown_generators.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nOVERALL RESULTS")
    print(
        main_table.loc[
            main_table["metric"].isin([
                "auroc_id_vs_ood",
                "fpr95",
                "oscr",
                "aupr_ood",
                "unknown_reject_rate",
            ])
        ][[
            "method", "metric", "mean",
            "sd_across_seed_means",
            "sd_across_fold_means",
        ]].round(5).to_string(index=False)
    )

    print("\nSME: HARDEST UNKNOWN GENERATORS")

    difficult = per_generator.loc[
        per_generator["method"] == "sme"
    ].sort_values("reject_rate_mean")

    print(
        difficult.head(10).round(4).to_string(
            index=False
        )
    )

    print("\nOUTPUT:", OUTPUT)
    print("RUNS VERIFIED: 25")
    print("METHOD RESULTS VERIFIED: 100")
    print("ALL SUMMARIES SAVED")


if __name__ == "__main__":
    main()
