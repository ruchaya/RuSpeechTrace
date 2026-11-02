#!/usr/bin/env python
"""Summary of paired resampled fixed4 open-set experiments.

Reports SD across training selections for a fixed fold AND SD across folds
of seed-averaged results separately. Neither is a confidence interval on
independent corpora or independently sampled unknown generators.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["auroc_id_vs_ood", "aupr_ood", "fpr95", "oscr", "macro_f1_with_unknown", "known_accept_rate", "unknown_reject_rate"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/raw/fixed4_lr_C1_paired_80pct_train_5seeds")
    p.add_argument("--out", default="results/summaries")
    p.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    p.add_argument("--expected-folds", type=int, default=5)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    rows = []
    for path in root.glob("*/seed*/fold*/metrics.json"):
        o = json.loads(path.read_text(encoding="utf-8"))
        for method, m in o["methods"].items():
            rows.append({"representation": o["representation"], "seed": int(o["seed"]), "fold": str(o["fold"]), "method": method,
                         "train_sha256": o["train_sha256"], **{key: m[key] for key in METRICS}})
    if not rows:
        raise SystemExit("No finished fold results found")
    d = pd.DataFrame(rows).sort_values(["representation", "method", "fold", "seed"])
    reps = sorted(d.representation.unique())
    if set(reps) != {"acoustic_lr_C1", "wavlm_lr_C1"}:
        raise SystemExit("Need both Acoustic and WavLM representations to report paired results")
    methods = sorted(d.method.unique())
    if methods != sorted(["msp", "energy", "sme", "prototype"]):
        raise SystemExit(f"Unexpected methods: {methods}")
    for (rep, method), x in d.groupby(["representation", "method"]):
        if sorted(x.seed.unique()) != sorted(a.expected_seeds) or x.fold.nunique() != a.expected_folds:
            raise SystemExit(f"INCOMPLETE {rep}/{method}: seeds={sorted(x.seed.unique())}, folds={sorted(x.fold.unique())}")
        expected = len(a.expected_seeds) * a.expected_folds
        if len(x) != expected or x.duplicated(["seed", "fold"]).any():
            raise SystemExit(f"Duplicate/missing cells for {rep}/{method}")
    # Pairing sanity check: Acoustic and WavLM must use identical training IDs per seed/fold.
    h = d.pivot(index=["seed", "fold", "method"], columns="representation", values="train_sha256")
    if not (h["acoustic_lr_C1"] == h["wavlm_lr_C1"]).all():
        raise SystemExit("Acoustic and WavLM training selections do not match")
    out.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["train_sha256"]).to_csv(out/"fixed4_lr_80pct_5seeds_all_results.csv", index=False)
    # First average seeds within a fold, then calculate between-fold SD to reflect unknown-generator set variability.
    foldmeans = d.groupby(["representation", "method", "fold"], as_index=False)[METRICS].mean()
    foldmeans.to_csv(out/"fixed4_lr_80pct_5seeds_fold_means.csv", index=False)
    summary = foldmeans.groupby(["representation", "method"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{key}_{stat}_over_folds" for key, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out/"fixed4_lr_80pct_5seeds_summary.csv", index=False)
    # Seed uncertainty on same held-out data: mean over folds per seed; variation across training compositions only.
    seedmeans = d.groupby(["representation", "method", "seed"], as_index=False)[METRICS].mean()
    seedmeans.to_csv(out/"fixed4_lr_80pct_5seeds_seed_means.csv", index=False)
    pairs = foldmeans.pivot(index=["method", "fold"], columns="representation", values=METRICS)
    paired = []
    for method in methods:
        row = {"method": method}
        for metric in METRICS:
            diff = pairs.loc[method, (metric, "acoustic_lr_C1")] - pairs.loc[method, (metric, "wavlm_lr_C1")]
            row[f"delta_acoustic_minus_wavlm_{metric}_mean"] = float(diff.mean())
            row[f"delta_acoustic_minus_wavlm_{metric}_sd_over_folds"] = float(diff.std(ddof=1))
            row[f"acoustic_better_folds_{metric}"] = int((diff < 0).sum() if metric == "fpr95" else (diff > 0).sum())
        paired.append(row)
    pd.DataFrame(paired).to_csv(out/"fixed4_lr_80pct_5seeds_paired_fold_differences.csv", index=False)
    print("PASS: two representations, matching paired training subsets, 5 seeds x 5 folds x 4 methods")
    print(summary[["representation", "method", "auroc_id_vs_ood_mean_over_folds", "fpr95_mean_over_folds", "oscr_mean_over_folds", "macro_f1_with_unknown_mean_over_folds"]].to_string(index=False))
    print(f"Results saved in {out}")


if __name__ == "__main__":
    main()
