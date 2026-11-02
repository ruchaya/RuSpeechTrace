"""Drop-in metric definitions for efficient fixed-4s open-set runs.

The FPR95/AUPR conventions and OSCR integration order intentionally reproduce
source_attribution.metrics from the supplied project. The old OSCR implementation
loops once per distinct score and is quadratic in test size. The implementation
below uses sorted-score cumulative counts for O(N log N) complexity.

This module is imported only by run_open_set_fixed4_optimized.py; no existing
project module, script, or frozen split is overwritten.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from source_attribution.metrics import fpr95


def oscr_fast(y_known_true, y_known_pred, known_id_score, unknown_id_score):
    sk = np.asarray(known_id_score, dtype=float)
    su = np.asarray(unknown_id_score, dtype=float)
    true = np.asarray(y_known_true).astype(str)
    pred = np.asarray(y_known_pred).astype(str)
    if not len(sk) or not len(su):
        raise ValueError("OSCR requires nonempty known and unknown test samples.")
    if len(sk) != len(true) or len(sk) != len(pred):
        raise ValueError("OSCR known scores and labels must have matching lengths.")
    if not np.isfinite(sk).all() or not np.isfinite(su).all():
        raise ValueError("OSCR requires finite scores.")

    order_known = np.argsort(sk)
    sorted_known = sk[order_known]
    sorted_correct = (true == pred)[order_known].astype(np.int64)
    cumulative_correct = np.r_[0, np.cumsum(sorted_correct, dtype=np.int64)]
    sorted_unknown = np.sort(su)
    # Match the original threshold order and integrate with its np.argsort(FPR).
    thresholds = np.r_[np.inf, np.unique(np.r_[sk, su])[::-1], -np.inf]
    known_lt_count = np.searchsorted(sorted_known, thresholds, side="left")
    unknown_lt_count = np.searchsorted(sorted_unknown, thresholds, side="left")
    ccr = (cumulative_correct[-1] - cumulative_correct[known_lt_count]) / len(sk)
    fpr = (len(su) - unknown_lt_count) / len(su)
    curve_order = np.argsort(fpr)
    return float(np.trapezoid(ccr[curve_order], fpr[curve_order]))


def open_set_metrics(known_true, known_pred, known_id_score, unknown_id_score):
    known_scores = np.asarray(known_id_score, dtype=float)
    unknown_scores = np.asarray(unknown_id_score, dtype=float)
    y_id = np.r_[np.ones(len(known_scores), dtype=int), np.zeros(len(unknown_scores), dtype=int)]
    id_score = np.r_[known_scores, unknown_scores]
    return {
        "auroc_id_vs_ood": float(roc_auc_score(y_id, id_score)),
        "aupr_ood": float(average_precision_score(1 - y_id, -id_score)),
        "aupr_id": float(average_precision_score(y_id, id_score)),
        "fpr95": fpr95(y_id, id_score),
        "oscr": oscr_fast(known_true, known_pred, known_scores, unknown_scores),
    }
