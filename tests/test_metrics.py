import numpy as np

from source_attribution.metrics import (
    closed_set_metrics,
    fpr95,
    oscr,
    top_k_accuracy,
)
from source_attribution.ood import energy_id_score, sme_id_score


def test_fpr95_perfect():
    y = np.r_[np.ones(100), np.zeros(100)]
    s = np.r_[np.linspace(.8, 1, 100), np.linspace(0, .2, 100)]
    assert fpr95(y, s) == 0.


def test_oscr_perfectish():
    y = np.array(["a", "b", "a", "b"])
    p = y.copy()
    ks = np.array([.9, .85, .88, .83])
    us = np.array([.1, .2, .15])
    assert oscr(y, p, ks, us) > .95


def test_energy_orientation():
    assert energy_id_score(
        np.array([[8., 0.], [7., 0.]])
    ).mean() > energy_id_score(
        np.array([[.1, .1], [0., 0.]])
    ).mean()


def test_sme_finite():
    assert np.isfinite(
        sme_id_score(
            np.array([[2., 0., -1.], [0., 0., 0.]]),
            4.,
        )
    ).all()


def test_closed_set_macro_f1_and_balanced_accuracy():
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    proba = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    metrics = closed_set_metrics(
        y_true,
        y_pred,
        proba,
        np.array(["a", "b"]),
    )
    assert np.isclose(metrics["macro_f1"], 11.0 / 15.0)
    assert np.isclose(metrics["balanced_accuracy"], 0.75)
    assert np.isclose(metrics["top1_accuracy"], 0.75)


def test_top_k_accuracy():
    classes = np.array(["a", "b", "c"])
    y_true = np.array(["a", "b", "c"])
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.5, 0.4, 0.1],
        [0.6, 0.3, 0.1],
    ])
    assert np.isclose(top_k_accuracy(y_true, proba, classes, 1), 1 / 3)
    assert np.isclose(top_k_accuracy(y_true, proba, classes, 2), 2 / 3)
    assert np.isclose(top_k_accuracy(y_true, proba, classes, 3), 1.0)
