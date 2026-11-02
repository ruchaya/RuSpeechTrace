"""Standalone equivalence test, run before long experiments."""
from __future__ import annotations
import numpy as np
from fast_open_set_metrics import oscr_fast, open_set_metrics
from source_attribution.metrics import oscr, open_set_metrics as reference_open_set_metrics


def main():
    random = np.random.default_rng(20261021)
    for trial in range(100):
        n_known = int(random.integers(1, 90))
        n_unknown = int(random.integers(1, 100))
        true = random.choice(['A','B','C'], size=n_known)
        pred = random.choice(['A','B','C'], size=n_known)
        k = random.normal(size=n_known)
        u = random.normal(size=n_unknown)
        if trial % 3 == 0:
            k = np.round(k, 1)
            u = np.round(u, 1)
        if trial % 3 == 1:
            k = np.round(k, 0)
            u = np.round(u, 0)
        assert np.isclose(oscr_fast(true, pred, k, u), oscr(true, pred, k, u), atol=1e-12), trial
        fast = open_set_metrics(true, pred, k, u)
        slow = reference_open_set_metrics(true, pred, k, u)
        for name in slow:
            assert np.isclose(fast[name], slow[name], atol=1e-12), (trial, name, fast[name], slow[name])
    print('PASS: optimized metrics match original metrics on 100 test cases, including tied scores')


if __name__ == '__main__':
    main()
