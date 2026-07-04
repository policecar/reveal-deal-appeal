"""Unit tests for the CV harness helpers (no ML training involved)."""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from crossval import cross_validate_baseline, fold_metrics


def test_fold_metrics_zero_division_safe():
    """All-majority predictions must yield 0 F2 without raising/warning."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = fold_metrics(1, y_true, y_pred)
    assert m["f2_win"] == 0.0
    assert 0.0 < m["macro_f1"] < 1.0


def test_cross_validate_baseline_learns_separable_data():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    X = rng.normal(size=(40, 5))
    X[y == 1] += 3.0  # cleanly separable

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    res = cross_validate_baseline(lambda: LogisticRegression(), X, y, folds)

    assert len(res) == 5
    assert np.mean([m["f2_win"] for m in res]) > 0.9
