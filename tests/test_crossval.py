"""Unit tests for the CV harness helpers (no ML training involved)."""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from crossval import (
    cross_validate_baseline,
    cross_validate_chunk_maxprob,
    fold_metrics,
    mean_pool_chunks,
)


def test_fold_metrics_zero_division_safe():
    """All-majority predictions must yield 0 F2 without raising/warning."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = fold_metrics(1, y_true, y_pred)
    assert m["f2_win"] == 0.0
    assert 0.0 < m["macro_f1"] < 1.0
    assert "auroc" not in m

    m = fold_metrics(1, y_true, y_pred, y_score=np.array([0.1, 0.2, 0.3, 0.9]))
    assert m["auroc"] == 1.0  # the win outranks every no-win


def _separable_data(n_pairs: int = 20):
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * n_pairs)
    X = rng.normal(size=(2 * n_pairs, 5))
    X[y == 1] += 3.0  # cleanly separable
    return X, y


def test_cross_validate_baseline_learns_separable_data():
    X, y = _separable_data()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    res = cross_validate_baseline(lambda: LogisticRegression(), X, y, folds)

    assert len(res["folds"]) == 5
    assert np.mean([m["f2_win"] for m in res["folds"]]) > 0.9
    assert not np.isnan(res["oof_scores"]).any()


def test_mean_pool_chunks_groups_by_owner():
    chunk_emb = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    owners = np.array([0, 0, 1])
    pooled = mean_pool_chunks(chunk_emb, owners, n_calls=2)
    assert np.allclose(pooled, [[2.0, 0.0], [0.0, 5.0]])


def test_cross_validate_chunk_maxprob_scores_calls():
    """Calls chunked 2x each; per-call score = max chunk win probability."""
    X, y = _separable_data()
    # duplicate every call into two identical chunks
    chunk_emb = np.repeat(X, 2, axis=0)
    owners = np.repeat(np.arange(len(y)), 2)

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    res = cross_validate_chunk_maxprob(
        lambda: LogisticRegression(), chunk_emb, owners, y, folds
    )

    assert len(res["folds"]) == 5
    assert np.mean([m["f2_win"] for m in res["folds"]]) > 0.9
    assert not np.isnan(res["oof_scores"]).any()
