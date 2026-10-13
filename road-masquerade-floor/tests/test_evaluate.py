import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluate import (_auc_fast, bootstrap_auc_ci, make_windows, metrics,
                          threshold_at_fpr, window_labels, window_scores)


def test_windows_and_labels():
    w = make_windows(0, 10, 2, 1)
    assert w[0].tolist() == [0, 2] and w[-1].tolist() == [8, 10]
    y = window_labels(w, (4.5, 5.5))
    assert y.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]


def test_window_scores_mean_max():
    t = np.array([0.1, 0.2, 1.5, 3.5]); s = np.array([1, 0, 1, 5.0])
    w = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert window_scores(t, s, w, "mean").tolist() == [0.5, 1, 0, 5]
    assert window_scores(t, s, w, "max").tolist() == [1, 1, 0, 5]


def test_auc_fast_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); s = rng.integers(0, 5, 300).astype(float) + y
    assert np.isclose(_auc_fast(s[y == 1], s[y == 0]), roc_auc_score(y, s))


def test_metrics_and_threshold():
    neg = np.linspace(0, 1, 101)
    thr = threshold_at_fpr(neg, 0.01)
    assert (neg > thr).mean() <= 0.01
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.9, 2.0])
    m = metrics(y, s, thr)
    assert m["auc_roc"] == 1.0 and m["det_rate_at_1pct_fpr"] == 0.5
    lo, hi = bootstrap_auc_ci(y, s, n_boot=200)
    assert lo == hi == 1.0
