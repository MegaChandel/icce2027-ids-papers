"""Windowed evaluation under ROAD's INTERVAL-level labels (blueprint §9.2).

Per-frame scores are aggregated into sliding time windows. A window is
POSITIVE iff it overlaps the labelled injection interval:
    w_start < inj_end  and  w_end > inj_start
This is the overlap rule of Moriano et al.'s published protocol, with the
corner case (window strictly containing the interval) also counted positive.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def make_windows(t_start: float, t_end: float, length: float, stride: float) -> np.ndarray:
    """[n, 2] array of (start, end) windows covering [t_start, t_end]."""
    starts = np.arange(t_start, max(t_end - length, t_start) + 1e-12, stride)
    return np.stack([starts, starts + length], 1)


def window_scores(t: np.ndarray, s: np.ndarray, windows: np.ndarray, agg: str = "mean") -> np.ndarray:
    """Aggregate frame scores `s` at sorted times `t` into each window.

    agg="mean": mean frame score in the window (flag rate for 0/1 scores).
    agg="max":  max frame score in the window.
    Empty windows score 0.
    """
    order = np.argsort(t, kind="stable")
    t, s = t[order], np.asarray(s, float)[order]
    lo = np.searchsorted(t, windows[:, 0], "left")
    hi = np.searchsorted(t, windows[:, 1], "left")
    n = hi - lo
    if agg == "mean":
        cs = np.concatenate([[0.0], np.cumsum(s)])
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), 0.0)
    if agg == "max":
        out = np.zeros(len(windows))
        for i, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                out[i] = s[a:b].max()
        return out
    raise ValueError(agg)


def window_labels(windows: np.ndarray, interval: tuple[float, float] | None) -> np.ndarray:
    if interval is None:
        return np.zeros(len(windows), int)
    a, b = interval
    return ((windows[:, 0] < b) & (windows[:, 1] > a)).astype(int)


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Threshold such that at most `fpr` of held-out ambient windows exceed it."""
    return float(np.quantile(neg_scores, 1 - fpr, method="higher"))


def metrics(y: np.ndarray, score: np.ndarray, thr: float | None = None) -> dict:
    out = {"n_windows": int(len(y)), "n_pos": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auc_roc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc_roc"] = out["pr_auc"] = float("nan")
    if thr is not None:
        out["det_rate_at_1pct_fpr"] = float((score[y == 1] > thr).mean()) if y.sum() else float("nan")
        out["fpr_in_capture"] = float((score[y == 0] > thr).mean()) if (y == 0).any() else float("nan")
    return out


def bootstrap_auc_ci(y, score, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Stratified bootstrap over windows: 95% CI on AUC-ROC.

    Note: overlapping windows are not independent, so this CI is optimistic;
    the sensitivity experiment over window length/stride addresses that.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    sp, sn = score[pos], score[neg]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        a = sp[rng.integers(0, len(sp), len(sp))]
        b = sn[rng.integers(0, len(sn), len(sn))]
        aucs[i] = _auc_fast(a, b)
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def bootstrap_auc_diff_ci(y, s1, s2, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Paired stratified bootstrap: 95% CI on AUC(s1) - AUC(s2) for one capture."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.empty(n_boot)
    for i in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        d[i] = _auc_fast(s1[p], s1[n]) - _auc_fast(s2[p], s2[n])
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def _auc_fast(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as 1/2."""
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    obs = np.concatenate([[True], sa[1:] != sa[:-1]])
    dense = np.cumsum(obs) - 1
    cnt = np.flatnonzero(np.concatenate([obs, [True]]))
    avg = (cnt[1:] + cnt[:-1] + 1) / 2.0  # average 1-based rank of each tie group
    r = np.empty(len(a))
    r[order] = avg[dense]
    return r
