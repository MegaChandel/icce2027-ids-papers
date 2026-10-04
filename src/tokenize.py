"""Shared tokenisation, windowing and train/test splits (blueprint II §8.1, revised).

Index 0 is reserved for "message ID unseen in the vocabulary" (OOV); cross-campaign
transfer produces exactly these. Windows never span a file boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .inventory import load, load_cfg

OOV = 0


@dataclass(frozen=True)
class Segment:
    """A contiguous slice [lo, hi) of a capture, as fractions of its frame count."""
    name: str
    lo: float = 0.0
    hi: float = 1.0


def frames(seg: Segment):
    df = load(seg.name)
    n = len(df)
    return df.iloc[int(n * seg.lo): int(n * seg.hi)]


def build_vocab(segments) -> dict[int, int]:
    ids = sorted({int(x) for s in segments for x in frames(s)["msgid"].unique()})
    return {m: i + 1 for i, m in enumerate(ids)}


def windows(seg: Segment, vocab: dict[int, int], length: int, stride: int, with_dt: bool = False,
            with_stats: bool = False):
    """Returns (tokens int64[n, L], duration float[n], oov_rate float[n]) and, if with_dt,
    log inter-arrival times float32[n, L] (first step of each window uses the gap before it)."""
    df = frames(seg)
    tok = np.array([vocab.get(int(m), OOV) for m in df["msgid"].to_numpy()], np.int64)
    t = df["t"].to_numpy()
    if len(tok) < length:
        empty = (np.zeros((0, length), np.int64), np.zeros(0), np.zeros(0))
        return empty + ((np.zeros((0, length), np.float32),) if with_dt else ()) + \
            ((np.zeros((0, length, 2), np.float32),) if with_stats else ())
    starts = np.arange(0, len(tok) - length + 1, stride)
    idx = starts[:, None] + np.arange(length)[None, :]
    w = tok[idx]
    dur = t[starts + length - 1] - t[starts]
    extra = ()
    if with_dt:
        gap = np.log(np.maximum(np.diff(t, prepend=t[0]), 1e-6)).astype(np.float32)
        extra += (gap[idx],)
    if with_stats:
        extra += (step_stats(tok, t, len(vocab) + 1)[idx],)
    return (w, dur, (w == OOV).mean(1)) + extra


def step_stats(tok: np.ndarray, t: np.ndarray, vocab_size: int, win: int = 30) -> np.ndarray:
    """GALAXY's per-step statistics over the preceding `win` messages: Shannon entropy of the
    message IDs and mean log inter-arrival time. Returns float32 [N, 2]."""
    n = len(tok)
    oh = np.zeros((n + 1, vocab_size), np.int32)
    oh[np.arange(1, n + 1), tok] = 1
    cum = oh.cumsum(0)
    lo = np.maximum(np.arange(1, n + 1) - win, 0)
    cnt = cum[1:] - cum[lo]
    p = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * np.log2(p), 0.0)).sum(1)
    gap = np.log(np.maximum(np.diff(t, prepend=t[0]), 1e-6))
    cg = np.concatenate([[0.0], np.cumsum(gap)])
    mgap = (cg[1:] - cg[lo]) / np.maximum(np.arange(1, n + 1) - lo, 1)
    return np.stack([ent, mgap], 1).astype(np.float32)


def caps_by(campaign: str):
    return [c for c in load_cfg()["captures"] if c["campaign"] == campaign]


def split(campaign: str, test_frac: float = 0.2):
    """(train segments, test segments) with their labels/attacks, per campaign.

    OLD:    rep 1 files -> train, rep 2 files -> test (file split).
    NEW:    one file per class -> first 80% train, last 20% test (temporal block split).
    SHADOW: per (scenario, class), the last ceil(20%) of files -> test (file split, as GALAXY).
    """
    caps = caps_by(campaign)
    tr, te = [], []
    if campaign == "OLD":
        for c in caps:
            (tr if c["rep"] == 1 else te).append((Segment(c["name"]), c))
    elif campaign == "NEW":
        for c in caps:
            tr.append((Segment(c["name"], 0.0, 1 - test_frac), c))
            te.append((Segment(c["name"], 1 - test_frac, 1.0), c))
    elif campaign == "SHADOW":
        groups = {}
        for c in caps:
            groups.setdefault((c["scenario"], c["label"]), []).append(c)
        for g in groups.values():
            g = sorted(g, key=lambda c: c["rep"])
            k = max(1, math.ceil(len(g) * test_frac))
            for c in g[:-k]:
                tr.append((Segment(c["name"]), c))
            for c in g[-k:]:
                te.append((Segment(c["name"]), c))
    else:
        raise ValueError(campaign)
    return tr, te


def make_xy(pairs, vocab, length, stride, attacks=None, with_dt=False, with_stats=False):
    """Stack windows of (segment, capture) pairs. `attacks` filters by attack type."""
    X, D, O, T, ST, y, a = [], [], [], [], [], [], []
    for seg, c in pairs:
        if attacks is not None and c["attack"] not in attacks:
            continue
        out = windows(seg, vocab, length, stride, with_dt, with_stats)
        X.append(out[0]); D.append(out[1]); O.append(out[2])
        k = 3
        if with_dt:
            T.append(out[k]); k += 1
        if with_stats:
            ST.append(out[k])
        y.append(np.full(len(out[0]), c["label"])); a += [c["attack"]] * len(out[0])
    if not X:
        return None
    xy = dict(X=np.concatenate(X), dur=np.concatenate(D), oov=np.concatenate(O),
              y=np.concatenate(y), attack=np.array(a))
    if with_dt:
        xy["dt"] = np.concatenate(T)
    if with_stats:
        xy["stats"] = np.concatenate(ST)
    return xy
