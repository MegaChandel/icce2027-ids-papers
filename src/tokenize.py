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


def windows(seg: Segment, vocab: dict[int, int], length: int, stride: int, with_dt: bool = False):
    """Returns (tokens int64[n, L], duration float[n], oov_rate float[n]) and, if with_dt,
    log inter-arrival times float32[n, L] (first step of each window uses the gap before it)."""
    df = frames(seg)
    tok = np.array([vocab.get(int(m), OOV) for m in df["msgid"].to_numpy()], np.int64)
    t = df["t"].to_numpy()
    if len(tok) < length:
        empty = (np.zeros((0, length), np.int64), np.zeros(0), np.zeros(0))
        return empty + ((np.zeros((0, length), np.float32),) if with_dt else ())
    starts = np.arange(0, len(tok) - length + 1, stride)
    idx = starts[:, None] + np.arange(length)[None, :]
    w = tok[idx]
    dur = t[starts + length - 1] - t[starts]
    if not with_dt:
        return w, dur, (w == OOV).mean(1)
    gap = np.log(np.maximum(np.diff(t, prepend=t[0]), 1e-6)).astype(np.float32)
    return w, dur, (w == OOV).mean(1), gap[idx]


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


def make_xy(pairs, vocab, length, stride, attacks=None, with_dt=False):
    """Stack windows of (segment, capture) pairs. `attacks` filters by attack type."""
    X, D, O, T, y, a = [], [], [], [], [], []
    for seg, c in pairs:
        if attacks is not None and c["attack"] not in attacks:
            continue
        out = windows(seg, vocab, length, stride, with_dt)
        X.append(out[0]); D.append(out[1]); O.append(out[2])
        if with_dt:
            T.append(out[3])
        y.append(np.full(len(out[0]), c["label"])); a += [c["attack"]] * len(out[0])
    if not X:
        return None
    xy = dict(X=np.concatenate(X), dur=np.concatenate(D), oov=np.concatenate(O),
              y=np.concatenate(y), attack=np.array(a))
    if with_dt:
        xy["dt"] = np.concatenate(T)
    return xy
