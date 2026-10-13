"""Trivial payload-level detectors T1-T4 (blueprint §8.2).

All detectors are fitted on AMBIENT captures only. Fitting refuses any frame
table whose `family` column (if present) is not "ambient" -- this invariant is
what makes the detectors "trivial", so it is enforced in code.

Frame tables (raw) have columns: t, aid, dlc, b0..b7 (b_i is -1 when i >= dlc).
Signal tables are wide: t, aid, plus one float column per signal (NaN when the
signal does not belong to that AID).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BYTE_COLS = [f"b{i}" for i in range(8)]


class AttackDataInFitError(ValueError):
    pass


def _check_ambient(frames: pd.DataFrame | list[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(frames, (list, tuple)):
        frames = pd.concat(frames, ignore_index=True)
    if "family" in frames.columns:
        bad = set(frames["family"].unique()) - {"ambient"}
        if bad:
            raise AttackDataInFitError(f"non-ambient data passed to fit: {bad}")
    return frames


# ---------------------------------------------------------------- raw bytes

@dataclass
class ByteValueSet:
    """T1: flag a frame if any byte takes a value never seen for (AID, byte)."""
    name: str = "T1_valueset"
    seen: dict = field(default_factory=dict)  # aid -> bool[8, 257] (index 256 == absent byte)

    def fit(self, frames):
        f = _check_ambient(frames)
        b = f[BYTE_COLS].to_numpy(np.int16)
        b = np.where(b < 0, 256, b)
        for aid, idx in f.groupby("aid").indices.items():
            m = np.zeros((8, 257), bool)
            for j in range(8):
                m[j, np.unique(b[idx, j])] = True
            self.seen[aid] = m
        return self

    def byte_flags(self, frames) -> np.ndarray:
        """bool[n, 8]: per-byte violation. Unseen AIDs violate on every byte."""
        b = frames[BYTE_COLS].to_numpy(np.int16)
        b = np.where(b < 0, 256, b)
        out = np.ones((len(frames), 8), bool)
        for aid, idx in frames.groupby("aid").indices.items():
            m = self.seen.get(aid)
            if m is None:
                continue
            for j in range(8):
                out[idx, j] = ~m[j, b[idx, j]]
        return out

    def score(self, frames) -> np.ndarray:
        return self.byte_flags(frames).sum(1).astype(float)


@dataclass
class ByteRange:
    """T2: flag a frame if any byte falls outside [min, max] seen for (AID, byte)."""
    name: str = "T2_range"
    lo: dict = field(default_factory=dict)
    hi: dict = field(default_factory=dict)

    def fit(self, frames):
        f = _check_ambient(frames)
        g = f.groupby("aid")[BYTE_COLS]
        for aid, mn in g.min().iterrows():
            self.lo[aid] = mn.to_numpy(np.int16)
        for aid, mx in g.max().iterrows():
            self.hi[aid] = mx.to_numpy(np.int16)
        return self

    def byte_flags(self, frames) -> np.ndarray:
        b = frames[BYTE_COLS].to_numpy(np.int16)
        out = np.ones((len(frames), 8), bool)
        for aid, idx in frames.groupby("aid").indices.items():
            if aid not in self.lo:
                continue
            out[idx] = (b[idx] < self.lo[aid]) | (b[idx] > self.hi[aid])
        return out

    def byte_excess(self, frames, normalise: bool = False) -> np.ndarray:
        """float[n, 8]: distance outside [min, max] per byte (unseen AID: 256 per byte).
        normalise=True divides by the range width + 1, making bytes comparable."""
        b = frames[BYTE_COLS].to_numpy(np.int16)
        out = np.full((len(frames), 8), 256.0)
        for aid, idx in frames.groupby("aid").indices.items():
            if aid not in self.lo:
                continue
            lo, hi = self.lo[aid], self.hi[aid]
            d = np.maximum(lo - b[idx], 0) + np.maximum(b[idx] - hi, 0)
            out[idx] = d / (hi - lo + 1) if normalise else d
        return out

    def score(self, frames) -> np.ndarray:
        """Continuous score: total distance outside the range, summed over bytes."""
        return self.byte_excess(frames).sum(1)


@dataclass
class PayloadNovelty:
    """T4: flag a frame if its exact payload tuple was never seen for that AID."""
    name: str = "T4_novelty"
    seen: dict = field(default_factory=dict)  # aid -> set of int64 payload keys

    @staticmethod
    def _keys(frames) -> np.ndarray:
        b = frames[BYTE_COLS].to_numpy(np.int64) + 1  # absent (-1) -> 0
        k = np.zeros(len(frames), np.uint64)
        for j in range(8):
            k = k * np.uint64(257) + b[:, j].astype(np.uint64)
        return k

    def fit(self, frames):
        f = _check_ambient(frames)
        k = self._keys(f)
        for aid, idx in f.groupby("aid").indices.items():
            self.seen[aid] = set(np.unique(k[idx]).tolist())
        return self

    def score(self, frames) -> np.ndarray:
        k = self._keys(frames)
        out = np.ones(len(frames))
        for aid, idx in frames.groupby("aid").indices.items():
            s = self.seen.get(aid)
            if s is None:
                continue
            out[idx] = [0.0 if x in s else 1.0 for x in k[idx].tolist()]
        return out


# ------------------------------------------------------------------ signals

@dataclass
class SignalEnvelope:
    """T3: flag if any decoded signal leaves its ambient [p_lo, p_hi] envelope.

    Signal tables follow ROAD's signal_extractions layout: one row per frame,
    columns t, aid, s1..sK where s_i is "Signal_i_of_ID" (meaning depends on aid).
    Envelopes are therefore kept per (aid, i).

    `q` is the tail mass trimmed at each end, in percent (0 -> [p0, p100],
    0.1 -> [p0.1, p99.9]). Score is the max envelope excursion normalised by
    the envelope width (min width 1), so it is comparable across signals.
    Unseen AIDs score 1.0 per present signal.
    """
    q: float = 0.0
    name: str = "T3_envelope"
    lo: dict = field(default_factory=dict)  # aid -> float[K] (NaN: signal never seen)
    hi: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.q:
            self.name = f"T3_envelope_p{self.q:g}"

    @staticmethod
    def signal_cols(df) -> list[str]:
        return [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]

    def fit(self, signals):
        s = _check_ambient(signals)
        cols = self.signal_cols(s)
        v = s[cols].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN signal slots
            for aid, idx in s.groupby("aid").indices.items():
                self.lo[aid] = np.nanpercentile(v[idx], self.q, axis=0)
                self.hi[aid] = np.nanpercentile(v[idx], 100 - self.q, axis=0)
        return self

    def signal_excess(self, signals) -> np.ndarray:
        """float[n, K]: per-row, per-signal normalised excursion (0 inside)."""
        cols = self.signal_cols(signals)
        v = signals[cols].to_numpy(float)
        out = np.where(np.isnan(v), 0.0, 1.0)  # default: unseen AID
        for aid, idx in signals.groupby("aid").indices.items():
            if aid not in self.lo:
                continue
            lo, hi = self.lo[aid], self.hi[aid]
            w = np.maximum(np.nan_to_num(hi - lo, nan=0.0), 1.0)
            e = np.maximum(lo - v[idx], 0) + np.maximum(v[idx] - hi, 0)
            e = np.where(np.isnan(lo) & ~np.isnan(v[idx]), 1.0, e)  # signal unseen in ambient
            out[idx] = np.nan_to_num(e / w, nan=0.0)
        return out

    def score(self, signals) -> np.ndarray:
        e = self.signal_excess(signals)
        return e.max(axis=1) if e.shape[1] else np.zeros(len(signals))


RAW_DETECTORS = {"T1_valueset": ByteValueSet, "T2_range": ByteRange, "T4_novelty": PayloadNovelty}


def fit_trivial(ambient_frames, ambient_signals=None, t3_q=(0.0, 0.1)):
    """Fit every trivial detector on ambient data. Raises on any attack data."""
    dets = [cls().fit(ambient_frames) for cls in RAW_DETECTORS.values()]
    if ambient_signals is not None:
        dets += [SignalEnvelope(q=q).fit(ambient_signals) for q in t3_q]
    return dets
