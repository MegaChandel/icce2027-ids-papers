"""Learned controls (blueprint §9.1). Fitted on ambient only, like the trivial detectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from .detectors import BYTE_COLS, _check_ambient


@dataclass
class PerAIDIsolationForest:
    """One IsolationForest per AID over the 8 payload bytes.

    Score = -score_samples (higher = more anomalous). Unseen AIDs get the maximum
    score observed on ambient plus one, so they rank above every fitted frame.
    """
    seed: int = 0
    n_estimators: int = 200
    max_fit: int = 20_000  # frames per AID used for fitting (random subsample)
    name: str = "IF_perAID"
    models: dict = field(default_factory=dict)
    unseen_score: float = 1.0

    def fit(self, frames):
        f = _check_ambient(frames)
        rng = np.random.default_rng(self.seed)
        b = f[BYTE_COLS].to_numpy(np.float32)
        top = 0.0
        for aid, idx in f.groupby("aid").indices.items():
            if len(idx) > self.max_fit:
                idx = rng.choice(idx, self.max_fit, replace=False)
            m = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1)
            m.fit(b[idx])
            self.models[aid] = m
            top = max(top, float(-m.score_samples(b[idx]).max()))
        self.unseen_score = top + 1.0
        return self

    def score(self, frames) -> np.ndarray:
        b = frames[BYTE_COLS].to_numpy(np.float32)
        out = np.full(len(frames), self.unseen_score)
        for aid, idx in frames.groupby("aid").indices.items():
            m = self.models.get(aid)
            if m is not None:
                out[idx] = -m.score_samples(b[idx])
        return out
