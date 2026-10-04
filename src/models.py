"""Detector families (blueprint II §8.2) plus two trivial baselines.

All detectors expose fit(train: dict) -> self and score(test: dict) -> float[n] (higher =
more anomalous / more likely attack). `train`/`test` are dicts from tokenize.make_xy.

  M1 LSTM   supervised LSTM classifier over ID embeddings (MUVIDS/GALAXY family)
  M2 CNN    supervised 1D-CNN over ID embeddings
  M3 AE     one-class LSTM autoencoder, trained on normal windows only
  M4 NGRAM  unigram+bigram counts + logistic regression (deterministic control)
  R1 RATE   trivial: |log window duration - median of normal train| (normal only)
  R2 HIST   trivial: L1 distance of window ID histogram to the mean normal histogram
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from torch import nn

torch.set_num_threads(4)


class _Seq(nn.Module):
    def __init__(self, vocab_size, kind, emb=32, hidden=64, timing=False):
        super().__init__()
        self.kind, self.timing = kind, timing
        self.emb = nn.Embedding(vocab_size, emb)
        if kind == "lstm":
            self.enc = nn.LSTM(emb + int(timing), hidden, num_layers=2, batch_first=True)
        else:
            self.enc = nn.Sequential(nn.Conv1d(emb, hidden, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, dt=None):
        e = self.emb(x)
        if self.timing:
            e = torch.cat([e, dt.unsqueeze(-1)], -1)
        if self.kind == "lstm":
            h, _ = self.enc(e)
            z = h[:, -1]
        else:
            z = self.enc(e.transpose(1, 2)).amax(-1)
        return self.out(z).squeeze(-1)


class _AE(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.enc = nn.LSTM(emb, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        _, (h, _) = self.enc(self.emb(x))
        z = h[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        d, _ = self.dec(z)
        return self.out(d)  # [B, L, V]


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class TorchDetector:
    """timing: None (IDs only), "raw" (log inter-arrival time as an extra input) or "zscore"
    (log inter-arrival z-scored with the statistics of the campaign being scored, i.e. train
    stats at fit time and test stats at score time; unlabelled test data only)."""

    def __init__(self, kind, vocab_size, seed=0, epochs=5, bs=256, lr=1e-3, timing=None):
        self.kind, self.V, self.seed, self.epochs, self.bs, self.lr = kind, vocab_size, seed, epochs, bs, lr
        self.timing = timing
        self.name = {"lstm": "M1_LSTM", "cnn": "M2_CNN", "ae": "M3_AE"}[kind] + \
            {None: "", "raw": "+dt", "zscore": "+dtz"}[timing]

    def _dt(self, xy):
        if self.timing is None:
            return None
        d = xy["dt"]
        if self.timing == "zscore":
            d = (d - d.mean()) / (d.std() + 1e-6)
        return torch.from_numpy(d.astype(np.float32))

    def fit(self, tr):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X, y = tr["X"], tr["y"]
        if self.kind == "ae":
            X, y = X[y == 0], y[y == 0]
            self.net = _AE(self.V)
            loss_fn = nn.CrossEntropyLoss()
        else:
            self.net = _Seq(self.V, self.kind, timing=self.timing is not None)
            pw = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        Dt = self._dt(tr)
        self.net.train()
        for _ in range(self.epochs):
            for b in _batches(len(X), self.bs, rng):
                opt.zero_grad()
                out = self.net(Xt[b]) if Dt is None else self.net(Xt[b], Dt[b])
                loss = loss_fn(out.reshape(-1, self.V), Xt[b].reshape(-1)) if self.kind == "ae" \
                    else loss_fn(out, yt[b])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, te):
        self.net.eval()
        X = torch.from_numpy(te["X"])
        Dt = self._dt(te)
        out = []
        for i in range(0, len(X), 2048):
            xb = X[i:i + 2048]
            if self.kind == "ae":
                lp = torch.log_softmax(self.net(xb), -1)
                out.append(-lp.gather(-1, xb.unsqueeze(-1)).squeeze(-1).mean(1))
            else:
                out.append(self.net(xb) if Dt is None else self.net(xb, Dt[i:i + 2048]))
        return torch.cat(out).numpy()


class NGram:
    name = "M4_NGRAM"

    def __init__(self, vocab_size, seed=0):
        self.V = vocab_size

    def _feats(self, X):
        n, L = X.shape
        uni = sparse.csr_matrix((np.ones(n * L), (np.repeat(np.arange(n), L), X.ravel())),
                                shape=(n, self.V))
        bi_ids = (X[:, :-1] * self.V + X[:, 1:]).ravel()
        bi = sparse.csr_matrix((np.ones(len(bi_ids)), (np.repeat(np.arange(n), L - 1), bi_ids)),
                               shape=(n, self.V * self.V))
        f = sparse.hstack([uni, bi]).tocsr()
        return f.multiply(1.0 / L).tocsr()

    def fit(self, tr):
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        self.lr.fit(self._feats(tr["X"]), tr["y"])
        return self

    def score(self, te):
        return self.lr.decision_function(self._feats(te["X"]))


class Rate:
    """Trivial: how far the window's duration is from normal (fitted on normal windows)."""
    name = "R1_RATE"

    def __init__(self, *a, **k):
        pass

    def fit(self, tr):
        d = np.log(np.maximum(tr["dur"][tr["y"] == 0], 1e-6))
        self.med = np.median(d)
        self.mad = np.median(np.abs(d - self.med)) + 1e-6
        return self

    def score(self, te):
        return np.abs(np.log(np.maximum(te["dur"], 1e-6)) - self.med) / self.mad


class Hist:
    """Trivial: L1 distance between the window's ID histogram and the mean normal histogram."""
    name = "R2_HIST"

    def __init__(self, vocab_size, *a, **k):
        self.V = vocab_size

    def _h(self, X):
        h = np.zeros((len(X), self.V))
        np.add.at(h, (np.repeat(np.arange(len(X)), X.shape[1]), X.ravel()), 1.0)
        return h / X.shape[1]

    def fit(self, tr):
        self.mu = self._h(tr["X"][tr["y"] == 0]).mean(0)
        return self

    def score(self, te):
        return np.abs(self._h(te["X"]) - self.mu).sum(1)


def make(name, vocab_size, seed=0, epochs=5):
    if name == "M1_LSTM":
        return TorchDetector("lstm", vocab_size, seed, epochs)
    if name == "M1_LSTM+dt":
        return TorchDetector("lstm", vocab_size, seed, epochs, timing="raw")
    if name == "M1_LSTM+dtz":
        return TorchDetector("lstm", vocab_size, seed, epochs, timing="zscore")
    if name == "M2_CNN":
        return TorchDetector("cnn", vocab_size, seed, epochs)
    if name == "M3_AE":
        return TorchDetector("ae", vocab_size, seed, epochs)
    return {"M4_NGRAM": NGram, "R1_RATE": Rate, "R2_HIST": Hist}[name](vocab_size)


STOCHASTIC = {"M1_LSTM", "M2_CNN", "M3_AE", "M1_LSTM+dt", "M1_LSTM+dtz"}
TIMING = {"M1_LSTM+dt", "M1_LSTM+dtz"}
ALL = ["M1_LSTM", "M2_CNN", "M3_AE", "M4_NGRAM", "R1_RATE", "R2_HIST"]
