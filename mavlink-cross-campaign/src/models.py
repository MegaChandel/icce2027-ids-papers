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


# --------------------------------------------------------------------------------------------
# GPU models added for the extended study: faithful GALAXY, a Transformer encoder, and DANN.
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Galaxy(nn.Module):
    """GALAXY (Lee & Kim, PRDC 2025, Sec. V-D, Table VI): embedding 32 -> single LSTM (128) ->
    additive attention whose query q_t = tanh(W_q s_t + b_q) comes from local statistics
    s_t = [message-ID entropy, mean inter-arrival] over the preceding 30 messages;
    e_t = v^T tanh(h_t + q_t) + b; alpha = softmax(e); c = sum alpha_t h_t ->
    dense 128 (ReLU) -> dropout 0.3 -> sigmoid."""

    def __init__(self, V, emb=32, hid=128):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.q = nn.Linear(2, hid)
        self.v = nn.Linear(hid, 1)
        self.head = nn.Sequential(nn.Linear(hid, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x, st):
        h, _ = self.lstm(self.emb(x))
        e = self.v(torch.tanh(h + torch.tanh(self.q(st)))).squeeze(-1)
        a = torch.softmax(e, 1)
        return self.head((a.unsqueeze(-1) * h).sum(1)).squeeze(-1)


class _TF(nn.Module):
    def __init__(self, V, d=64, L=512):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.zeros(1, L + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(d, 1)

    def forward(self, x, st=None):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.emb(x)], 1)
        h = self.enc(h + self.pos[:, : h.shape[1]])
        return self.head(h[:, 0]).squeeze(-1)


class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class _DANN(nn.Module):
    """LSTM feature extractor with a label head and a campaign (domain) head behind a gradient
    reversal layer (Ganin & Lempitsky, 2015)."""

    def __init__(self, V, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.lstm = nn.LSTM(emb, hid, num_layers=2, batch_first=True)
        self.label = nn.Linear(hid, 1)
        self.domain = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def features(self, x):
        h, _ = self.lstm(self.emb(x))
        return h[:, -1]

    def forward(self, x, st=None):
        return self.label(self.features(x)).squeeze(-1)


class GPUDetector:
    """Supervised GPU detector: GALAXY (paper hyper-parameters), TF, or DANN.
    DANN additionally needs `target_unlabeled` (windows of the target campaign, no labels)."""

    def __init__(self, kind, V, seed=0, epochs=10, target_unlabeled=None):
        self.kind, self.V, self.seed, self.epochs = kind, V, seed, epochs
        self.tgt = target_unlabeled
        self.name = {"galaxy": "GALAXY", "tf": "TF", "dann": "DANN_LSTM"}[kind]

    def _st(self, xy):
        if self.kind != "galaxy":
            return None
        return torch.from_numpy(((xy["stats"] - self.st_mu) / self.st_sd).astype(np.float32))

    def fit(self, tr):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X, y = torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"].astype(np.float32))
        if self.kind == "galaxy":
            self.st_mu = tr["stats"].reshape(-1, 2).mean(0)
            self.st_sd = tr["stats"].reshape(-1, 2).std(0) + 1e-6
            self.net = _Galaxy(self.V).to(DEV)
            opt = torch.optim.RMSprop(self.net.parameters(), lr=1e-4)
            bs, lossf = 64, nn.BCEWithLogitsLoss()
        else:
            self.net = (_TF(self.V, L=X.shape[1]) if self.kind == "tf" else _DANN(self.V)).to(DEV)
            opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
            pw = torch.tensor((tr["y"] == 0).sum() / max((tr["y"] == 1).sum(), 1), device=DEV, dtype=torch.float32)
            bs, lossf = 256, nn.BCEWithLogitsLoss(pos_weight=pw)
        ST = self._st(tr)
        T = torch.from_numpy(self.tgt["X"]) if self.kind == "dann" else None
        steps = self.epochs * int(np.ceil(len(X) / bs))
        step = 0
        for _ in range(self.epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(len(X)), int(np.ceil(len(X) / bs))):
                xb = X[b].to(DEV)
                if self.kind == "dann":
                    lam = 2.0 / (1 + np.exp(-10 * step / steps)) - 1  # standard DANN schedule
                    tb = T[rng.integers(0, len(T), len(b))].to(DEV)
                    f_s, f_t = self.net.features(xb), self.net.features(tb)
                    d = self.net.domain(_GradRev.apply(torch.cat([f_s, f_t]), lam)).squeeze(-1)
                    dl = nn.functional.binary_cross_entropy_with_logits(
                        d, torch.cat([torch.zeros(len(b)), torch.ones(len(b))]).to(DEV))
                    loss = lossf(self.net.label(f_s).squeeze(-1), y[b].to(DEV)) + dl
                else:
                    out = self.net(xb, ST[b].to(DEV) if ST is not None else None)
                    loss = lossf(out, y[b].to(DEV))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
        return self

    @torch.no_grad()
    def score(self, te):
        self.net.eval()
        X, ST = torch.from_numpy(te["X"]), self._st(te)
        out = []
        for i in range(0, len(X), 2048):
            out.append(self.net(X[i:i + 2048].to(DEV),
                                ST[i:i + 2048].to(DEV) if ST is not None else None).float().cpu())
        return torch.cat(out).numpy()


_make_cpu = make


def make(name, vocab_size, seed=0, epochs=5, target_unlabeled=None):  # noqa: F811
    if name == "GALAXY":
        return GPUDetector("galaxy", vocab_size, seed, epochs=10)
    if name == "TF":
        return GPUDetector("tf", vocab_size, seed, epochs=10)
    if name == "DANN_LSTM":
        return GPUDetector("dann", vocab_size, seed, epochs=10, target_unlabeled=target_unlabeled)
    return _make_cpu(name, vocab_size, seed, epochs)


STOCHASTIC = STOCHASTIC | {"GALAXY", "TF", "DANN_LSTM"}
NEEDS_STATS = {"GALAXY"}
