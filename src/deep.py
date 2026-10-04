"""Deep detectors on the bus-state representation (GPU).

CTX  Context-consistency Transformer (proposed). Each selected AID is one token whose
     features are the bits and normalised values of its last K steps. A random subset of
     tokens is masked (features zeroed + learned mask vector) and the model predicts the
     masked AIDs' current 8 bytes (8 x 256-way) from the other AIDs only. Trained on ambient
     only. Score per AID = NLL of the observed bytes when that AID is masked, z-scored on
     held-out ambient; step score = max over AIDs.
PAE  Payload autoencoder (deep baseline without context). Same tokens, but each token only
     sees itself through a 16-d bottleneck; score = reconstruction NLL, z-scored, max over AIDs.
SUP  Supervised window classifier (shortcut test). Token encoder per step, mean over AIDs,
     GRU over steps, logit. Trained on labelled attack captures.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BITS = torch.tensor([1 << i for i in range(8)], dtype=torch.uint8)


def featurize(x_u8: torch.Tensor) -> torch.Tensor:
    """x: uint8 [..., K, A, 8] -> float [..., A, K*(64+8)] (bits + normalised bytes per step)."""
    bits = (x_u8.unsqueeze(-1) & BITS.to(x_u8.device)).ne(0).float()  # [..., K, A, 8, 8]
    bits = bits.flatten(-2)                                            # [..., K, A, 64]
    val = x_u8.float() / 255.0                                         # [..., K, A, 8]
    f = torch.cat([bits, val], -1)                                     # [..., K, A, 72]
    f = f.movedim(-3, -2)                                              # [..., A, K, 72]
    return f.flatten(-2)


class ByteScorer:
    """Turns per-byte NLL [T, A, 8] into a step score using held-out AMBIENT statistics only.
    Bytes the model cannot predict on held-out ambient traffic (counters, checksums, noisy
    sensors: mean NLL > `max_nll` nats) are excluded; the rest are z-scored per (AID, byte)
    and the step score is the maximum z."""

    def __init__(self, val_nll, max_nll=1.0):
        flat = val_nll.reshape(-1, val_nll.shape[1] * 8)
        self.mu = flat.mean(0)
        self.sd = flat.std(0) + 1e-3
        self.keep = self.mu < max_nll

    def z(self, nll):
        flat = nll.reshape(len(nll), -1)
        return ((flat - self.mu) / self.sd)[:, self.keep]

    def __call__(self, nll):
        return self.z(nll).max(1)


class CTX(nn.Module):
    def __init__(self, n_aid, K, d=192, layers=4, heads=6, context=True):
        super().__init__()
        self.context = context
        self.inp = nn.Linear(K * 72, d)
        self.aid = nn.Embedding(n_aid, d)
        self.mask_vec = nn.Parameter(torch.zeros(d))
        if context:
            enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
        else:  # PAE: per-token bottleneck, no information flow between AIDs
            self.enc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 16), nn.GELU(),
                                     nn.Linear(16, d), nn.GELU())
        self.out = nn.Linear(d, 8 * 256)

    def forward(self, x_u8, mask):
        """x_u8 [B, K, A, 8]; mask bool [B, A] (True = hidden). Returns logits [B, A, 8, 256]."""
        B, K, A, _ = x_u8.shape
        h = self.inp(featurize(x_u8))
        if self.context:
            h = torch.where(mask.unsqueeze(-1), self.mask_vec.expand_as(h), h)
        h = h + self.aid.weight.unsqueeze(0)
        h = self.enc(h)
        return self.out(h).view(B, A, 8, 256)


def nll_per_byte(logits, target_u8):
    """logits [B, A, 8, 256], target uint8 [B, A, 8] -> NLL per byte [B, A, 8]."""
    lp = torch.log_softmax(logits.float(), -1)
    return -lp.gather(-1, target_u8.long().unsqueeze(-1)).squeeze(-1)


def nll_per_aid(logits, target_u8):
    return nll_per_byte(logits, target_u8).sum(-1)


class Windows:
    """Random-access K-step contexts over a list of bus-state arrays."""

    def __init__(self, arrays, K, stride=1):
        self.arrays, self.K = arrays, K
        self.index = [(i, t) for i, S in enumerate(arrays) for t in range(K - 1, len(S), stride)]

    def __len__(self):
        return len(self.index)

    def batch(self, idx):
        xs = [self.arrays[i][t - self.K + 1: t + 1] for i, t in (self.index[j] for j in idx)]
        return torch.from_numpy(np.stack(xs))


def train_ctx(train_arrays, val_arrays, n_aid, K=10, context=True, iters=20000, bs=256, lr=3e-4,
              seed=0, max_mask=8, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CTX(n_aid, K, context=context).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.05)
    tr, va = Windows(train_arrays, K), Windows(val_arrays, K, stride=25)
    rng = np.random.default_rng(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    for it in range(iters):
        model.train()
        x = tr.batch(rng.integers(0, len(tr), bs)).to(DEV, non_blocking=True)
        if context:
            m = torch.zeros(bs, n_aid, dtype=torch.bool, device=DEV)
            k = torch.randint(1, max_mask + 1, (bs,), device=DEV)
            r = torch.rand(bs, n_aid, device=DEV).argsort(1)
            m = r < k.unsqueeze(1)
        else:
            m = torch.ones(bs, n_aid, dtype=torch.bool, device=DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = model(x, m)
        loss = (nll_per_aid(logits, x[:, -1]) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if (it + 1) % 2000 == 0:
            log(f"  it {it + 1}/{iters} train nll/aid {loss.item():.3f}")
    return model


@torch.no_grad()
def score_ctx(model, S, K=10, groups=8, bs=512, seed=0):
    """Per-step, per-AID, per-byte NLL with each AID masked inside a fixed random partition of
    `groups` groups. Returns float32 [T, A, 8] (first K-1 steps padded with the first value)."""
    model.eval()
    A = S.shape[1]
    part = np.random.default_rng(seed).permutation(A) % groups if model.context else np.zeros(A, int)
    G = groups if model.context else 1
    W = Windows([S], K)
    out = np.zeros((len(W), A, 8), np.float32)
    for s in range(0, len(W), bs):
        idx = np.arange(s, min(s + bs, len(W)))
        x = W.batch(idx).to(DEV)
        for g in range(G):
            m = torch.from_numpy(part == g).to(DEV).unsqueeze(0).expand(len(idx), A)
            if not model.context:
                m = torch.ones_like(m)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                nll = nll_per_byte(model(x, m), x[:, -1])
            sel = torch.from_numpy(part == g).to(DEV) if model.context else torch.ones(A, dtype=torch.bool, device=DEV)
            out[idx[:, None], np.flatnonzero(sel.cpu().numpy())[None, :]] = nll[:, sel].float().cpu().numpy()
    pad = np.repeat(out[:1], K - 1, 0)
    return np.concatenate([pad, out])


class SUP(nn.Module):
    def __init__(self, n_aid, d=128):
        super().__init__()
        self.inp = nn.Linear(72, d)
        self.aid = nn.Embedding(n_aid, d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tok = nn.TransformerEncoder(enc, 2)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, x_u8):
        """x [B, T, A, 8] -> logit [B]."""
        B, T, A, _ = x_u8.shape
        f = featurize(x_u8.reshape(B * T, 1, A, 8))           # [B*T, A, 72]
        h = self.tok(self.inp(f) + self.aid.weight.unsqueeze(0)).mean(1).view(B, T, -1)
        _, z = self.gru(h)
        return self.head(z[-1]).squeeze(-1)


def window_tensor(S, y, t, length_s=1.0, stride_s=0.25, sub=5, hz=50):
    """Bus-state windows (subsampled by `sub` in time) with overlap labels."""
    L, st = int(length_s * hz), int(stride_s * hz)
    starts = np.arange(0, len(S) - L + 1, st)
    X = np.stack([S[s:s + L:sub] for s in starts])
    Y = np.array([y[s:s + L].max() for s in starts], np.float32)
    return X, Y


def train_sup(X, Y, n_aid, epochs=15, bs=128, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SUP(n_aid).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pw = torch.tensor((Y == 0).sum() / max((Y == 1).sum(), 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(X)), math.ceil(len(X) / bs)):
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = lossf(model(Xt[b].to(DEV)).float(), Yt[b].to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


@torch.no_grad()
def score_sup(model, X, bs=256):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            out.append(model(torch.from_numpy(X[s:s + bs]).to(DEV)).float().cpu().numpy())
    return np.concatenate(out)
