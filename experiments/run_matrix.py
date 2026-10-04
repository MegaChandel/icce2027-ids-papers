"""Experiment (b): within- vs cross-campaign transfer matrix.

Within and cross settings that share a target campaign are scored on the SAME test split,
so the gap Within-B - Cross(A->B) compares like with like.

python experiments/run_matrix.py --seeds 5 --out results/matrix.csv
"""
import argparse
import itertools
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import ALL, STOCHASTIC, TIMING, make  # noqa: E402
from src.tokenize import build_vocab, make_xy, split  # noqa: E402

SHARED = {"normal", "heartbeat", "ping", "request"}  # attack types present in OLD and NEW


def parts(campaign):
    tr, te = split(campaign)
    return tr, te


def settings():
    o_tr, o_te = parts("OLD")
    n_tr, n_te = parts("NEW")
    s_tr, s_te = parts("SHADOW")
    return {
        "Within-OLD": (o_tr, o_te, SHARED),
        "Within-NEW": (n_tr, n_te, SHARED),
        "OLD->NEW": (o_tr + o_te, n_te, SHARED),
        "NEW->OLD": (n_tr + n_te, o_te, SHARED),
        "Within-SHADOW": (s_tr, s_te, None),
        "HCRL->SHADOW": (o_tr + o_te + n_tr + n_te, s_te, None),
        "SHADOW->OLD": (s_tr + s_te, o_te, None),
        "SHADOW->NEW": (s_tr + s_te, n_te, None),
    }


def metrics(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)


def run_one(job):
    setting, vocab_mode, model, seed, L, stride, epochs = job
    tr_pairs, te_pairs, keep = settings()[setting]
    if keep is not None:
        tr_pairs = [p for p in tr_pairs if p[1]["attack"] in keep]
        te_pairs = [p for p in te_pairs if p[1]["attack"] in keep]
    tr_segs = [s for s, _ in tr_pairs]
    vocab = build_vocab(tr_segs if vocab_mode == "train" else tr_segs + [s for s, _ in te_pairs])
    tr = make_xy(tr_pairs, vocab, L, stride, with_dt=model in TIMING)
    te = make_xy(te_pairs, vocab, L, stride, with_dt=model in TIMING)
    t0 = time.time()
    det = make(model, len(vocab) + 1, seed=seed, epochs=epochs).fit(tr)
    s = det.score(te)
    rows = []
    base = dict(setting=setting, vocab=vocab_mode, model=model, seed=seed, L=L, stride=stride,
                oov_rate=float(te["oov"].mean()), fit_s=round(time.time() - t0, 1))
    auc, ap = metrics(te["y"], s)
    rows.append({**base, "attack": "all", "auc_roc": auc, "pr_auc": ap, "n_test": len(s)})
    norm = te["attack"] == "normal"
    for a in sorted(set(te["attack"]) - {"normal"}):
        m = norm | (te["attack"] == a)
        auc, ap = metrics(te["y"][m], s[m])
        rows.append({**base, "attack": a, "auc_roc": auc, "pr_auc": ap, "n_test": int(m.sum()),
                     "oov_rate": float(te["oov"][te["attack"] == a].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/matrix.csv"))
    ap.add_argument("--vocab", nargs="+", default=["train", "union"])
    ap.add_argument("--settings", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    sets = args.settings or list(settings())
    jobs = [(st, v, m, sd, args.L, args.stride, args.epochs)
            for st, v, m in itertools.product(sets, args.vocab, args.models)
            for sd in (range(args.seeds) if m in STOCHASTIC else [0])]
    print(f"{len(jobs)} jobs", flush=True)
    rows, t0 = [], time.time()
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows += r
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    v = df[(df.attack == "all") & (df.vocab == "train")]
    print(v.groupby(["model", "setting"])["auc_roc"].agg(["mean", "std"]).unstack("setting")["mean"]
          .round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
