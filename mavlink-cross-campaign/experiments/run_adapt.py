"""Few-second target calibration (proposed deployment fix).

Assumption: the first N seconds a detector sees in a new campaign (e.g. after take-off) are
benign. One-class detectors (Rate, Histogram, LSTM-AE) are fitted on
  'source'        : the source campaign's normal training traffic (the cross setting so far),
  'target-N'      : only the first N seconds of each benign TRAIN file of the target campaign,
  'source+target-N': both,
and scored on the target campaign's usual test split (same as Within-target).

python experiments/run_adapt.py --out results/adapt.csv
"""
import argparse
import pathlib
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inventory import load  # noqa: E402
from src.models import make  # noqa: E402
from src.tokenize import Segment, build_vocab, make_xy, split  # noqa: E402

SHARED = {"normal", "heartbeat", "ping", "request"}
SETTINGS = {"OLD->NEW": (["OLD"], "NEW"), "NEW->OLD": (["NEW"], "OLD"),
            "HCRL->SHADOW": (["OLD", "NEW"], "SHADOW"), "SHADOW->OLD": (["SHADOW"], "OLD"),
            "SHADOW->NEW": (["SHADOW"], "NEW")}
MODELS = ["R1_RATE", "R2_HIST", "M3_AE"]


def keep_shared(pairs, camp):
    return [p for p in pairs if camp == "SHADOW" or p[1]["attack"] in SHARED]


def first_seconds(pairs, N):
    """First N seconds of each benign train segment."""
    out = []
    for seg, c in pairs:
        if c["label"] != 0:
            continue
        dur = load(seg.name).t.max()
        span = seg.hi - seg.lo
        hi = seg.lo + min(span, N / dur)
        out.append((Segment(seg.name, seg.lo, hi), c))
    return out


def job(args):
    setting, model, mode, N, seed = args
    srcs, tgt = SETTINGS[setting]
    src_pairs = []
    for s in srcs:
        a, b = split(s)
        src_pairs += keep_shared(a + b, s)
    t_tr, t_te = split(tgt)
    t_tr, t_te = keep_shared(t_tr, tgt), keep_shared(t_te, tgt)
    calib = first_seconds(t_tr, N) if N else []
    if mode == "source":
        fit_pairs = [p for p in src_pairs if p[1]["label"] == 0]
    elif mode == "target":
        fit_pairs = calib
    else:
        fit_pairs = [p for p in src_pairs if p[1]["label"] == 0] + calib
    vocab = build_vocab([s for s, _ in fit_pairs])
    tr = make_xy(fit_pairs, vocab, 32, 16)
    te = make_xy(t_te, vocab, 32, 16)
    s = make(model, len(vocab) + 1, seed=seed).fit(tr).score(te)
    return {"setting": setting, "model": model, "mode": mode, "N_s": N, "seed": seed,
            "n_fit_windows": len(tr["X"]), "auc_roc": roc_auc_score(te["y"], s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/adapt.csv"))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    jobs = []
    for st in SETTINGS:
        for m in MODELS:
            seeds = range(args.seeds) if m == "M3_AE" else [0]
            for sd in seeds:
                jobs.append((st, m, "source", 0, sd))
                for N in (15, 30, 60, 120):
                    jobs += [(st, m, "target", N, sd), (st, m, "source+target", N, sd)]
    print(len(jobs), "jobs", flush=True)
    with mp.get_context("spawn").Pool(8) as pool:
        rows = pool.map(job, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby(["setting", "model", "mode", "N_s"])["auc_roc"].mean().unstack(["mode", "N_s"]).round(2).to_string())


if __name__ == "__main__":
    main()
