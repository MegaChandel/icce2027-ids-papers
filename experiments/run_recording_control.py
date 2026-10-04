"""Recording-identity control.

Label two BENIGN recordings of the same campaign as 0 and 1 (OLD normal_1 vs normal_2) and
train the supervised detectors under the within-file temporal split used for NEW (first 80% /
last 20% of each file). Any AUC well above 0.5 means that this split rewards telling
recordings apart, which needs no attack signal at all. The same is repeated for the NEW
heartbeat file versus the NEW normal file, and with a FILE split on OLD for contrast.

python experiments/run_recording_control.py --out results/recording_control.csv
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import STOCHASTIC, make  # noqa: E402
from src.tokenize import Segment, build_vocab, make_xy  # noqa: E402

MODELS = ["M1_LSTM", "M2_CNN", "M4_NGRAM", "R1_RATE", "R2_HIST"]


def pairs(names_labels, lo, hi):
    return [(Segment(n, lo, hi), {"label": y, "attack": "normal" if y == 0 else "x"})
            for n, y in names_labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/recording_control.csv"))
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    tasks = {
        # two benign recordings, temporal split inside each file
        "benign-vs-benign (OLD, temporal split)": (
            pairs([("old_normal_1", 0), ("old_normal_2", 1)], 0, .8),
            pairs([("old_normal_1", 0), ("old_normal_2", 1)], .8, 1)),
        # the NEW heartbeat task as evaluated in Within-NEW
        "heartbeat vs normal (NEW, temporal split)": (
            pairs([("new_normal_1", 0), ("new_heartbeat_1", 1)], 0, .8),
            pairs([("new_normal_1", 0), ("new_heartbeat_1", 1)], .8, 1)),
        # the OLD heartbeat task with a file split (train rep 1, test rep 2)
        "heartbeat vs normal (OLD, file split)": (
            pairs([("old_normal_1", 0), ("old_heartbeat_1", 1)], 0, 1),
            pairs([("old_normal_2", 0), ("old_heartbeat_2", 1)], 0, 1)),
    }
    rows = []
    for task, (tr_p, te_p) in tasks.items():
        vocab = build_vocab([s for s, _ in tr_p])
        tr, te = make_xy(tr_p, vocab, 32, 16), make_xy(te_p, vocab, 32, 16)
        for m in MODELS:
            for seed in (range(args.seeds) if m in STOCHASTIC else [0]):
                s = make(m, len(vocab) + 1, seed=seed).fit(tr).score(te)
                auc = roc_auc_score(te["y"], s)
                rows.append({"task": task, "model": m, "seed": seed, "auc_roc": auc})
        print(task, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby(["task", "model"])["auc_roc"].agg(["mean", "std"]).round(3).unstack("model")["mean"].to_string())


if __name__ == "__main__":
    main()
