"""Deep detectors on ROAD and ROAD-IR (GPU).

  --mode ctx   context-consistency Transformer (proposed), ambient-only training
  --mode pae   payload autoencoder (no context), ambient-only training
  --mode sup   supervised window classifier, leave-one-capture-out over the 13 masquerade
               captures; trained on ROAD (or ROAD-IR with --train-ir) and tested on the held-out
               capture in both ROAD and ROAD-IR form

python experiments/run_deep.py --mode ctx --seeds 3 --out results/deep_ctx.csv
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import busstate, deep  # noqa: E402
from src.evaluate import (bootstrap_auc_ci, make_windows, metrics, threshold_at_fpr,  # noqa: E402
                          window_labels, window_scores)
from src.io_road import load_captures_cfg  # noqa: E402
from src.pipeline import load_experiment_cfg, split_ambient  # noqa: E402

L_S, STRIDE_S = 1.0, 0.25


def windows_eval(t, step_score, interval):
    w = make_windows(0.0, float(t[-1]), L_S, STRIDE_S)
    return window_scores(t, step_score, w, "mean"), window_labels(w, interval)


def run_unsup(mode, seeds, iters, out, seed_start=0):
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    fit, cal = split_ambient(caps, exp)
    aids = np.load(busstate.INTERIM / "bus_aids.npy")
    tr, va = [], []
    for n in fit:
        _, S, _ = busstate.load(n)
        k = int(len(S) * 0.9)
        tr.append(S[:k]); va.append(S[k:])
    masq = [c for c in caps["attacks"] if c["family"] == "masquerade"]
    rows, attrib = [], []
    for seed in range(seed_start, seed_start + seeds):
        t0 = time.time()
        model = deep.train_ctx(tr, va, len(aids), context=(mode == "ctx"), iters=iters, seed=seed,
                               log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), ROOT / f"results/models/{mode}_seed{seed}.pt")
        print(f"[{mode} seed {seed}] trained in {time.time() - t0:.0f}s", flush=True)
        V = np.concatenate([deep.score_ctx(model, S, seed=seed) for S in va])
        scorer = deep.ByteScorer(V)
        print(f"  predictable bytes kept: {int(scorer.keep.sum())}/{scorer.keep.size}", flush=True)

        def step_scores(S):
            nll = deep.score_ctx(model, S, seed=seed)
            z_aid = ((nll - scorer.mu.reshape(-1, 8)) / scorer.sd.reshape(-1, 8))
            z_aid = np.where(scorer.keep.reshape(-1, 8), z_aid, -np.inf).max(2)  # per-AID max over kept bytes
            return scorer(nll), np.where(np.isfinite(z_aid), z_aid, 0.0)

        cal_w = []
        for n in cal:
            t, S, _ = busstate.load(n)
            s, _ = step_scores(S)
            cal_w.append(windows_eval(t, s, None)[0])
        thr = threshold_at_fpr(np.concatenate(cal_w), exp["target_fpr"])
        for c in masq:
            iv = (c["interval_start_s"], c["interval_end_s"])
            for variant, name in (("ROAD", c["name"]), ("ROAD-IR", c["name"] + "__IR"),
                                  ("TWIN", c["name"] + "__TWIN")):
                t, S, y = busstate.load(name)
                s, z = step_scores(S)
                ws, wy = windows_eval(t, s, iv)
                m = metrics(wy, ws, thr)
                m["ci_low"], m["ci_high"] = bootstrap_auc_ci(wy, ws, n_boot=2000, seed=seed)
                rows.append({"mode": mode, "seed": seed, "variant": variant, "capture": c["name"],
                             "attack_type": c["attack_type"], **m})
                if variant == "TWIN":  # no attack present: attribution undefined
                    continue
                lift = z[y == 1].mean(0) - z[y == 0].mean(0)  # which AID's error rises during the attack
                top = int(np.argmax(lift))
                attrib.append({"mode": mode, "seed": seed, "variant": variant, "capture": c["name"],
                               "target_aid": c["target_aid"], "top_aid": int(aids[top]),
                               "target_rank": int((lift > lift[list(aids).index(c["target_aid"])]).sum()) + 1,
                               "top_lift": float(lift[top])})
        pd.DataFrame(rows).to_csv(out, index=False)
        pd.DataFrame(attrib).to_csv(pathlib.Path(out).with_name(f"deep_{mode}_attribution.csv"), index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["variant", "attack_type"])[["auc_roc", "det_rate_at_1pct_fpr"]].mean().round(3).to_string())


def run_sup(seeds, train_ir, out):
    caps = load_captures_cfg()
    aids = np.load(busstate.INTERIM / "bus_aids.npy")
    masq = [c for c in caps["attacks"] if c["family"] == "masquerade"]
    data = {}
    for c in masq:
        for suf in ("", "__IR", "__TWIN"):
            t, S, y = busstate.load(c["name"] + suf)
            data[c["name"] + suf] = deep.window_tensor(S, y, t)
    rows = []
    for seed in range(seeds):
        for held in masq:
            trn = [c["name"] + ("__IR" if train_ir else "") for c in masq if c["name"] != held["name"]]
            X = np.concatenate([data[n][0] for n in trn]); Y = np.concatenate([data[n][1] for n in trn])
            model = deep.train_sup(X, Y, len(aids), seed=seed)
            for variant, suf in (("ROAD", ""), ("ROAD-IR", "__IR"), ("TWIN", "__TWIN")):
                Xt, Yt = data[held["name"] + suf]
                m = metrics(Yt.astype(int), deep.score_sup(model, Xt))
                rows.append({"mode": "sup", "train": "ROAD-IR" if train_ir else "ROAD", "seed": seed,
                             "variant": variant, "capture": held["name"],
                             "attack_type": held["attack_type"], **m})
            print(f"[sup seed {seed}] held out {held['name']}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["variant", "attack_type"])["auc_roc"].mean().round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ctx", "pae", "sup"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--train-ir", action="store_true")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    (ROOT / "results/models").mkdir(parents=True, exist_ok=True)
    print("device:", deep.DEV, torch.cuda.get_device_name(0) if deep.DEV == "cuda" else "", flush=True)
    if args.mode == "sup":
        run_sup(args.seeds, args.train_ir, args.out)
    else:
        run_unsup(args.mode, args.seeds, args.iters, args.out, args.seed_start)


if __name__ == "__main__":
    main()
