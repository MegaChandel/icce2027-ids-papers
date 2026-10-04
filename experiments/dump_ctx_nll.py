"""Score every capture once with a trained context model and store per-byte NLL, so that
scoring rules can be compared offline without retraining.

python experiments/dump_ctx_nll.py --mode ctx --seed 0
-> results/nll/<mode>_s<seed>/<capture>.npy  (float16 [T, A, 8]) and val.npy, plus t/y per capture
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import busstate, deep  # noqa: E402
from src.io_road import load_captures_cfg  # noqa: E402
from src.pipeline import load_experiment_cfg, split_ambient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ctx")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    fit, cal = split_ambient(caps, exp)
    aids = np.load(busstate.INTERIM / "bus_aids.npy")
    model = deep.CTX(len(aids), 10, context=args.mode == "ctx").to(deep.DEV)
    model.load_state_dict(torch.load(ROOT / f"results/models/{args.mode}_seed{args.seed}.pt", map_location=deep.DEV))
    out = ROOT / f"results/nll/{args.mode}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    val = []
    for n in fit:
        _, S, _ = busstate.load(n)
        val.append(deep.score_ctx(model, S[int(len(S) * 0.9):], seed=args.seed))
    np.save(out / "val.npy", np.concatenate(val).astype(np.float16))
    names = list(cal)
    for c in caps["attacks"]:
        if c["family"] == "masquerade":
            names += [c["name"], c["name"] + "__IR", c["name"] + "__TWIN"]
    for n in names:
        t, S, y = busstate.load(n)
        np.save(out / f"{n}.npy", deep.score_ctx(model, S, seed=args.seed).astype(np.float16))
        np.savez(out / f"{n}.ty.npz", t=t, y=y)
        print(n, flush=True)


if __name__ == "__main__":
    main()
