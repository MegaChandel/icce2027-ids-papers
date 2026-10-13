"""Compare scoring rules for the context model offline (all statistics from ambient only).

  busmax_z     max over predictable (AID, byte) of z-scored NLL (bus-wide alarm)
  busmax_tail  max over predictable bytes of the ambient tail probability, -log10(1 - ECDF)
  perid_tail   per-ID detector: the attacked AID's own tail score (one detector per CAN ID,
               evaluated on the ID under attack, as in per-ID IDS deployments)
  perid_delta  per-ID change detector: that tail score minus its median over the previous 10 s
  bus_calib    bus-wide alarm needing no knowledge of the attacked ID: each AID's window score is
               turned into an ambient tail probability using the AID's own window-score ECDF on
               held-out fit-ambient traffic (val), and the alarm is the max over all 64 AIDs
Thresholds for TPR@1%FPR come from the two held-out ambient captures, per rule (and per AID for
the per-ID rules). TWIN = the same drive with no attack: AUC there should stay near 0.5.

python experiments/ctx_scoring.py --mode ctx --seed 0 --out results/ctx_scoring_s0.csv
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import busstate  # noqa: E402
from src.evaluate import make_windows, metrics, threshold_at_fpr, window_labels, window_scores  # noqa: E402
from src.io_road import load_captures_cfg  # noqa: E402
from src.pipeline import load_experiment_cfg, split_ambient  # noqa: E402

HZ, CHANGE_S = 50, 10.0


class Tail:
    def __init__(self, val, max_nll=1.0, q=2001):
        flat = val.reshape(len(val), -1).astype(np.float32)
        self.keep = flat.mean(0) < max_nll
        self.qs = np.quantile(flat, np.linspace(0, 1, q), axis=0)  # [q, A*8]
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-3

    def tail(self, nll):
        flat = nll.reshape(len(nll), -1).astype(np.float32)
        out = np.empty_like(flat)
        for j in range(flat.shape[1]):
            out[:, j] = np.searchsorted(self.qs[:, j], flat[:, j], side="right") / self.qs.shape[0]
        return -np.log10(np.clip(1 - out, 1e-4, 1))

    def z(self, nll):
        return (nll.reshape(len(nll), -1).astype(np.float32) - self.mu) / self.sd


def per_aid_max(scores, keep, A):
    s = np.where(keep, scores, -np.inf).reshape(len(scores), A, 8).max(2)
    return np.where(np.isfinite(s), s, 0.0)


def change(x, hz=HZ, span=CHANGE_S):
    """x [T] -> x - rolling median of the previous `span` seconds (strided for speed)."""
    k, st = int(span * hz), 10
    base = np.empty_like(x)
    for i in range(0, len(x), st):
        lo = max(0, i - k)
        base[i:i + st] = np.median(x[lo:i]) if i > 0 else x[0]
    return x - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ctx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = ROOT / f"results/nll/{args.mode}_s{args.seed}"
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    _, cal = split_ambient(caps, exp)
    aids = list(np.load(busstate.INTERIM / "bus_aids.npy"))
    A = len(aids)
    T = Tail(np.load(d / "val.npy"))

    def step_rules(name, target=None):
        nll = np.load(d / f"{name}.npy")
        ty = np.load(d / f"{name}.ty.npz")
        tl, z = T.tail(nll), T.z(nll)
        r = {"busmax_z": np.where(T.keep, z, -np.inf).max(1), "busmax_tail": np.where(T.keep, tl, -np.inf).max(1)}
        pa = per_aid_max(tl, T.keep, A)
        return ty["t"], ty["y"], r, pa

    def win(t, s, iv):
        w = make_windows(0.0, float(t[-1]), 1.0, 0.25)
        return window_scores(t, s, w, "mean"), window_labels(w, iv)

    # per-AID window-score ECDF on val (held-out fit ambient), for the bus_calib rule
    vnll = np.load(d / "val.npy")
    vt = np.arange(len(vnll)) / HZ
    vpa = per_aid_max(T.tail(vnll), T.keep, A)
    vw = make_windows(0.0, float(vt[-1]), 1.0, 0.25)
    val_win = np.stack([window_scores(vt, vpa[:, a], vw, "mean") for a in range(A)], 1)  # [W, A]
    val_sorted = np.sort(val_win, 0)

    def bus_calib(t, pa):
        w = make_windows(0.0, float(t[-1]), 1.0, 0.25)
        ws = np.stack([window_scores(t, pa[:, a], w, "mean") for a in range(A)], 1)
        F = np.stack([np.searchsorted(val_sorted[:, a], ws[:, a], side="right") for a in range(A)], 1) / len(val_sorted)
        return -np.log10(np.clip(1 - F, 1.0 / len(val_sorted), 1)).max(1), w

    # calibration: bus-wide rules, and per-AID rules for every AID
    cal_bus, cal_aid, cal_aid_d = {}, {a: [] for a in range(A)}, {a: [] for a in range(A)}
    for n in cal:
        t, _, r, pa = step_rules(n)
        for k, s in r.items():
            cal_bus.setdefault(k, []).append(win(t, s, None)[0])
        cal_bus.setdefault("bus_calib", []).append(bus_calib(t, pa)[0])
        for a in range(A):
            cal_aid[a].append(win(t, pa[:, a], None)[0])
            cal_aid_d[a].append(win(t, change(pa[:, a]), None)[0])
    thr_bus = {k: threshold_at_fpr(np.concatenate(v), exp["target_fpr"]) for k, v in cal_bus.items()}
    thr_aid = {a: threshold_at_fpr(np.concatenate(v), exp["target_fpr"]) for a, v in cal_aid.items()}
    thr_aid_d = {a: threshold_at_fpr(np.concatenate(v), exp["target_fpr"]) for a, v in cal_aid_d.items()}

    rows = []
    for c in caps["attacks"]:
        if c["family"] != "masquerade":
            continue
        iv = (c["interval_start_s"], c["interval_end_s"])
        ai = aids.index(c["target_aid"])
        for variant, suf in (("ROAD", ""), ("ROAD-IR", "__IR"), ("TWIN", "__TWIN")):
            t, _, r, pa = step_rules(c["name"] + suf)
            cand = {**{k: (s, thr_bus[k]) for k, s in r.items()},
                    "perid_tail": (pa[:, ai], thr_aid[ai]), "perid_delta": (change(pa[:, ai]), thr_aid_d[ai])}
            bc, bw = bus_calib(t, pa)
            rows.append({"mode": args.mode, "seed": args.seed, "rule": "bus_calib", "variant": variant,
                         "capture": c["name"], "attack_type": c["attack_type"],
                         **metrics(window_labels(bw, iv), bc, thr_bus["bus_calib"])})
            for rule, (s, thr) in cand.items():
                ws, wy = win(t, s, iv)
                rows.append({"mode": args.mode, "seed": args.seed, "rule": rule, "variant": variant,
                             "capture": c["name"], "attack_type": c["attack_type"], **metrics(wy, ws, thr)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.pivot_table(index=["rule", "attack_type"], columns="variant", values="auc_roc").round(2).to_string())
    print(df.pivot_table(index=["rule", "attack_type"], columns="variant", values="det_rate_at_1pct_fpr").round(2).to_string())


if __name__ == "__main__":
    main()
