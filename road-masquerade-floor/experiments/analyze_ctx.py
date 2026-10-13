"""Compare scoring rules on stored per-byte NLL of a trained model (all calibrated on ambient).

Rules
  bus_z      max over predictable (AID, byte) of the z-scored NLL           (bus-wide alarm)
  bus_tail   max over predictable (AID, byte) of the ambient tail score     (bus-wide alarm)
             tail = -log10 P_ambient(NLL >= x), from the held-out ambient distribution
  aid_tail   per-AID detector (max tail over that AID's predictable bytes), evaluated for the
             attacked AID: the per-identifier deployment used by per-ID IDS
  aid_change per-AID detector on the CHANGE of its tail score: 1 s mean minus the median of the
             preceding 10 s (reacts to onsets, cancels slow driving-phase drift)
  bus_calib  bus-wide alarm from the 64 per-AID detectors: each AID's WINDOW score is converted
             to its tail probability under that AID's held-out-ambient window distribution, and
             the window score is -log10 of the smallest one (rarest AID wins; no target knowledge).
             AIDs that are non-stationary are excluded first, using ambient data only: for a
             stationary AID the mean window tail score is about 1/ln(10) = 0.43; an AID is
             dropped if its mean window tail score on held-out ambient data (training sessions'
             held-out tail or the held-out sessions) exceeds 1.0 (odometer/clock-like fields).
Variants: ROAD, ROAD-IR (in-range counterfactual), TWIN (same drive, attack removed).

python experiments/analyze_ctx.py --mode ctx --seed 0 --out results/ctx_rules.csv
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

HZ = 50
CAL_S = 4.0  # first seconds of each drive used for session calibration (attacks start >= 4.3 s)


class Tail:
    """Per-(AID, byte) empirical tail score from held-out ambient NLL."""

    def __init__(self, val, max_nll=1.0, n_ref=20000, seed=0):
        flat = val.reshape(len(val), -1).astype(np.float32)
        self.keep = flat.mean(0) < max_nll
        rng = np.random.default_rng(seed)
        ref = flat[rng.choice(len(flat), min(n_ref, len(flat)), replace=False)]
        self.ref = np.sort(ref, 0)
        self.n = len(ref)
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-3

    def tail(self, nll):
        flat = nll.reshape(len(nll), -1).astype(np.float32)
        out = np.empty_like(flat)
        for j in range(flat.shape[1]):
            ge = self.n - np.searchsorted(self.ref[:, j], flat[:, j], side="left")
            out[:, j] = -np.log10((ge + 1) / (self.n + 1))
        return out  # [T, A*8]

    def z(self, nll):
        return (nll.reshape(len(nll), -1).astype(np.float32) - self.mu) / self.sd


def per_aid(x, keep, A):
    x = np.where(keep[None, :], x, -np.inf).reshape(len(x), A, 8).max(2)
    return np.where(np.isfinite(x), x, 0.0)


def change(s, hz=HZ, short=1.0, long=10.0):
    """1 s rolling mean minus median of the preceding 10 s (per column)."""
    df = pd.DataFrame(s)
    m = df.rolling(int(short * hz), min_periods=1).mean()
    base = df.shift(int(short * hz)).rolling(int(long * hz), min_periods=1).median()
    return (m - base.fillna(m)).to_numpy()


def windows(t, s, iv):
    w = make_windows(0.0, float(t[-1]), 1.0, 0.25)
    return window_scores(t, s, w, "mean"), window_labels(w, iv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ctx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = ROOT / f"results/nll/{args.mode}_s{args.seed}"
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    _, cal = split_ambient(caps, exp)
    aids = list(np.load(busstate.INTERIM / "bus_aids.npy"))
    A = len(aids)
    T = Tail(np.load(d / "val.npy"))
    print(f"predictable bytes: {int(T.keep.sum())}/{T.keep.size}")

    def rules(n):
        nll = np.load(d / f"{n}.npy")
        tl, z = T.tail(nll), T.z(nll)
        aid_t = per_aid(tl, T.keep, A)
        return {"bus_z": np.where(T.keep, z, -np.inf).max(1),
                "bus_tail": np.where(T.keep, tl, -np.inf).max(1),
                "aid_tail": aid_t, "aid_change": change(aid_t)}

    # thresholds at 1% FPR on held-out ambient windows (bus rules: one threshold; aid rules: per AID)
    thr = {}
    calr = [(np.load(d / f"{n}.ty.npz")["t"], rules(n)) for n in cal]
    for k in ("bus_z", "bus_tail"):
        thr[k] = threshold_at_fpr(np.concatenate([windows(t, r[k], None)[0] for t, r in calr]), 0.01)
    for k in ("aid_tail", "aid_change"):
        thr[k] = [threshold_at_fpr(np.concatenate([windows(t, r[k][:, a], None)[0] for t, r in calr]), 0.01)
                  for a in range(A)]
    # per-AID window-score reference distributions for bus_calib
    calw = np.stack([np.concatenate([windows(t, r["aid_tail"][:, a], None)[0] for t, r in calr])
                     for a in range(A)], 1)  # [Wcal, A]
    val_aid = per_aid(T.tail(np.load(d / "val.npy")), T.keep, A)
    tv = np.arange(len(val_aid)) / HZ
    valw = np.stack([windows(tv, val_aid[:, a], None)[0] for a in range(A)], 1)
    drift = np.maximum(calw.mean(0), valw.mean(0))
    stationary = drift <= 1.0
    print("non-stationary AIDs (ambient-only test):", [hex(aids[a]) for a in np.flatnonzero(~stationary)],
          "drift share:", np.round(drift[~stationary], 2).tolist())
    calw_sorted = np.sort(calw, 0)

    def bus_calib(t, aid_scores):
        w = make_windows(0.0, float(t[-1]), 1.0, 0.25)
        W = np.stack([window_scores(t, aid_scores[:, a], w, "mean") for a in range(A)], 1)
        ge = np.stack([len(calw) - np.searchsorted(calw_sorted[:, a], W[:, a], side="left") for a in range(A)], 1)
        return (-np.log10((ge + 1) / (len(calw) + 1)))[:, stationary].max(1), w

    cal_bc = np.concatenate([bus_calib(t, r["aid_tail"])[0] for t, r in calr])
    thr["bus_calib"] = threshold_at_fpr(cal_bc, 0.01)
    rows = []
    for c in caps["attacks"]:
        if c["family"] != "masquerade":
            continue
        iv = (c["interval_start_s"], c["interval_end_s"])
        a = aids.index(c["target_aid"])
        for variant, suf in (("ROAD", ""), ("ROAD-IR", "__IR"), ("TWIN", "__TWIN")):
            t = np.load(d / f"{c['name']}{suf}.ty.npz")["t"]
            r = rules(c["name"] + suf)
            for k in ("bus_z", "bus_tail", "aid_tail", "aid_change", "bus_calib"):
                if k == "bus_calib":
                    ws, w = bus_calib(t, r["aid_tail"])
                    wy = window_labels(w, iv)
                    m = metrics(wy, ws, thr[k])
                else:
                    s, th = (r[k], thr[k]) if k.startswith("bus") else (r[k][:, a], thr[k][a])
                    ws, wy = windows(t, s, iv)
                    m = metrics(wy, ws, th)
                    # session calibration: threshold = max window score in the first CAL_S seconds
                    # of the same drive (every ROAD attack starts later); rates on the remainder
                    w = make_windows(0.0, float(t[-1]), 1.0, 0.25)
                    early, rest = w[:, 1] <= CAL_S, w[:, 0] >= CAL_S
                    th_s = ws[early].max()
                    m["sess_tpr"] = float((ws[rest & (wy == 1)] > th_s).mean())
                    m["sess_fpr"] = float((ws[rest & (wy == 0)] > th_s).mean()) if (rest & (wy == 0)).any() else np.nan
                rows.append({"model": args.mode, "seed": args.seed, "rule": k, "variant": variant,
                             "capture": c["name"], "attack_type": c["attack_type"], **m})
    df = pd.DataFrame(rows)
    out = args.out or ROOT / f"results/{args.mode}_rules_s{args.seed}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["rule", "attack_type"], columns="variant", values="auc_roc").round(2).to_string())
    print(df.pivot_table(index=["rule", "attack_type"], columns="variant", values="det_rate_at_1pct_fpr").round(2).to_string())


if __name__ == "__main__":
    main()
