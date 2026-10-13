"""Experiment (d): sensitivity to window length, T3 percentile and amount of ambient data.

Each sweep varies one factor and keeps the others at configs/experiment.yaml defaults.
Stride is always length/4. Masquerade captures only.

python experiments/run_sensitivity.py --out results/sensitivity.csv
"""
import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.io_road import ROOT, load_captures_cfg  # noqa: E402
from src.pipeline import (attack_caps, calibrate, evaluate_capture, fit_all,  # noqa: E402
                          is_signal_det, load_experiment_cfg, split_ambient)

DETS = ("T1_valueset", "T2_range", "T3_envelope")


def run(dets, caps, cal_names, length, agg, fpr, tag, value):
    stride = length / 4
    thr, _ = calibrate(dets, cal_names, length, stride, agg, fpr)
    rows = []
    for cap in attack_caps(caps, ("masquerade",)):
        for d in dets:
            m, _ = evaluate_capture(d, cap, length, stride, agg, thr[d.name])
            rows.append({"sweep": tag, "value": value, "capture": cap["name"],
                         "attack_type": cap["attack_type"], "detector": d.name, **m})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/sensitivity.csv"))
    args = ap.parse_args()
    exp, caps = load_experiment_cfg(), load_captures_cfg()
    W, S = exp["window"], exp["sensitivity"]
    fit_names, cal_names = split_ambient(caps, exp)
    rows, t0 = [], time.time()

    base = [d for d in fit_all(fit_names, t3_q=(0.0,)) if d.name in DETS]
    for L in S["window_length_s"]:
        rows += run(base, caps, cal_names, L, W["agg"], exp["target_fpr"], "window_length_s", L)
    print(f"window sweep done ({time.time() - t0:.0f}s)", flush=True)

    t3s = [d for d in fit_all(fit_names, t3_q=tuple(S["t3_quantiles"])) if is_signal_det(d)]
    for d in t3s:
        rows += run([d], caps, cal_names, W["length_s"], W["agg"], exp["target_fpr"],
                    "t3_quantile", d.q)
    print(f"t3 sweep done ({time.time() - t0:.0f}s)", flush=True)

    for frac in S["ambient_fraction"]:
        dets = [d for d in fit_all(fit_names, t3_q=(0.0,), fraction=frac) if d.name in DETS]
        rows += run(dets, caps, cal_names, W["length_s"], W["agg"], exp["target_fpr"],
                    "ambient_fraction", frac)
        print(f"  ambient fraction {frac} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    df.to_csv(out, index=False)
    print(df.groupby(["sweep", "value", "detector", "attack_type"])["auc_roc"].mean()
          .unstack("attack_type").round(2).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
