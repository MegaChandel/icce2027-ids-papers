"""Learned control: per-AID IsolationForest on payload bytes, 3 seeds (blueprint §9.1).

python experiments/run_controls.py --out results/controls.csv
"""
import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.comparators import PerAIDIsolationForest  # noqa: E402
from src.io_road import ROOT, load_captures_cfg  # noqa: E402
from src.pipeline import (attack_caps, calibrate, evaluate_capture, load_ambient,  # noqa: E402
                          load_experiment_cfg, split_ambient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/controls.csv"))
    args = ap.parse_args()
    exp, caps = load_experiment_cfg(), load_captures_cfg()
    W, IF = exp["window"], exp["isolation_forest"]
    fit_names, cal_names = split_ambient(caps, exp)
    amb = load_ambient(fit_names)
    rows, t0 = [], time.time()
    for seed in IF["seeds"]:
        det = PerAIDIsolationForest(seed=seed, n_estimators=IF["n_estimators"]).fit(amb)
        thr, _ = calibrate([det], cal_names, W["length_s"], W["stride_s"], W["agg"], exp["target_fpr"])
        for cap in attack_caps(caps):
            m, _ = evaluate_capture(det, cap, W["length_s"], W["stride_s"], W["agg"], thr[det.name])
            rows.append({"capture": cap["name"], "family": cap["family"],
                         "attack_type": cap["attack_type"], "detector": det.name, "seed": seed, **m})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    df.to_csv(out, index=False)
    s = (df[df.family == "masquerade"].groupby("capture")[["auc_roc", "det_rate_at_1pct_fpr", "frame_auc_roc"]]
         .agg(["mean", "std"]).round(3))
    print(s.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
