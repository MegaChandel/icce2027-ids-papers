"""Experiment (b): the trivial-baseline floor on all masquerade + fabrication captures.

python experiments/run_floor.py --out results/floor.csv
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.io_road import ROOT, load_cached, load_captures_cfg  # noqa: E402
from src.pipeline import (attack_caps, calibrate, evaluate_capture, fit_all,  # noqa: E402
                          is_signal_det, load_experiment_cfg, split_ambient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiment.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/floor.csv"))
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    exp, caps = load_experiment_cfg(args.config), load_captures_cfg()
    W = exp["window"]
    n_boot = exp["n_boot"] if args.n_boot is None else args.n_boot
    fit_names, cal_names = split_ambient(caps, exp)

    t0 = time.time()
    dets = fit_all(fit_names, t3_q=exp["t3_quantiles"])
    print(f"fitted {[d.name for d in dets]} on {len(fit_names)} ambient captures "
          f"in {time.time() - t0:.0f}s", flush=True)

    rows, store = [], {}
    for agg in ("mean", "max"):
        thr, cal = calibrate(dets, cal_names, W["length_s"], W["stride_s"], agg, exp["target_fpr"])
        if agg == W["agg"]:
            # Ambient self-test (blueprint E2): frame flag rate on held-out ambient.
            for d in dets:
                kind = "sig" if is_signal_det(d) else "raw"
                fr = np.mean([np.mean(d.score(load_cached(n, kind)) > 0) for n in cal_names])
                print(f"  self-test {d.name:18s} held-out ambient frame flag rate = {fr:.4%}  "
                      f"thr={thr[d.name]:.4g}", flush=True)
        for cap in attack_caps(caps):
            for d in dets:
                if is_signal_det(d) and not cap.get("signal_path"):
                    continue
                m, (y, s) = evaluate_capture(d, cap, W["length_s"], W["stride_s"], agg,
                                             thr[d.name], n_boot=n_boot if agg == W["agg"] else 0,
                                             seed=exp["seed"])
                rows.append({"capture": cap["name"], "family": cap["family"],
                             "attack_type": cap["attack_type"], "detector": d.name, "agg": agg,
                             **m})
                if agg == W["agg"]:
                    store[f"{cap['name']}|{d.name}"] = np.stack([y, s])
        print(f"agg={agg} done ({time.time() - t0:.0f}s)", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    np.savez_compressed(out.with_suffix(".windows.npz"), **store)
    view = df[(df["agg"] == W["agg"]) & (df["family"] == "masquerade")]
    print(view.pivot_table(index="capture", columns="detector", values="auc_roc").round(3).to_string())
    print(view.pivot_table(index="capture", columns="detector",
                           values="det_rate_at_1pct_fpr").round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
