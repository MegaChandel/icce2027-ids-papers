"""Experiment (c): which component of the trivial rule carries the result.

Variants of T2 (byte range), all fitted on the same ambient captures:
  T2_range        - baseline (sum of byte excursions over all AIDs)
  T2_norm         - excursions normalised by each byte's ambient range width
  T2_byte{j}      - only byte position j contributes
  T2_target_only  - only frames of the attacked AID contribute (oracle upper bound)
  T2_excl_target  - the attacked AID is ignored (tests for physical side-effects)
plus the per-(capture, byte) flag rate of injected frames for Figure 4.

python experiments/run_ablation.py --out results/ablation.csv
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.detectors import ByteRange  # noqa: E402
from src.evaluate import metrics, threshold_at_fpr, window_scores  # noqa: E402
from src.io_road import ROOT, load_cached, load_captures_cfg  # noqa: E402
from src.pipeline import (attack_caps, capture_windows, load_ambient,  # noqa: E402
                          load_experiment_cfg, split_ambient)


def variants(t2, df, target_aid):
    ex = t2.byte_excess(df)
    is_t = (df["aid"] == target_aid).to_numpy()
    out = {"T2_range": ex.sum(1), "T2_norm": t2.byte_excess(df, normalise=True).sum(1),
           "T2_target_only": np.where(is_t, ex.sum(1), 0.0),
           "T2_excl_target": np.where(is_t, 0.0, ex.sum(1))}
    for j in range(8):
        out[f"T2_byte{j}"] = ex[:, j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/ablation.csv"))
    args = ap.parse_args()
    exp, caps = load_experiment_cfg(), load_captures_cfg()
    W = exp["window"]
    fit_names, cal_names = split_ambient(caps, exp)
    t2 = ByteRange().fit(load_ambient(fit_names))

    # Calibration windows for every variant that does not depend on a target AID.
    cal = {}
    for n in cal_names:
        df = load_cached(n)
        w, _ = capture_windows(df, W["length_s"], W["stride_s"])
        for k, s in variants(t2, df, target_aid=-1).items():
            cal.setdefault(k, []).append(window_scores(df["t"].to_numpy(), s, w, W["agg"]))
    thr = {k: threshold_at_fpr(np.concatenate(v), exp["target_fpr"]) for k, v in cal.items()}

    rows, heat = [], []
    for cap in attack_caps(caps):
        df = load_cached(cap["name"])
        w, y = capture_windows(df, W["length_s"], W["stride_s"],
                               (cap["interval_start_s"], cap["interval_end_s"]))
        for k, s in variants(t2, df, cap["target_aid"]).items():
            ws = window_scores(df["t"].to_numpy(), s, w, W["agg"])
            # target_only / excl_target depend on the capture's target AID, so they have no
            # ambient calibration threshold; report AUC only for them.
            m = metrics(y, ws, None if k in ("T2_target_only", "T2_excl_target") else thr[k])
            fm = metrics(df["label"].to_numpy(), s)
            rows.append({"capture": cap["name"], "family": cap["family"], "attack_type": cap["attack_type"],
                         "variant": k, **m, "frame_auc_roc": fm["auc_roc"]})
        inj = df[df["label"] == 1]
        fr = (t2.byte_excess(inj) > 0).mean(0)
        heat.append({"capture": cap["name"], "family": cap["family"],
                     **{f"b{j}": float(fr[j]) for j in range(8)}})
    out = pathlib.Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(heat).to_csv(out.with_name("byte_flag_rates.csv"), index=False)
    df = pd.DataFrame(rows)
    v = df[df.family == "masquerade"]
    print(v.pivot_table(index="capture", columns="variant", values="auc_roc").round(2).to_string())
    print(pd.DataFrame(heat).query("family == 'masquerade'").round(2).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
