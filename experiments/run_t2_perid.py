"""T2 (byte range) as a per-ID detector on ROAD, ROAD-IR and the benign twins: only frames of
the attacked AID contribute, mirroring the per-ID evaluation of the deep models.
python experiments/run_t2_perid.py --out results/t2_perid.csv"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detectors import ByteRange  # noqa: E402
from src.evaluate import metrics, threshold_at_fpr, window_scores  # noqa: E402
from src.io_road import load_cached, load_captures_cfg  # noqa: E402
from src.pipeline import capture_windows, load_ambient, load_experiment_cfg, split_ambient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    fit, cal = split_ambient(caps, exp)
    t2 = ByteRange().fit(load_ambient(fit))

    def per_id_scores(df, aid):
        s = t2.score(df)
        return np.where(df["aid"].to_numpy() == aid, s, 0.0)

    rows = []
    for c in caps["attacks"]:
        if c["family"] != "masquerade":
            continue
        cw = []
        for n in cal:
            df = load_cached(n)
            w, _ = capture_windows(df, 1.0, 0.25)
            cw.append(window_scores(df.t.to_numpy(), per_id_scores(df, c["target_aid"]), w, "mean"))
        thr = threshold_at_fpr(np.concatenate(cw), exp["target_fpr"])
        for variant, suf in (("ROAD", ""), ("ROAD-IR", "__IR"), ("TWIN", "__TWIN")):
            df = load_cached(c["name"] + suf)
            w, y = capture_windows(df, 1.0, 0.25, (c["interval_start_s"], c["interval_end_s"]))
            ws = window_scores(df.t.to_numpy(), per_id_scores(df, c["target_aid"]), w, "mean")
            rows.append({"rule": "T2_perid", "variant": variant, "capture": c["name"],
                         "attack_type": c["attack_type"], **metrics(y, ws, thr)})
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(df.pivot_table(index="attack_type", columns="variant", values="auc_roc").round(2).to_string())


if __name__ == "__main__":
    main()
