"""Experiment (e): why transfer fails. Model-free footprint of each attack in each campaign.

For every (campaign, attack): UAV output rate, HEARTBEAT share, and the Jensen-Shannon
distance of its message-ID distribution to (i) its own campaign's normal traffic and
(ii) the other campaigns' normal traffic. Also the share of each campaign's normal
traffic made of IDs unseen in the other campaign (the OOV mass).

python experiments/run_errors.py --out results/errors.csv
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inventory import load, load_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/errors.csv"))
    args = ap.parse_args()
    caps = load_cfg()["captures"]
    groups = {}
    for c in caps:
        groups.setdefault((c["campaign"], c["attack"]), []).append(c["name"])
    data = {k: pd.concat([load(n) for n in v], ignore_index=True) for k, v in groups.items()}
    ids = sorted({int(x) for d in data.values() for x in d.msgid.unique()})
    pos = {m: i for i, m in enumerate(ids)}

    def dist(d):
        h = np.zeros(len(ids))
        np.add.at(h, [pos[int(m)] for m in d.msgid], 1)
        return h / h.sum()

    def rate(names):
        return np.mean([len(load(n)) / max(load(n).t.max(), 1e-9) for n in names])

    P = {k: dist(d) for k, d in data.items()}
    rows = []
    for (camp, att), d in sorted(data.items()):
        r = {"campaign": camp, "attack": att, "frames": len(d), "rate_fps": rate(groups[(camp, att)]),
             "heartbeat_share": float((d.msgid == 0).mean()), "n_ids": d.msgid.nunique()}
        for other in ("OLD", "NEW", "SHADOW"):
            if (other, "normal") in P:
                r[f"jsd_to_{other}_normal"] = float(jensenshannon(P[(camp, att)], P[(other, "normal")], base=2))
        rows.append(r)
    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    df.to_csv(out, index=False)
    print(df.round(3).to_string())

    oov = []
    for a in ("OLD", "NEW", "SHADOW"):
        for b in ("OLD", "NEW", "SHADOW"):
            if a == b:
                continue
            seen = set(data[(a, "normal")].msgid.unique())
            for att in sorted({k[1] for k in data if k[0] == b}):
                d = data[(b, att)]
                oov.append({"train": a, "test": b, "attack": att,
                            "oov_frame_share": float((~d.msgid.isin(seen)).mean())})
    o = pd.DataFrame(oov)
    o.to_csv(out.with_name("oov.csv"), index=False)
    print(o.pivot_table(index=["train", "test"], columns="attack", values="oov_frame_share").round(3).to_string())


if __name__ == "__main__":
    main()
