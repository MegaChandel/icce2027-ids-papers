"""Experiment (e): the identical trivial detectors on the CrySyS corpus (Gazdag et al. 2023).

CrySyS synthesises attacks on 26 benign recordings of one vehicle. Masquerade = "msg-mod"
(the victim's own frames are modified in place); fabrication = "msg-inj". Payload strategies:
CONST, ADD-INCR, ADD-DECR, POS-OFFSET, NEG-OFFSET, REPLAY, on IDs 0x110 and 0x410, over
20% or 5% of the trace. Detectors for recording r are fitted on the OTHER 25 benign
recordings only (leave-one-recording-out); no retuning. Window protocol as for ROAD.

python experiments/run_crysys.py --out results/crysys.csv
"""
import argparse
import json
import pathlib
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detectors import BYTE_COLS, ByteRange, ByteValueSet  # noqa: E402
from src.evaluate import make_windows, metrics, window_labels, window_scores  # noqa: E402

BASE = ROOT / "data/raw/crysys/CrySyS dataset of CAN traffic logs containing fabrication and masquerade attacks"
CACHE = ROOT / "data/interim/crysys"
PAT = re.compile(r"-malicious-([A-Z_-]+?)-msg-(inj|mod)-(0x[0-9a-f]+)-([0-9.]+)-([0-9.]+)\.log$")


def parse(path) -> pd.DataFrame:
    """candump lines '(ts) can0 ID#DATA [flag<TAB>flag]'; returns t, aid, dlc, b0..b7, flag."""
    ts_l, fr_l, fl_l = [], [], []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 3 and f[0].startswith("("):
                ts_l.append(f[0]); fr_l.append(f[2]); fl_l.append(f[3] if len(f) > 3 else "False")
    rows = pd.DataFrame({"ts": ts_l, "frame": fr_l, "flag": fl_l})
    ts = rows["ts"].str.strip("()").astype(float).to_numpy()
    parts = rows["frame"].str.split("#", n=1, expand=True)
    aid = np.array([int(a, 16) for a in parts[0]], np.int32)
    pay = parts[1].fillna("")
    dlc = (pay.str.len() // 2).to_numpy(np.int8)
    b = np.frombuffer(bytes.fromhex("".join(pay.str.slice(0, 16).str.ljust(16, "0"))), np.uint8) \
        .reshape(-1, 8).astype(np.int16)
    b[np.arange(8)[None, :] >= dlc[:, None]] = -1
    df = pd.DataFrame(b, columns=BYTE_COLS)
    df.insert(0, "dlc", dlc); df.insert(0, "aid", aid); df.insert(0, "t", ts)
    df["flag"] = (rows["flag"].fillna("False") != "False").astype(np.int8).to_numpy()
    return df


def benign(rec):
    p = CACHE / f"{rec}-benign.parquet"
    if not p.exists():
        df = parse(BASE / rec / f"{rec}-benign.log")
        df["family"] = "ambient"
        df.to_parquet(p, index=False)
    return pd.read_parquet(p)


def attacks(rec):
    out = []
    for f in sorted((BASE / rec).glob(f"{rec}-malicious-*-msg-*.log")):
        if f.name.endswith("-inj-messages.log") or "DOUBLE" in f.name:
            continue
        m = PAT.search(f.name)
        if m:
            meta = json.loads(f.with_suffix(".json").read_text())
            mk = {x["description"]: x["time"] for x in meta["markers"]}
            out.append(dict(file=f, strategy=m.group(1).replace("_", "-"),
                            family="masquerade" if m.group(2) == "mod" else "fabrication",
                            target=m.group(3), span=f"{m.group(4)}-{m.group(5)}",
                            start=mk["Start of the attack."], end=mk["End of the attack."]))
    return out


def merged_detectors(per_rec, exclude):
    """Combine per-recording fitted statistics: value-set union, range min/max."""
    t1, t2 = ByteValueSet(), ByteRange()
    for rec, (v, r) in per_rec.items():
        if rec == exclude:
            continue
        for aid, m in v.seen.items():
            t1.seen[aid] = t1.seen[aid] | m if aid in t1.seen else m.copy()
        for aid in r.lo:
            t2.lo[aid] = np.minimum(t2.lo[aid], r.lo[aid]) if aid in t2.lo else r.lo[aid].copy()
            t2.hi[aid] = np.maximum(t2.hi[aid], r.hi[aid]) if aid in t2.hi else r.hi[aid].copy()
    return [t1, t2]


def fit_one(rec):
    df = benign(rec)
    return rec, (ByteValueSet().fit(df), ByteRange().fit(df))


def eval_rec(args):
    rec, dets, L, stride = args
    rows = []
    for a in attacks(rec):
        df = parse(a["file"])
        w = make_windows(0.0, float(df.t.max()), L, stride)
        y = window_labels(w, (a["start"], a["end"]))
        for d in dets:
            s = window_scores(df.t.to_numpy(), d.score(df), w, "mean")
            m = metrics(y, s)
            fl = df["flag"].to_numpy()
            fm = metrics(fl, d.score(df)) if a["family"] == "masquerade" and fl.any() else {"auc_roc": np.nan}
            # share of modified frames that violate the detector (masquerade only)
            oor = float((d.score(df)[fl == 1] > 0).mean()) if a["family"] == "masquerade" and fl.any() else np.nan
            rows.append({"recording": rec, **{k: a[k] for k in ("strategy", "family", "target", "span")},
                         "detector": d.name, "auc_roc": m["auc_roc"], "frame_auc_roc": fm["auc_roc"], "flagged_share": oor,
                         "n_pos_windows": m["n_pos"], "n_windows": m["n_windows"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/crysys.csv"))
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = sorted(p.name for p in BASE.iterdir() if p.is_dir())
    with Pool(12) as pool:
        per_rec = dict(pool.map(fit_one, recs))
    jobs = [(r, merged_detectors(per_rec, r), 1.0, 0.25) for r in recs]
    rows = []
    with Pool(12) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_rec, jobs), 1):
            rows += r
            print(f"  {i}/{len(recs)} recordings", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    g = df.groupby(["family", "strategy", "detector"])["auc_roc"].mean().unstack("detector")
    print(g.round(3).to_string())
    print(df.groupby(["family", "detector"])["auc_roc"].agg(["mean", "median"]).round(3).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
