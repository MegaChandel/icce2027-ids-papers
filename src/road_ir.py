"""ROAD-IR: an in-range counterfactual of the ROAD masquerade captures.

For every injected frame (label 1) of a masquerade capture, each byte the attack wrote
(non-'X' in `injection_data`) is clamped into the [min, max] range that byte takes for that
AID in the fit-set ambient captures. Everything else is untouched. The attack keeps its
meaning (speedometer at the highest speed ever seen while the car is slow; three wheel
speeds at the top of their range and one at zero; coolant at its ambient maximum; the
reverse-light attacks are already in range and are unchanged), but no injected byte is
out of range any more, so the payload shortcut is removed by construction.

python -m src.road_ir     # writes data/interim/<name>__IR.raw.parquet
"""
from __future__ import annotations

import numpy as np

from .detectors import BYTE_COLS
from .io_road import cache_path, load_cached, load_captures_cfg
from .pipeline import load_ambient, load_experiment_cfg, split_ambient


def ambient_ranges():
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    fit, _ = split_ambient(caps, exp)
    amb = load_ambient(fit)
    g = amb.groupby("aid")[BYTE_COLS]
    return g.min(), g.max()


def make_ir(cap, lo, hi):
    df = load_cached(cap["name"]).copy()
    aid, pat = cap["target_aid"], cap["injection_data"]
    inj = df["label"].to_numpy() == 1
    changed = 0
    for j in range(8):
        if pat[2 * j: 2 * j + 2] == "XX":
            continue
        col = f"b{j}"
        v = df.loc[inj, col].to_numpy()
        c = np.clip(v, lo.loc[aid, col], hi.loc[aid, col])
        changed += int((c != v).sum())
        df.loc[inj, col] = c.astype(df[col].dtype)
    df.to_parquet(cache_path(cap["name"] + "__IR", "raw"), index=False)
    return changed, int(inj.sum())


def main():
    caps = load_captures_cfg()
    lo, hi = ambient_ranges()
    for cap in caps["attacks"]:
        if cap["family"] != "masquerade":
            continue
        changed, n = make_ir(cap, lo, hi)
        print(f"{cap['name']:45s} injected={n:5d} bytes clamped={changed}")


if __name__ == "__main__":
    main()


def make_twin(cap):
    """Benign twin of a masquerade capture: its fabrication original with every injected frame
    removed, i.e. the same drive with no attack at all. Same interval labels are kept, so a
    detector that still 'finds' the interval is reacting to the driving phase, not the attack."""
    fab = load_cached(cap["pair_with"])
    twin = fab[fab["label"] == 0].reset_index(drop=True)
    twin.to_parquet(cache_path(cap["name"] + "__TWIN", "raw"), index=False)
    return len(fab) - len(twin)


def main_twins():
    for cap in load_captures_cfg()["attacks"]:
        if cap["family"] == "masquerade":
            print(f"{cap['name']:45s} removed injected frames: {make_twin(cap)}")
