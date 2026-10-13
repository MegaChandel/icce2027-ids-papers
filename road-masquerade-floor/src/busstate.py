"""Bus-state representation for the deep models.

Every 1/HZ seconds, record the most recent payload of each selected AID (zero-order hold):
S[t, a, :] = the 8 bytes last seen for AID a at time <= t (absent bytes -> 0; before the first
frame of an AID -> 0). Selected AIDs: present in every fit-set ambient capture at >= 5 Hz
(64 AIDs, including all attacked identifiers). Selection uses ambient data only.

Labels per step: 1 if the step lies inside the capture's labelled injection interval.
"""
from __future__ import annotations

import numpy as np

from .detectors import BYTE_COLS
from .io_road import INTERIM, load_cached, load_captures_cfg
from .pipeline import load_experiment_cfg, split_ambient

HZ = 50
MIN_RATE = 5.0


def select_aids() -> np.ndarray:
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    fit, _ = split_ambient(caps, exp)
    keep = None
    for n in fit:
        df = load_cached(n)
        r = df.groupby("aid").size() / df.t.max()
        s = set(r[r >= MIN_RATE].index)
        keep = s if keep is None else keep & s
    return np.array(sorted(keep), np.int64)


def build(df, aids: np.ndarray, hz: int = HZ):
    """df: raw frame table (t, aid, b0..b7). Returns (times[T], S uint8[T, A, 8])."""
    grid = np.arange(0.0, float(df.t.max()), 1.0 / hz)
    S = np.zeros((len(grid), len(aids), 8), np.uint8)
    pos = {a: i for i, a in enumerate(aids)}
    sub = df[df.aid.isin(pos)]
    for aid, g in sub.groupby("aid"):
        t = g.t.to_numpy()
        b = np.clip(g[BYTE_COLS].to_numpy(), 0, 255).astype(np.uint8)
        idx = np.searchsorted(t, grid, side="right") - 1  # last frame at or before each step
        ok = idx >= 0
        S[ok, pos[aid]] = b[idx[ok]]
    return grid, S


def cache(name: str, aids: np.ndarray, interval=None):
    out = INTERIM / f"{name}.bus.npz"
    if not out.exists():
        t, S = build(load_cached(name), aids)
        y = np.zeros(len(t), np.int8)
        if interval is not None:
            y = ((t >= interval[0]) & (t <= interval[1])).astype(np.int8)
        np.savez_compressed(out, t=t, S=S, y=y, aids=aids)
    return out


def load(name: str):
    z = np.load(INTERIM / f"{name}.bus.npz")
    return z["t"], z["S"], z["y"]


def main():
    from multiprocessing import Pool
    caps = load_captures_cfg()
    aids = select_aids()
    np.save(INTERIM / "bus_aids.npy", aids)
    jobs = [(c["name"], None) for c in caps["ambient"]]
    for c in caps["attacks"]:
        if c["family"] == "masquerade":
            iv = (c["interval_start_s"], c["interval_end_s"])
            jobs += [(c["name"], iv), (c["name"] + "__IR", iv), (c["name"] + "__TWIN", iv)]
    with Pool(24) as pool:
        for p in pool.starmap(cache, [(n, aids, iv) for n, iv in jobs]):
            print(p.name, flush=True)
    print(f"{len(aids)} AIDs, {len(jobs)} captures")


if __name__ == "__main__":
    main()
