"""Parsing ROAD captures and caching them to parquet (blueprint §10 C3).

Raw logs are candump format:  (timestamp) iface AID#HEXPAYLOAD
Timestamps are converted to seconds elapsed since the first frame, which is
the clock ROAD's `injection_interval` metadata uses. AIDs are parsed from hex.
ROAD obfuscates AID order, so the numeric AID carries no priority meaning.

Every cached table carries a `family` column (ambient | fabrication |
masquerade | fuzzing | accelerator) so the detectors' ambient-only fitting
guard can see where data came from, and a per-frame `label`:
  masquerade  - target AID inside the injection interval (all such frames are
                injected, because the legitimate ones were deleted)
  fabrication - target AID inside the interval AND payload matches the
                injected pattern (non-'X' bytes of injection_data)
  fuzzing     - inside the interval AND payload == FFFFFFFFFFFFFFFF
  ambient / accelerator - 0
"""
from __future__ import annotations

import pathlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from .detectors import BYTE_COLS

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"


def load_raw_capture(path) -> pd.DataFrame:
    """Columns: t (float64 s since first frame), aid (int32), dlc (int8),
    b0..b7 (int16; -1 where the byte is absent because dlc < 8)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["ts", "iface", "frame"],
                     dtype=str, engine="c")
    ts = df["ts"].str.strip("()").astype(np.float64).to_numpy()
    parts = df["frame"].str.split("#", n=1, expand=True)
    aid = np.array([int(a, 16) for a in parts[0]], np.int32)
    pay = parts[1].fillna("")
    dlc = (pay.str.len() // 2).to_numpy(np.int8)
    hexbuf = "".join(pay.str.slice(0, 16).str.ljust(16, "0"))
    b = np.frombuffer(bytes.fromhex(hexbuf), np.uint8).reshape(-1, 8).astype(np.int16)
    b[np.arange(8)[None, :] >= dlc[:, None]] = -1
    out = pd.DataFrame(b, columns=BYTE_COLS)
    out.insert(0, "dlc", dlc)
    out.insert(0, "aid", aid)
    out.insert(0, "t", ts - ts[0])
    return out


def load_signal_capture(path) -> pd.DataFrame:
    """Columns: t, aid, label (ROAD's own per-message label), s1..sK (float32),
    where s_i is ROAD's "Signal_i_of_ID". Available for 17/33 attacks + all ambient."""
    df = pd.read_csv(path)
    ren = {"Time": "t", "ID": "aid", "Label": "label"}
    ren.update({c: "s" + c.split("_")[1] for c in df.columns if c.startswith("Signal_")})
    df = df.rename(columns=ren)
    sig = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    df[sig] = df[sig].astype(np.float32)
    df["aid"] = df["aid"].astype(np.int32)
    df["label"] = df["label"].astype(np.int8)
    return df


def frame_labels(df: pd.DataFrame, cap: dict) -> np.ndarray:
    fam = cap["family"]
    if fam in ("ambient", "accelerator") or cap.get("interval_start_s") is None:
        return np.zeros(len(df), np.int8)
    inside = (df["t"] >= cap["interval_start_s"]) & (df["t"] <= cap["interval_end_s"])
    pat = cap["injection_data"]
    match = np.ones(len(df), bool)
    for j in range(8):
        hx = pat[2 * j: 2 * j + 2]
        if hx != "XX":
            match &= df[f"b{j}"].to_numpy() == int(hx, 16)
    if fam == "fuzzing":
        lab = inside & match
    elif fam == "masquerade":
        lab = inside & (df["aid"] == cap["target_aid"])
    else:  # fabrication
        lab = inside & (df["aid"] == cap["target_aid"]) & match
    return np.asarray(lab, np.int8)


def load_captures_cfg(path=ROOT / "configs" / "captures.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def all_captures(cfg) -> list[dict]:
    return list(cfg["ambient"]) + list(cfg["attacks"])


def cache_path(name: str, kind: str = "raw") -> pathlib.Path:
    return INTERIM / f"{name}.{kind}.parquet"


def _cache_one(cap: dict) -> str:
    raw = load_raw_capture(ROOT / cap["path"])
    raw["label"] = frame_labels(raw, cap)
    raw["family"] = cap["family"]
    raw.to_parquet(cache_path(cap["name"], "raw"), index=False)
    msg = f"{cap['name']}: {len(raw)} frames, {int(raw.label.sum())} labelled"
    if cap.get("signal_path"):
        sig = load_signal_capture(ROOT / cap["signal_path"])
        sig["family"] = cap["family"]
        sig.to_parquet(cache_path(cap["name"], "sig"), index=False)
        msg += f"; signals {len(sig)} rows, {int(sig.label.sum())} labelled"
    return msg


def cache_all(cfg, processes: int = 8) -> None:
    """Write data/interim/<name>.{raw,sig}.parquet for every capture."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    caps = sorted(all_captures(cfg), key=lambda c: -c["elapsed_sec"])  # big first
    with Pool(processes) as pool:
        for msg in pool.imap_unordered(_cache_one, caps):
            print(msg, flush=True)


def load_cached(name: str, kind: str = "raw") -> pd.DataFrame:
    return pd.read_parquet(cache_path(name, kind))
