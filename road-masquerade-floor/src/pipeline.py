"""Shared fit -> score -> window -> metrics pipeline used by every experiment."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

from .detectors import fit_trivial
from .evaluate import (bootstrap_auc_ci, make_windows, metrics, threshold_at_fpr,
                       window_labels, window_scores)
from .io_road import ROOT, load_captures_cfg, load_cached


def load_experiment_cfg(path=ROOT / "configs" / "experiment.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def split_ambient(cfg_caps: dict, cfg_exp: dict):
    cal = set(cfg_exp["calibration_ambient"])
    names = [c["name"] for c in cfg_caps["ambient"]]
    assert cal <= set(names), cal - set(names)
    return [n for n in names if n not in cal], sorted(cal)


def load_ambient(names, kind="raw", fraction=1.0):
    """Concatenate ambient captures; `fraction` keeps the first share of each capture."""
    dfs = []
    for n in names:
        df = load_cached(n, kind)
        if fraction < 1.0:
            df = df[df["t"] <= df["t"].max() * fraction]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def fit_all(fit_names, t3_q=(0.0, 0.1), fraction=1.0):
    amb_raw = load_ambient(fit_names, "raw", fraction)
    amb_sig = load_ambient(fit_names, "sig", fraction)
    return fit_trivial(amb_raw, amb_sig, t3_q=t3_q)


def is_signal_det(det) -> bool:
    return det.name.startswith("T3")


def capture_windows(df, length, stride, interval=None):
    w = make_windows(0.0, float(df["t"].max()), length, stride)
    return w, window_labels(w, interval)


def score_windows(det, df, windows, agg):
    return window_scores(df["t"].to_numpy(), det.score(df), windows, agg)


def calibrate(dets, cal_names, length, stride, agg, fpr):
    """Per-detector threshold at `fpr` over windows of held-out ambient captures."""
    thr, cal_scores = {}, {}
    for d in dets:
        kind = "sig" if is_signal_det(d) else "raw"
        s = []
        for n in cal_names:
            df = load_cached(n, kind)
            w, _ = capture_windows(df, length, stride)
            s.append(score_windows(d, df, w, agg))
        cal_scores[d.name] = np.concatenate(s)
        thr[d.name] = threshold_at_fpr(cal_scores[d.name], fpr)
    return thr, cal_scores


def evaluate_capture(det, cap, length, stride, agg, thr, n_boot=0, seed=0):
    kind = "sig" if is_signal_det(det) else "raw"
    df = load_cached(cap["name"], kind)
    interval = (cap["interval_start_s"], cap["interval_end_s"])
    w, y = capture_windows(df, length, stride, interval)
    s = score_windows(det, df, w, agg)
    m = metrics(y, s, thr)
    if n_boot:
        m["ci_low"], m["ci_high"] = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
    # Supplementary: frame-level AUC using reconstructed per-frame labels.
    fs, fl = det.score(df), df["label"].to_numpy()
    fm = metrics(fl, fs)
    m["frame_auc_roc"], m["frame_pr_auc"] = fm["auc_roc"], fm["pr_auc"]
    return m, (y, s)


def attack_caps(cfg_caps, families=("masquerade", "fabrication")):
    return [c for c in cfg_caps["attacks"] if c["family"] in families]
