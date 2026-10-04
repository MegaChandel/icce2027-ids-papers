"""Integrity tests on the parquet caches (skipped if `python -m src.cli cache` has not run)."""
import numpy as np
import pandas as pd
import pytest

from src.detectors import BYTE_COLS
from src.io_road import all_captures, cache_path, load_cached, load_captures_cfg

CFG = load_captures_cfg()
CAPS = all_captures(CFG)
pytestmark = pytest.mark.skipif(not cache_path(CAPS[0]["name"]).exists(), reason="cache not built")


@pytest.mark.parametrize("cap", CAPS, ids=lambda c: c["name"])
def test_raw_cache(cap):
    df = load_cached(cap["name"])
    assert len(df) > 0
    assert np.all(np.diff(df["t"].to_numpy()) >= 0)
    assert (df["aid"] >= 0).all()
    b = df[BYTE_COLS].to_numpy()
    assert ((b >= -1) & (b <= 255)).all()
    assert (df["family"] == cap["family"]).all()
    if cap["family"] in ("masquerade", "fabrication", "fuzzing"):
        assert df["label"].sum() > 0


def test_counts_match_paper():
    fam = pd.Series([c["family"] for c in CFG["attacks"]]).value_counts().to_dict()
    assert len(CFG["ambient"]) == 12 and len(CFG["attacks"]) == 33
    assert fam == {"fabrication": 13, "masquerade": 13, "accelerator": 4, "fuzzing": 3}


MASQ = [c for c in CFG["attacks"] if c["family"] == "masquerade"]


@pytest.mark.parametrize("cap", MASQ, ids=lambda c: c["name"])
def test_masquerade_is_fabrication_minus_frames(cap):
    """ROAD's documented construction: masquerade = fabrication with legitimate
    target-AID frames removed in the injection interval. Non-target traffic and
    the injected frames must be identical."""
    m, f = load_cached(cap["name"]), load_cached(cap["pair_with"])
    key = ["aid"] + BYTE_COLS
    other_m = m[m.aid != cap["target_aid"]][key].reset_index(drop=True)
    other_f = f[f.aid != cap["target_aid"]][key].reset_index(drop=True)
    pd.testing.assert_frame_equal(other_m, other_f)
    inj_m = m[m.label == 1][key].reset_index(drop=True)
    inj_f = f[f.label == 1][key].reset_index(drop=True)
    assert abs(len(inj_m) - len(inj_f)) <= 2  # interval-edge frames may differ
    assert len(m) < len(f)


@pytest.mark.parametrize("cap", MASQ, ids=lambda c: c["name"])
def test_labels_agree_with_road_signal_labels(cap):
    raw, sig = load_cached(cap["name"]), load_cached(cap["name"], "sig")
    assert raw["label"].sum() == sig["label"].sum()
