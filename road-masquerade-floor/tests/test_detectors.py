import numpy as np
import pandas as pd
import pytest

from src.detectors import (AttackDataInFitError, ByteRange, ByteValueSet,
                           PayloadNovelty, SignalEnvelope, fit_trivial)


def frames(aid, b0s, family=None):
    n = len(b0s)
    d = {"t": np.arange(n, dtype=float), "aid": [aid] * n, "dlc": [8] * n}
    for j in range(8):
        d[f"b{j}"] = list(b0s) if j == 0 else [0] * n
    df = pd.DataFrame(d)
    if family:
        df["family"] = family
    return df


@pytest.fixture
def amb():
    return frames(0x100, [1, 2, 3, 1, 2, 3], family="ambient")


def test_t1_valueset(amb):
    t1 = ByteValueSet().fit(amb)
    s = t1.score(frames(0x100, [9, 2]))
    assert s[0] > 0 and s[1] == 0


def test_t2_range(amb):
    t2 = ByteRange().fit(amb)
    fl = t2.byte_flags(frames(0x100, [2, 0, 4])).any(1)
    assert list(fl) == [False, True, True]


def test_t1_superset_of_t2():
    amb = frames(0x100, [1, 3, 5], family="ambient")
    test = frames(0x100, list(range(0, 8)))
    f1 = ByteValueSet().fit(amb).byte_flags(test).any(1)
    f2 = ByteRange().fit(amb).byte_flags(test).any(1)
    assert np.all(f1 >= f2) and f1.sum() > f2.sum()  # 2 and 4 are in range but unseen


def test_unseen_aid_flagged_by_all(amb):
    test = frames(0x200, [1])
    for d in fit_trivial(amb):
        assert d.score(test)[0] > 0, d.name


def test_novelty(amb):
    t4 = PayloadNovelty().fit(amb)
    assert list(t4.score(frames(0x100, [1, 7]))) == [0.0, 1.0]


def test_dlc_short_frames():
    amb = frames(0x100, [1, 2], family="ambient")
    amb.loc[:, [f"b{j}" for j in range(4, 8)]] = -1  # dlc 4
    amb["dlc"] = 4
    test = amb.drop(columns="family").copy()
    for d in fit_trivial(amb):
        assert (d.score(test) == 0).all(), d.name


def test_refuses_attack_data(amb):
    bad = frames(0x100, [9], family="masquerade")
    with pytest.raises(AttackDataInFitError):
        fit_trivial([amb, bad])
    with pytest.raises(AttackDataInFitError):
        ByteValueSet().fit(bad)


def test_signal_envelope():
    amb = pd.DataFrame({"t": range(5), "aid": 1, "s1": [0.0, 1, 2, 3, 4],
                        "s2": np.nan, "family": "ambient"})
    t3 = SignalEnvelope().fit(amb)
    test = pd.DataFrame({"t": range(5), "aid": [1, 1, 1, 1, 2],
                         "s1": [2.0, 8.0, -4.0, 2.0, 0.0], "s2": [np.nan] * 3 + [5.0, np.nan]})
    s = t3.score(test)
    assert s[0] == 0 and s[1] == pytest.approx(1.0) and s[2] == pytest.approx(1.0)
    assert s[3] == 1.0  # s2 never seen for aid 1 in ambient
    assert s[4] == 1.0  # unseen aid


def test_signal_envelope_per_aid():
    amb = pd.DataFrame({"t": range(4), "aid": [1, 1, 2, 2], "s1": [0.0, 1, 100, 101],
                        "family": "ambient"})
    t3 = SignalEnvelope().fit(amb)
    test = pd.DataFrame({"t": range(2), "aid": [1, 2], "s1": [100.0, 0.0]})
    assert (t3.score(test) > 0).all()  # each value is only normal for the OTHER aid
