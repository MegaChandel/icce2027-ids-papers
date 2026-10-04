import numpy as np
import pytest

from src.models import ALL, make
from src.tokenize import OOV, Segment, build_vocab, make_xy, split, windows


def test_oov_is_zero_and_vocab_starts_at_one():
    v = build_vocab([Segment("old_normal_1")])
    assert OOV == 0 and min(v.values()) == 1 and 0 not in v.values()


def test_window_shapes_and_no_boundary_crossing():
    v = build_vocab([Segment("old_normal_1")])
    w, d, o = windows(Segment("old_normal_1"), v, 32, 16)
    assert w.shape[1] == 32 and len(w) == len(d) == len(o)
    assert (o == 0).all()  # in-vocabulary
    tr, _ = split("OLD")
    xy = make_xy(tr, v, 32, 16)
    n_expected = sum(len(windows(s, v, 32, 16)[0]) for s, _ in tr)
    assert len(xy["X"]) == n_expected  # windows built per file, then stacked


def test_train_only_vocab_produces_oov_on_other_campaign():
    v_old = build_vocab([s for s, _ in split("OLD")[0]])
    _, te = split("SHADOW")
    xy = make_xy(te, v_old, 32, 16)
    assert xy["oov"].mean() > 0  # SHADOW uses IDs OLD never saw


def test_splits_disjoint():
    for camp in ("OLD", "NEW", "SHADOW"):
        tr, te = split(camp)
        if camp == "NEW":
            for (a, _), (b, _) in zip(tr, te):
                assert a.name == b.name and a.hi <= b.lo
        else:
            assert not ({s.name for s, _ in tr} & {s.name for s, _ in te})
        assert {c["label"] for _, c in te} == {0, 1}


@pytest.mark.parametrize("name", ALL)
def test_models_smoke(name):
    rng = np.random.default_rng(0)
    X = rng.integers(1, 10, (200, 16)); y = (rng.random(200) < 0.5).astype(int)
    X[y == 1, :4] = 9
    xy = dict(X=X, y=y, dur=1 + y + rng.random(200) * 0.1, oov=np.zeros(200))
    s = make(name, 10, epochs=2).fit(xy).score(xy)
    assert s.shape == (200,) and np.isfinite(s).all()
