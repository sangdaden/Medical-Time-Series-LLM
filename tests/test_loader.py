import numpy as np
from src.data import loader
from src import config


def test_map_symbol_to_class():
    assert loader.map_symbol("N") == 0
    assert loader.map_symbol("V") == 2
    assert loader.map_symbol("?") is None  # unmapped symbols dropped


def test_segment_beats_shapes():
    # synthetic signal + r-peaks; avoid network in unit test
    sig = np.arange(2000, dtype=np.float32)
    rpeaks = [500, 1000, 1500]
    symbols = ["N", "V", "N"]
    X, y = loader.segment_beats(sig, rpeaks, symbols)
    assert X.shape == (3, config.WINDOW)
    assert list(y) == [0, 2, 0]


def test_balance_classes_caps_each_class():
    # class 0 over-represented, class 1 under per_class, class 2 exactly at cap
    X = np.arange(100, dtype=np.float32).reshape(100, 1)
    y = np.array([0] * 80 + [1] * 5 + [2] * 15, dtype=np.int64)
    Xb, yb = loader.balance_classes(X, y, per_class=10, seed=0)
    counts = np.bincount(yb, minlength=len(config.CLASSES))
    assert counts[0] == 10   # capped from 80
    assert counts[1] == 5    # kept (below cap)
    assert counts[2] == 10   # capped from 15
    assert Xb.shape[0] == yb.shape[0] == 25


def test_segment_beats_drops_edge_and_unmapped():
    sig = np.arange(2000, dtype=np.float32)
    rpeaks = [10, 1000, 1990]        # 10 and 1990 too close to edges
    symbols = ["N", "?", "N"]        # middle unmapped
    X, y = loader.segment_beats(sig, rpeaks, symbols)
    assert X.shape == (0, config.WINDOW)
    assert y.shape == (0,)
