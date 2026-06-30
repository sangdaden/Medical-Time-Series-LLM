import numpy as np
import torch
from src import config
from src.forecast import _windows_from_record, RiskForecaster


def test_windows_labels_and_shapes():
    rr = np.arange(6, dtype=np.float32)
    y = np.array([0, 0, 0, 2, 0, 0], dtype=np.int64)  # V at index 3
    X, lab = _windows_from_record(rr, y, n=3, m=2)
    assert len(X) == 2 and len(lab) == 2
    assert lab == [1, 0]                  # t=2 sees V in next 2; t=3 does not
    assert list(X[0]) == [0.0, 1.0, 2.0]  # rr[0:3]
    assert all(len(w) == 3 for w in X)


def test_forecaster_output_shape():
    model = RiskForecaster()
    x = torch.randn(4, config.FORECAST_WINDOW)
    out = model(x)
    assert out.shape == (4,)
