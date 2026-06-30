"""Risk forecasting: from a window of recent beats' RR-trend, predict an imminent
abnormal (S/V) beat in the next few beats. A temporal task (not per-beat classification).
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from src import config
from src.data import loader
from src.train import get_device, ARTIFACTS


def _windows_from_record(rr: np.ndarray, y: np.ndarray, n: int, m: int):
    """Sliding windows within one record. X = n past RR values; label = abnormal in next m."""
    X, lab = [], []
    T = len(y)
    for t in range(n - 1, T - m):
        X.append(rr[t - n + 1 : t + 1])
        future = y[t + 1 : t + m + 1]
        lab.append(int(np.any((future == 1) | (future == 2))))
    return X, lab


def _record_sequences(record: str):
    """RR value + class per kept beat, in time order, for one record."""
    signal, rpeaks, symbols = loader._load_record(record)
    rpeaks = np.asarray(rpeaks)
    rr = loader._rr_series(rpeaks)
    rr_m, y_m = [], []
    for j, (r, sym) in enumerate(zip(rpeaks, symbols)):
        cls = loader.map_symbol(sym)
        if cls is None:
            continue
        start, end = r - config.PRE_SAMPLES, r + config.POST_SAMPLES
        if start < 0 or end > len(signal):
            continue
        rr_m.append(rr[j])
        y_m.append(cls)
    return np.asarray(rr_m, dtype=np.float32), np.asarray(y_m, dtype=np.int64)


def build_dataset(records, n: int = config.FORECAST_WINDOW, m: int = config.FORECAST_HORIZON):
    Xs, ys = [], []
    for record in records:
        rr, y = _record_sequences(record)
        if len(y) < n + m:
            continue
        X, lab = _windows_from_record(rr, y, n, m)
        Xs.extend(X)
        ys.extend(lab)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


class RiskForecaster(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: B x N
        out, _ = self.gru(x.unsqueeze(-1))      # B x N x hidden
        return self.head(out[:, -1, :]).squeeze(-1)


def train(epochs: int = 6, batch_size: int = 256, lr: float = 1e-3):
    os.makedirs(ARTIFACTS, exist_ok=True)
    device = get_device()
    print("device:", device)

    Xtr, ytr = build_dataset(config.TRAIN_RECORDS)
    Xte, yte = build_dataset(config.TEST_RECORDS)
    print("train/test windows:", len(ytr), len(yte),
          "| positive rate train/test: %.3f/%.3f" % (ytr.mean(), yte.mean()))

    # standardize RR using train stats
    mu, sd = Xtr.mean(), Xtr.std() + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    pos = max(ytr.sum(), 1); neg = max(len(ytr) - ytr.sum(), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

    model = RiskForecaster().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xt[b].to(device))
            loss = loss_fn(logits, yt[b].to(device))
            loss.backward(); opt.step()
        print(f"epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(Xte).to(device))).cpu().numpy()
    metrics = {
        "auroc": float(roc_auc_score(yte, probs)),
        "average_precision": float(average_precision_score(yte, probs)),
        "test_positive_rate": float(yte.mean()),
        "n_train_windows": int(len(ytr)),
        "n_test_windows": int(len(yte)),
        "window": config.FORECAST_WINDOW,
        "horizon": config.FORECAST_HORIZON,
    }
    print("forecast metrics:", metrics)
    with open(os.path.join(ARTIFACTS, "forecast_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    train()
