"""Train the multimodal classifier and compare ECG-only vs ECG+RR (cross-modal benefit)."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import config
from src.data import loader
from src.models.multimodal import MultimodalClassifier
from src.train import get_device, ARTIFACTS


def _prep():
    ecg_tr, rr_tr, y_tr = loader.load_split_multimodal(config.TRAIN_RECORDS)
    ecg_te, rr_te, y_te = loader.load_split_multimodal(config.TEST_RECORDS)
    # balance training set by class (same indices for both modalities)
    idx = loader.balanced_indices(y_tr, per_class=2000, seed=0)
    ecg_tr, rr_tr, y_tr = ecg_tr[idx], rr_tr[idx], y_tr[idx]
    # standardize RR features using train statistics
    mu, sd = rr_tr.mean(0), rr_tr.std(0) + 1e-6
    rr_tr = (rr_tr - mu) / sd
    rr_te = (rr_te - mu) / sd
    return ecg_tr, rr_tr, y_tr, ecg_te, rr_te, y_te


def _train_one(use_rr, data, device, epochs=8, batch_size=256, lr=1e-3):
    ecg_tr, rr_tr, y_tr, ecg_te, rr_te, y_te = data
    Xe, Xr, yt = torch.tensor(ecg_tr), torch.tensor(rr_tr), torch.tensor(y_tr)
    counts = np.bincount(y_tr, minlength=len(config.CLASSES))
    weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32).to(device)

    model = MultimodalClassifier(use_rr=use_rr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    n = len(Xe)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xe[b].to(device), Xr[b].to(device) if use_rr else None)
            loss = loss_fn(logits, yt[b].to(device))
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(ecg_te).to(device),
                       torch.tensor(rr_te).to(device) if use_rr else None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(1)
    try:
        auroc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_te, preds)),
        "macro_f1": float(f1_score(y_te, preds, average="macro")),
        "macro_auroc": float(auroc),
    }


def train():
    os.makedirs(ARTIFACTS, exist_ok=True)
    device = get_device()
    print("device:", device)
    data = _prep()
    print("train/test beats:", len(data[2]), len(data[5]))

    ecg_only = _train_one(False, data, device)
    print("ECG-only :", ecg_only)
    multimodal = _train_one(True, data, device)
    print("ECG + RR :", multimodal)

    out = {
        "ecg_only": ecg_only,
        "ecg_plus_rr": multimodal,
        "delta_macro_f1": multimodal["macro_f1"] - ecg_only["macro_f1"],
        "delta_macro_auroc": multimodal["macro_auroc"] - ecg_only["macro_auroc"],
    }
    with open(os.path.join(ARTIFACTS, "multimodal_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("delta macro-F1:", round(out["delta_macro_f1"], 4),
          "| delta AUROC:", round(out["delta_macro_auroc"], 4))
    return out


if __name__ == "__main__":
    train()
