"""Train the multimodal framework on several modality combinations and report the
incremental cross-modal benefit (ECG -> +RR -> +clinical)."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import config
from src.data import loader
from src.models.multimodal import MultimodalModel
from src.train import get_device, ARTIFACTS

# modality combinations to compare (cumulative)
COMBOS = [["ecg"], ["ecg", "rr"], ["ecg", "rr", "clinical"]]
# modalities that are not z-normalized by the loader and benefit from standardization
_STANDARDIZE = {"rr", "clinical"}


def _prep():
    """Load all configured modalities once; balance train; standardize tabular feats."""
    feats_tr, y_tr = loader.load_multimodal(config.TRAIN_RECORDS)
    feats_te, y_te = loader.load_multimodal(config.TEST_RECORDS)
    idx = loader.balanced_indices(y_tr, per_class=2000, seed=0)
    feats_tr = {k: v[idx] for k, v in feats_tr.items()}
    y_tr = y_tr[idx]
    for name in feats_tr:
        if name in _STANDARDIZE:
            mu, sd = feats_tr[name].mean(0), feats_tr[name].std(0) + 1e-6
            feats_tr[name] = (feats_tr[name] - mu) / sd
            feats_te[name] = (feats_te[name] - mu) / sd
    return feats_tr, y_tr, feats_te, y_te


def _to_tensors(feats, names, device, idx=None):
    out = {}
    for n in names:
        t = torch.tensor(feats[n])
        out[n] = (t[idx] if idx is not None else t).to(device)
    return out


def _train_one(modalities, data, device, epochs=8, batch_size=256, lr=1e-3):
    feats_tr, y_tr, feats_te, y_te = data
    yt = torch.tensor(y_tr)
    counts = np.bincount(y_tr, minlength=len(config.CLASSES))
    weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32).to(device)

    model = MultimodalModel(modalities).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    n = len(y_tr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            batch = _to_tensors(feats_tr, modalities, device, b)
            loss = loss_fn(model(batch), yt[b].to(device))
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(_to_tensors(feats_te, modalities, device)), dim=1).cpu().numpy()
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
    print("train/test beats:", len(data[1]), len(data[3]))

    results = {}
    for combo in COMBOS:
        m = _train_one(combo, data, device)
        key = "+".join(combo)
        results[key] = m
        print(f"{key:20s} -> macro-F1 {m['macro_f1']:.3f} | AUROC {m['macro_auroc']:.3f}")

    base = results["ecg"]["macro_f1"]
    out = {
        "combinations": results,
        "delta_macro_f1_vs_ecg": {k: results[k]["macro_f1"] - base for k in results},
    }
    with open(os.path.join(ARTIFACTS, "multimodal_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    train()
