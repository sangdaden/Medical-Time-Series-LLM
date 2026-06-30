"""Train TemporalTokenizer + ClassifierHead on MIT-BIH; save metrics + checkpoint."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data import loader
from src.models.temporal_encoder import TemporalTokenizer, ClassifierHead

ARTIFACTS = "artifacts"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(epochs: int = 8, batch_size: int = 256, lr: float = 1e-3, per_class: int = 2000):
    os.makedirs(ARTIFACTS, exist_ok=True)
    device = get_device()
    print("device:", device)

    # Train on a class-balanced subset (undersample the dominant Normal class);
    # evaluate on the full, naturally-imbalanced test split.
    X_tr, y_tr = loader.load_split(config.TRAIN_RECORDS, max_beats_per_record=None)
    X_tr, y_tr = loader.balance_classes(X_tr, y_tr, per_class=per_class, seed=0)
    X_te, y_te = loader.load_split(config.TEST_RECORDS, max_beats_per_record=None)
    print("train/test beats:", len(y_tr), len(y_te))
    print("train class counts:", np.bincount(y_tr, minlength=len(config.CLASSES)).tolist())
    print("test class counts:", np.bincount(y_te, minlength=len(config.CLASSES)).tolist())

    Xtr = torch.tensor(X_tr); ytr = torch.tensor(y_tr)
    Xte = torch.tensor(X_te).to(device); yte_np = y_te

    # class weights to counter imbalance
    counts = np.bincount(y_tr, minlength=len(config.CLASSES))
    weights = torch.tensor((counts.sum() / (counts + 1e-6)), dtype=torch.float32).to(device)

    backbone = TemporalTokenizer().to(device)
    head = ClassifierHead(num_classes=len(config.CLASSES)).to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    n = len(Xtr)
    for ep in range(epochs):
        backbone.train(); head.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx].to(device); yb = ytr[idx].to(device)
            opt.zero_grad()
            logits = head(backbone.pooled(xb))
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep+1}/{epochs} loss={total/n:.4f}")

    backbone.eval(); head.eval()
    with torch.no_grad():
        logits = head(backbone.pooled(Xte))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    acc = accuracy_score(yte_np, preds)
    f1 = f1_score(yte_np, preds, average="macro")
    try:
        auroc = roc_auc_score(yte_np, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")  # a class may be absent in test
    metrics = {"accuracy": float(acc), "macro_f1": float(f1), "macro_auroc": float(auroc)}
    print("metrics:", metrics)
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(yte_np, preds, labels=list(range(len(config.CLASSES))))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASSES))); ax.set_xticklabels(config.CLASSES)
    ax.set_yticks(range(len(config.CLASSES))); ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.colorbar(im)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.savefig(os.path.join(ARTIFACTS, "confusion_matrix.png"), bbox_inches="tight")

    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()},
               os.path.join(ARTIFACTS, "checkpoint.pt"))
    return metrics


if __name__ == "__main__":
    train()
