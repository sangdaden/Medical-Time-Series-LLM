"""Extract embeddings from the trained backbone and plot t-SNE/PCA (RQ1 evidence)."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src import config
from src.data import loader
from src.models.temporal_encoder import TemporalTokenizer
from src.train import get_device, ARTIFACTS


def extract_embeddings(max_beats: int = 300):
    device = get_device()
    backbone = TemporalTokenizer().to(device)
    ckpt = torch.load(os.path.join(ARTIFACTS, "checkpoint.pt"), map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()

    X, y = loader.load_split(config.TEST_RECORDS, max_beats_per_record=max_beats)
    with torch.no_grad():
        emb = backbone.pooled(torch.tensor(X).to(device)).cpu().numpy()
    return emb, y


def plot_tsne():
    emb, y = extract_embeddings()
    coords = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_idx, name in enumerate(config.CLASSES):
        m = y == cls_idx
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=6, label=name, alpha=0.6)
    ax.legend(title="AAMI class"); ax.set_title("t-SNE of ECG temporal embeddings")
    out = os.path.join(ARTIFACTS, "tsne.png")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)
    return out


if __name__ == "__main__":
    plot_tsne()
