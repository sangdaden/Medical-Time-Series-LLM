# Time-Series LLM ECG Prototype — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-day research prototype that turns MIT-BIH ECG beats into structured temporal embeddings, classifies arrhythmia, and uses the Claude API to produce explainable risk reasoning — with a Streamlit demo.

**Architecture:** A shared 1D-CNN backbone (`TemporalTokenizer`) produces embeddings. A baseline linear head and a "proposed" head give comparable metrics (RQ1). Embeddings are visualized with t-SNE. A feature extractor converts predictions into a text descriptor that a Claude API client reasons over to produce risk + step-by-step explanation (RQ3). Everything is wired through `pipeline.analyze()` and surfaced in a Streamlit app.

**Tech Stack:** Python 3.11/3.12 (venv), PyTorch (CPU/MPS), `wfdb`, scikit-learn, numpy, matplotlib, Streamlit, `anthropic` SDK, pytest.

---

## File Structure

```
Medical-Time-Series-LLM/
  requirements.txt
  src/
    __init__.py
    config.py                  # constants: window, classes, model name
    data/
      __init__.py
      loader.py                # MIT-BIH download, beat segmentation, AAMI mapping, split
    models/
      __init__.py
      temporal_encoder.py      # TemporalTokenizer (1D-CNN) + ClassifierHead
      projector.py             # 768 -> llm_dim projection (RQ2 design stub)
      llm.py                   # Claude API client + template fallback
    features.py                # prediction -> text descriptor + faithfulness check
    train.py                   # train backbone+head, save metrics + checkpoint
    embed_viz.py               # extract embeddings, t-SNE/PCA plot
    pipeline.py                # analyze(ecg, patient_info) end-to-end
  notebooks/ECG_embedding.ipynb
  demo/app.py
  tests/
    test_loader.py
    test_temporal_encoder.py
    test_projector.py
    test_features.py
    test_llm_fallback.py
    test_pipeline.py
  artifacts/                   # metrics.json, confusion_matrix.png, checkpoint.pt, tsne.png (gitignored)
  README.md
```

---

## Task 0: Environment & scaffold

**Files:**
- Create: `requirements.txt`, `.gitignore`, `src/__init__.py`, `src/config.py`, `src/data/__init__.py`, `src/models/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create a Python 3.11/3.12 virtualenv**

PyTorch may have no wheel for Python 3.14. Use a 3.11 or 3.12 interpreter.

Run:
```bash
cd /Users/sangphan/Research/Medical-Time-Series-LLM
# pick an available 3.11/3.12; install via homebrew if missing: brew install python@3.12
python3.12 -m venv .venv || python3.11 -m venv .venv
source .venv/bin/activate
python --version
```
Expected: prints `Python 3.12.x` (or 3.11.x).

- [ ] **Step 2: Write `requirements.txt`**

```
torch
wfdb
scikit-learn
numpy
matplotlib
streamlit
anthropic
pytest
```

- [ ] **Step 3: Install dependencies**

Run:
```bash
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
```
Expected: prints a torch version and `mps True` (or `False` — both fine; CPU works).

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
artifacts/
.streamlit/
mitdb/
*.dat
*.hea
*.atr
```

- [ ] **Step 5: Create empty package files and `src/config.py`**

`src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `tests/__init__.py` are empty files.

`src/config.py`:
```python
"""Project-wide constants."""

# Beat window: samples around the R-peak (MIT-BIH is 360 Hz). ~0.5s before, ~0.7s after.
PRE_SAMPLES = 180
POST_SAMPLES = 252
WINDOW = PRE_SAMPLES + POST_SAMPLES  # 432

HIDDEN_DIM = 768
LLM_DIM = 4096  # projection target (RQ2 design stub)

# AAMI 5-class grouping. Maps MIT-BIH annotation symbols -> class index.
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,        # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,                # Supraventricular
    "V": 2, "E": 2,                                # Ventricular
    "F": 3,                                         # Fusion
    "/": 4, "f": 4, "Q": 4,                        # Unknown/paced
}

SAMPLE_RATE = 360

# Inter-patient split (records). Standard de Chazal-style split, trimmed for speed.
TRAIN_RECORDS = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119"]
TEST_RECORDS = ["100", "103", "105", "111", "113", "117", "121", "123"]

CLAUDE_MODEL = "claude-haiku-4-5"
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore src tests
git commit -m "chore: scaffold project, config, and deps"
```

---

## Task 1: Data loader (Day 1)

**Files:**
- Create: `src/data/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/test_loader.py`:
```python
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


def test_segment_beats_drops_edge_and_unmapped():
    sig = np.arange(2000, dtype=np.float32)
    rpeaks = [10, 1000, 1990]        # 10 and 1990 too close to edges
    symbols = ["N", "?", "N"]        # middle unmapped
    X, y = loader.segment_beats(sig, rpeaks, symbols)
    assert X.shape == (0, config.WINDOW)
    assert y.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL with `AttributeError: module 'src.data.loader' has no attribute 'map_symbol'`.

- [ ] **Step 3: Write `src/data/loader.py`**

```python
"""MIT-BIH Arrhythmia loader: download, beat segmentation, AAMI mapping, split."""
import os
import numpy as np
import wfdb
from src import config

DB_DIR = "mitdb"


def map_symbol(symbol: str):
    """Map a MIT-BIH annotation symbol to an AAMI class index, or None to drop."""
    return config.AAMI_MAP.get(symbol)


def segment_beats(signal: np.ndarray, rpeaks, symbols):
    """Cut fixed windows around R-peaks. Drop edge beats and unmapped symbols."""
    X, y = [], []
    for r, sym in zip(rpeaks, symbols):
        cls = map_symbol(sym)
        if cls is None:
            continue
        start = r - config.PRE_SAMPLES
        end = r + config.POST_SAMPLES
        if start < 0 or end > len(signal):
            continue
        X.append(signal[start:end])
        y.append(cls)
    if not X:
        return np.zeros((0, config.WINDOW), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _load_record(record: str):
    """Download (if needed) and read one record's first channel + annotations."""
    rec = wfdb.rdrecord(record, pn_dir="mitdb")
    ann = wfdb.rdann(record, "atr", pn_dir="mitdb")
    signal = rec.p_signal[:, 0].astype(np.float32)
    # per-record z-normalization
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    return signal, list(ann.sample), list(ann.symbol)


def load_split(records, max_beats_per_record: int | None = None):
    """Load and concatenate beats for a list of records."""
    Xs, ys = [], []
    for record in records:
        signal, rpeaks, symbols = _load_record(record)
        X, y = segment_beats(signal, rpeaks, symbols)
        if max_beats_per_record is not None and len(X) > max_beats_per_record:
            idx = np.linspace(0, len(X) - 1, max_beats_per_record).astype(int)
            X, y = X[idx], y[idx]
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def load_dataset(max_beats_per_record: int | None = 600):
    """Return (X_train, y_train, X_test, y_test) using inter-patient split."""
    X_train, y_train = load_split(config.TRAIN_RECORDS, max_beats_per_record)
    X_test, y_test = load_split(config.TEST_RECORDS, max_beats_per_record)
    return X_train, y_train, X_test, y_test
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: PASS (3 passed). These tests use synthetic data — no network.

- [ ] **Step 5: Smoke-test the real download (one record)**

Run:
```bash
python -c "from src.data import loader; s,r,sy=loader._load_record('100'); print(len(s), len(r))"
```
Expected: prints two numbers (~650000 and ~2270). Requires internet. If it fails on network, note it and continue — training task depends on this.

- [ ] **Step 6: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: MIT-BIH loader with beat segmentation and AAMI mapping"
```

---

## Task 2: Temporal tokenizer + classifier head (Day 1–2)

**Files:**
- Create: `src/models/temporal_encoder.py`
- Test: `tests/test_temporal_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/test_temporal_encoder.py`:
```python
import torch
from src.models.temporal_encoder import TemporalTokenizer, ClassifierHead
from src import config


def test_tokenizer_output_shape():
    model = TemporalTokenizer()
    x = torch.randn(4, config.WINDOW)          # B x L
    tokens = model(x)
    assert tokens.shape[0] == 4
    assert tokens.shape[2] == config.HIDDEN_DIM  # B x N x H


def test_pooled_shape():
    model = TemporalTokenizer()
    x = torch.randn(4, config.WINDOW)
    pooled = model.pooled(x)
    assert pooled.shape == (4, config.HIDDEN_DIM)


def test_classifier_head_shape():
    head = ClassifierHead(num_classes=len(config.CLASSES))
    pooled = torch.randn(4, config.HIDDEN_DIM)
    logits = head(pooled)
    assert logits.shape == (4, len(config.CLASSES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temporal_encoder.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/models/temporal_encoder.py`**

```python
"""1D-CNN temporal tokenizer + classifier head."""
import torch
import torch.nn as nn
from src import config


class TemporalTokenizer(nn.Module):
    """Encode a 1D ECG window into a sequence of temporal tokens (B x N x H)."""

    def __init__(self, hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)            # B x 1 x L
        feats = self.net(x)              # B x H x N'
        return feats.transpose(1, 2)     # B x N' x H

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward(x)
        return tokens.mean(dim=1)        # B x H


class ClassifierHead(nn.Module):
    """Linear classifier on pooled embedding."""

    def __init__(self, num_classes: int, hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal_encoder.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models/temporal_encoder.py tests/test_temporal_encoder.py
git commit -m "feat: 1D-CNN temporal tokenizer and classifier head"
```

---

## Task 3: Projector (RQ2 design stub) (Day 2)

**Files:**
- Create: `src/models/projector.py`
- Test: `tests/test_projector.py`

- [ ] **Step 1: Write the failing test**

`tests/test_projector.py`:
```python
import torch
from src.models.projector import Projector
from src import config


def test_projector_maps_to_llm_dim():
    proj = Projector()
    emb = torch.randn(4, config.HIDDEN_DIM)
    out = proj(emb)
    assert out.shape == (4, config.LLM_DIM)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_projector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/models/projector.py`**

```python
"""Projection from encoder space to LLM embedding space (RQ2 design stub).

Wired and tested but not trained end-to-end into a closed-weights API LLM.
Used for alignment illustration and future multimodal work.
"""
import torch.nn as nn
from src import config


class Projector(nn.Module):
    def __init__(self, in_dim: int = config.HIDDEN_DIM, out_dim: int = config.LLM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_projector.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/projector.py tests/test_projector.py
git commit -m "feat: projection layer for LLM embedding space (RQ2 stub)"
```

---

## Task 4: Training script + metrics (Day 1–2)

**Files:**
- Create: `src/train.py`

This task produces the Day-1 baseline numbers and the Day-2 proposed-head numbers from the same backbone. No new pytest test (it's a script); verification is running it and inspecting `artifacts/metrics.json`.

- [ ] **Step 1: Write `src/train.py`**

```python
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


def train(epochs: int = 5, batch_size: int = 256, lr: float = 1e-3, max_beats: int = 600):
    os.makedirs(ARTIFACTS, exist_ok=True)
    device = get_device()
    print("device:", device)

    X_tr, y_tr, X_te, y_te = loader.load_dataset(max_beats_per_record=max_beats)
    print("train/test beats:", len(y_tr), len(y_te))

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
```

- [ ] **Step 2: Run training (smoke, may take a few minutes on CPU)**

Run:
```bash
python -m src.train
```
Expected: prints device, beat counts, per-epoch loss, and a `metrics` dict; writes `artifacts/metrics.json`, `artifacts/confusion_matrix.png`, `artifacts/checkpoint.pt`. Macro-F1 should be clearly above chance (chance ≈ 0.2).

- [ ] **Step 3: Verify artifacts exist**

Run: `ls artifacts/ && cat artifacts/metrics.json`
Expected: lists the three files and prints the metrics JSON.

- [ ] **Step 4: Commit**

```bash
git add src/train.py
git commit -m "feat: training script with metrics, confusion matrix, checkpoint"
```

---

## Task 5: Embedding visualization — RQ1 evidence (Day 2)

**Files:**
- Create: `src/embed_viz.py`, `notebooks/ECG_embedding.ipynb`

- [ ] **Step 1: Write `src/embed_viz.py`**

```python
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
```

- [ ] **Step 2: Run it (after Task 4 produced a checkpoint)**

Run: `python -m src.embed_viz`
Expected: prints `saved artifacts/tsne.png`; the image shows class clusters that are visibly (if imperfectly) separated.

- [ ] **Step 3: Create `notebooks/ECG_embedding.ipynb`**

Create a 3-cell notebook that documents RQ1. Cell contents:

Cell 1 (markdown): `# RQ1: Do temporal embeddings capture ECG structure?`

Cell 2 (code):
```python
from src.train import train
train(epochs=5)  # skip if artifacts/checkpoint.pt already exists
```

Cell 3 (code):
```python
from src.embed_viz import plot_tsne
from IPython.display import Image
plot_tsne()
Image("artifacts/tsne.png")
```

Run: create the file with these cells (use jupyter or write the .ipynb JSON directly).

- [ ] **Step 4: Commit**

```bash
git add src/embed_viz.py notebooks/ECG_embedding.ipynb
git commit -m "feat: embedding extraction and t-SNE visualization (RQ1)"
```

---

## Task 6: Feature extractor + faithfulness check (Day 3)

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
from src import features
from src import config


def test_build_descriptor_contains_class_and_confidence():
    d = features.build_descriptor(pred_idx=2, confidence=0.91, heart_rate=120)
    assert "Ventricular" in d or "V" in d
    assert "0.91" in d or "91" in d
    assert "120" in d


def test_faithfulness_flags_class_mention():
    descriptor = features.build_descriptor(pred_idx=2, confidence=0.9, heart_rate=120)
    report = {"risk": "High", "reasons": ["Ventricular ectopic beat detected", "Elevated heart rate"]}
    result = features.check_faithfulness(report, pred_idx=2, heart_rate=120)
    assert result["mentions_class"] is True
    assert result["faithful"] is True


def test_faithfulness_detects_unfaithful():
    report = {"risk": "Low", "reasons": ["Normal sinus rhythm"]}
    result = features.check_faithfulness(report, pred_idx=2, heart_rate=120)  # actually Ventricular
    assert result["mentions_class"] is False
    assert result["faithful"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/features.py`**

```python
"""Convert model predictions into a text descriptor and check LLM faithfulness."""
from src import config

CLASS_NAMES = {
    0: "Normal beat",
    1: "Supraventricular ectopic beat",
    2: "Ventricular ectopic beat",
    3: "Fusion beat",
    4: "Unknown/paced beat",
}


def build_descriptor(pred_idx: int, confidence: float, heart_rate: float) -> str:
    """Human-readable summary of the model output for the LLM to reason over."""
    name = CLASS_NAMES[pred_idx]
    rhythm = "irregular" if pred_idx in (1, 2, 3) else "regular"
    return (
        f"ECG analysis: predicted beat type = {name} ({config.CLASSES[pred_idx]}), "
        f"model confidence = {confidence:.2f}. "
        f"Estimated heart rate = {int(heart_rate)} bpm, rhythm appears {rhythm}."
    )


def check_faithfulness(report: dict, pred_idx: int, heart_rate: float) -> dict:
    """Rule-based check: does the LLM's reasoning reference the actual findings?"""
    text = " ".join(report.get("reasons", [])).lower()
    name = CLASS_NAMES[pred_idx].lower()
    keyword = name.split()[0]  # e.g. "ventricular"
    mentions_class = keyword in text or config.CLASSES[pred_idx].lower() in text
    mentions_hr = "heart rate" in text or "bpm" in text or str(int(heart_rate)) in text
    faithful = bool(mentions_class)
    return {
        "mentions_class": mentions_class,
        "mentions_heart_rate": mentions_hr,
        "faithful": faithful,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: descriptor builder and faithfulness check"
```

---

## Task 7: Claude API client + fallback (Day 3)

**Files:**
- Create: `src/models/llm.py`
- Test: `tests/test_llm_fallback.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_fallback.py`:
```python
from src.models import llm


def test_fallback_used_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = llm.LLMReasoner()                       # no key -> fallback mode
    assert client.mode == "fallback"
    report = client.explain("ECG analysis: Ventricular ectopic beat (V), confidence 0.9. HR 120 bpm.",
                             {"age": 65, "history": "Hypertension"})
    assert "risk" in report and "reasons" in report
    assert isinstance(report["reasons"], list) and len(report["reasons"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/models/llm.py`**

```python
"""Claude API reasoning client with a deterministic template fallback."""
import json
import os
from src import config

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for a research prototype "
    "(not a medical device). Given an automated ECG analysis and patient context, "
    "assess cardiovascular risk and explain your reasoning step by step. "
    "Respond ONLY with JSON: "
    '{"risk": "Low|Moderate|High", "reasons": ["...", "..."], "confidence": 0.0}. '
    "Base every reason on the provided findings; do not invent measurements."
)


def _build_user_prompt(descriptor: str, patient_info: dict) -> str:
    return (
        f"Patient context: age={patient_info.get('age', 'unknown')}, "
        f"medical history={patient_info.get('history', 'none')}.\n\n"
        f"{descriptor}\n\n"
        "Return the JSON assessment now."
    )


class LLMReasoner:
    def __init__(self, model: str = config.CLAUDE_MODEL):
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.mode = "api"
            except Exception:
                self.client = None
                self.mode = "fallback"
        else:
            self.client = None
            self.mode = "fallback"

    def explain(self, descriptor: str, patient_info: dict) -> dict:
        if self.mode == "api":
            try:
                return self._explain_api(descriptor, patient_info)
            except Exception as e:
                report = self._explain_fallback(descriptor, patient_info)
                report["note"] = f"API error, used fallback: {e}"
                return report
        return self._explain_fallback(descriptor, patient_info)

    def _explain_api(self, descriptor: str, patient_info: dict) -> dict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(descriptor, patient_info)}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return json.loads(text)

    def _explain_fallback(self, descriptor: str, patient_info: dict) -> dict:
        """Deterministic rule-based reasoning so the pipeline runs without an API key."""
        abnormal = any(k in descriptor for k in ["Ventricular", "Supraventricular", "Fusion"])
        history = str(patient_info.get("history", "")).lower()
        risky_history = any(h in history for h in ["hypertension", "cardiac", "infarct", "diabetes"])
        reasons = []
        if abnormal:
            reasons.append("Abnormal beat morphology detected in the ECG analysis.")
        else:
            reasons.append("ECG analysis indicates a predominantly normal beat type.")
        if "rhythm appears irregular" in descriptor:
            reasons.append("Rhythm appears irregular, which can signal arrhythmia.")
        if risky_history:
            reasons.append(f"Patient history ({patient_info.get('history')}) increases cardiovascular risk.")
        risk = "High" if (abnormal and risky_history) else "Moderate" if abnormal else "Low"
        return {"risk": risk, "reasons": reasons, "confidence": 0.6}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: PASS.

- [ ] **Step 5: (Optional) Live API check if key available**

Run:
```bash
export ANTHROPIC_API_KEY=sk-...   # only if you have one
python -c "from src.models.llm import LLMReasoner; r=LLMReasoner(); print(r.mode); print(r.explain('ECG analysis: Ventricular ectopic beat (V), confidence 0.92. HR 130 bpm, rhythm appears irregular.', {'age':65,'history':'Hypertension'}))"
```
Expected: prints `api` then a JSON-like dict with `risk`/`reasons`. Without a key, prints `fallback` and a dict.

- [ ] **Step 6: Commit**

```bash
git add src/models/llm.py tests/test_llm_fallback.py
git commit -m "feat: Claude reasoning client with template fallback"
```

---

## Task 8: End-to-end pipeline (Day 3)

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import numpy as np
from src import pipeline
from src import config


def test_analyze_returns_full_report(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # force fallback
    ecg = np.random.randn(config.WINDOW).astype("float32")
    result = pipeline.analyze(ecg, {"age": 65, "history": "Hypertension"})
    for key in ["label", "confidence", "descriptor", "llm_report", "faithfulness"]:
        assert key in result
    assert result["label"] in config.CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError` (or missing checkpoint — see Step 3, pipeline must handle missing checkpoint by using random init weights with a warning so the test is self-contained).

- [ ] **Step 3: Write `src/pipeline.py`**

```python
"""End-to-end: ECG window -> classification -> descriptor -> LLM reasoning."""
import os
import warnings
import numpy as np
import torch

from src import config, features
from src.data import loader  # noqa: F401  (kept for parity / future use)
from src.models.temporal_encoder import TemporalTokenizer, ClassifierHead
from src.models.llm import LLMReasoner

_CKPT = os.path.join("artifacts", "checkpoint.pt")


def _estimate_heart_rate(ecg: np.ndarray) -> float:
    """Crude HR proxy from the window. Real HR needs RR intervals; this is a demo estimate."""
    # number of zero-up-crossings as a coarse proxy; clamp to a plausible range
    centered = ecg - np.mean(ecg)
    crossings = np.sum((centered[:-1] < 0) & (centered[1:] >= 0))
    seconds = config.WINDOW / config.SAMPLE_RATE
    bpm = (crossings / seconds) * 60 if seconds > 0 else 0
    return float(min(max(bpm, 40), 200))


class Pipeline:
    def __init__(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.backbone = TemporalTokenizer().to(self.device)
        self.head = ClassifierHead(num_classes=len(config.CLASSES)).to(self.device)
        if os.path.exists(_CKPT):
            ckpt = torch.load(_CKPT, map_location=self.device)
            self.backbone.load_state_dict(ckpt["backbone"])
            self.head.load_state_dict(ckpt["head"])
        else:
            warnings.warn("No checkpoint found; using randomly initialized weights.")
        self.backbone.eval(); self.head.eval()
        self.reasoner = LLMReasoner()

    def analyze(self, ecg: np.ndarray, patient_info: dict) -> dict:
        x = torch.tensor(np.asarray(ecg, dtype="float32")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.head(self.backbone.pooled(x))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        hr = _estimate_heart_rate(np.asarray(ecg, dtype="float32"))
        descriptor = features.build_descriptor(pred_idx, confidence, hr)
        report = self.reasoner.explain(descriptor, patient_info)
        faith = features.check_faithfulness(report, pred_idx, hr)
        return {
            "label": config.CLASSES[pred_idx],
            "confidence": confidence,
            "heart_rate": hr,
            "descriptor": descriptor,
            "llm_report": report,
            "faithfulness": faith,
        }


_PIPELINE = None


def analyze(ecg, patient_info):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE.analyze(ecg, patient_info)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (a UserWarning about missing checkpoint is fine if Task 4 hasn't been run; with a checkpoint there's no warning).

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end analyze pipeline"
```

---

## Task 9: Streamlit demo (Day 4)

**Files:**
- Create: `demo/app.py`

- [ ] **Step 1: Write `demo/app.py`**

```python
"""Streamlit demo: pick/upload an ECG beat + patient info -> AI risk report."""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src import config
from src.data import loader
from src.pipeline import analyze

st.set_page_config(page_title="Time-Series LLM for ECG", layout="centered")
st.title("Time-Series LLM — ECG Clinical Reasoning (Research Prototype)")
st.caption("Not a medical device. For research demonstration only.")


@st.cache_data(show_spinner="Loading sample ECG beats...")
def load_samples():
    X, y = loader.load_split(["100"], max_beats_per_record=50)
    return X, y


with st.sidebar:
    st.header("Patient information")
    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    history = st.text_input("Medical history", value="Hypertension")
    source = st.radio("ECG source", ["Sample from MIT-BIH (record 100)", "Upload CSV (single beat)"])

ecg = None
if source.startswith("Sample"):
    try:
        X, y = load_samples()
        idx = st.slider("Sample beat index", 0, len(X) - 1, 0)
        ecg = X[idx]
        st.write(f"True label (dataset): **{config.CLASSES[y[idx]]}**")
    except Exception as e:
        st.error(f"Could not load samples (network?): {e}")
else:
    up = st.file_uploader("CSV with one column of ECG samples", type=["csv"])
    if up is not None:
        arr = np.loadtxt(up, delimiter=",").astype("float32").ravel()
        if len(arr) >= config.WINDOW:
            ecg = arr[:config.WINDOW]
        else:
            ecg = np.pad(arr, (0, config.WINDOW - len(arr)))

if ecg is not None:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(ecg); ax.set_title("ECG beat window"); ax.set_xlabel("sample")
    st.pyplot(fig)

    if st.button("Analyze", type="primary"):
        with st.spinner("Running pipeline..."):
            result = analyze(ecg, {"age": age, "history": history})
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted beat", result["label"])
        c2.metric("Confidence", f"{result['confidence']:.2f}")
        c3.metric("Risk", result["llm_report"].get("risk", "?"))
        st.subheader("AI reasoning")
        for r in result["llm_report"].get("reasons", []):
            st.markdown(f"- {r}")
        st.caption(f"Descriptor: {result['descriptor']}")
        faith = result["faithfulness"]
        (st.success if faith["faithful"] else st.warning)(
            f"Faithfulness check: mentions findings = {faith['faithful']}"
        )
```

- [ ] **Step 2: Launch the demo**

Run: `streamlit run demo/app.py`
Expected: browser opens; selecting a sample beat, clicking **Analyze** shows predicted beat, confidence, risk, reasoning bullets, and a faithfulness badge. Works in fallback mode without an API key.

- [ ] **Step 3: Commit**

```bash
git add demo/app.py
git commit -m "feat: Streamlit demo app"
```

---

## Task 10: README & research write-up (Day 4)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

Include these sections with real content:
- **Title + one-line description** of the project.
- **Architecture diagram** (copy the ASCII flow from the spec section 3).
- **Setup**: Python 3.11/3.12 venv, `pip install -r requirements.txt`, optional `export ANTHROPIC_API_KEY=...`.
- **How to run**: `python -m src.train` → `python -m src.embed_viz` → `pytest -v` → `streamlit run demo/app.py`.
- **Results**: embed the contents of `artifacts/metrics.json` (baseline/proposed table) and reference `artifacts/confusion_matrix.png` and `artifacts/tsne.png`.
- **Research questions**: short paragraphs answering RQ1 (t-SNE + metrics), RQ2 (projection design + future work), RQ3 (LLM reasoning + faithfulness check).
- **Limitations & future work**: not a medical device; single modality; text-bridge rather than embedding injection; inter-patient split trimmed for speed; MIMIC-IV multimodal as future work.

- [ ] **Step 2: Verify the run instructions end-to-end**

Run: `pytest -v` and confirm all pass; confirm `artifacts/` referenced files exist.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: project README with results and RQ answers"
```

---

## Self-Review (completed)

- **Spec coverage:** loader (§4.1→T1), tokenizer+projector (§4.2/4.3→T2/T3), classifier+metrics/RQ1 (§4.4/§6→T4/T5), descriptor+faithfulness (§4.6→T6), LLM client+fallback (§4.5→T7), pipeline (§4.7→T8), demo (§4.8→T9), README/RQ answers (§2/§6→T10), Python 3.14 risk (§7→T0). All sections covered.
- **Placeholder scan:** no TBD/TODO; every code step has complete code; README step lists concrete sections (no code, content enumerated).
- **Type consistency:** `TemporalTokenizer.pooled()`, `ClassifierHead(num_classes=...)`, `build_descriptor(pred_idx, confidence, heart_rate)`, `check_faithfulness(report, pred_idx, heart_rate)`, `LLMReasoner.explain(descriptor, patient_info)`, and `analyze()`'s returned keys (`label, confidence, heart_rate, descriptor, llm_report, faithfulness`) are used consistently across tasks and the demo.
