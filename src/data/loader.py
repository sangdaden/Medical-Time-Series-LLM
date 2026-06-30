"""MIT-BIH Arrhythmia loader: download, beat segmentation, AAMI mapping, split."""
import os
import numpy as np
import wfdb
from src import config

DB_DIR = "mitdb"
LOCAL_DIR = "mitdb"


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
    """Read one record's first channel + annotations.

    Uses the local cache in ``LOCAL_DIR`` if the record is present there
    (fast, offline); otherwise streams it from PhysioNet.
    """
    local = os.path.join(LOCAL_DIR, record)
    if os.path.exists(local + ".dat"):
        rec = wfdb.rdrecord(local)
        ann = wfdb.rdann(local, "atr")
    else:
        rec = wfdb.rdrecord(record, pn_dir="mitdb")
        ann = wfdb.rdann(record, "atr", pn_dir="mitdb")
    signal = rec.p_signal[:, 0].astype(np.float32)
    # per-record z-normalization
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    return signal, list(ann.sample), list(ann.symbol)


def balance_classes(X: np.ndarray, y: np.ndarray, per_class: int, seed: int = 0):
    """Cap each class to at most ``per_class`` samples (random undersampling)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(len(config.CLASSES)):
        ci = np.where(y == c)[0]
        if len(ci) > per_class:
            ci = rng.choice(ci, per_class, replace=False)
        chosen.append(ci)
    idx = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    rng.shuffle(idx)
    return X[idx], y[idx]


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
