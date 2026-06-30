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
