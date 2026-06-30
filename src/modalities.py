"""Concrete modalities matching the project description, registered into the framework:

- ``ecg``      physiological signal  — raw beat waveform
- ``rr``       wearable sensor data  — RR-interval (heart-rate) trend
- ``clinical`` clinical records      — age, sex, medication count (from MIT-BIH headers)

Importing this module populates the framework REGISTRY.
"""
import numpy as np
from src import config
from src.framework import ModalitySpec, register
from src.data import loader
from src.models.modality_encoders import EcgEncoder, VectorEncoder

CLINICAL_DIM = 3


# --- ECG (physiological) ------------------------------------------------------
def _ecg_prepare(signal, rpeaks, symbols, header):
    return signal


def _ecg_beat(ctx, j, r):
    return ctx[r - config.PRE_SAMPLES : r + config.POST_SAMPLES]


# --- RR trend (wearable) ------------------------------------------------------
def _rr_prepare(signal, rpeaks, symbols, header):
    return loader._rr_series(np.asarray(rpeaks))


def _rr_beat(ctx, j, r):
    return loader._rr_window(ctx, j, config.RR_CONTEXT)


# --- Clinical records ---------------------------------------------------------
def parse_clinical(comments) -> np.ndarray:
    """Parse [age/100, sex(M=1/F=0/unknown=0.5), n_meds/5] from MIT-BIH header comments.

    The medication count uses a simple heuristic (a short comma-separated, period-free
    line); narrative notes are treated as 0 medications. Good enough for a coarse
    clinical-context feature, not a clinical parser.
    """
    age, sex, n_meds = 60.0, 0.5, 0.0
    if comments:
        toks = comments[0].split()
        if len(toks) >= 2:
            try:
                age = float(toks[0])
            except ValueError:
                pass
            sex = 1.0 if toks[1].upper().startswith("M") else 0.0
        if len(comments) >= 2:
            line = comments[1]
            if "." not in line and len(line) < 60:   # a medication list, not prose
                n_meds = float(len([t for t in line.split(",") if t.strip()]))
    return np.array([age / 100.0, sex, n_meds / 5.0], dtype=np.float32)


def _clinical_prepare(signal, rpeaks, symbols, header):
    return parse_clinical(header.get("comments"))


def _clinical_beat(ctx, j, r):
    return ctx.copy()                                # static per record; copy to be safe


register(ModalitySpec("ecg", config.WINDOW, _ecg_prepare, _ecg_beat, lambda: EcgEncoder()))
register(ModalitySpec("rr", config.RR_CONTEXT, _rr_prepare, _rr_beat,
                      lambda: VectorEncoder(config.RR_CONTEXT, 64)))
register(ModalitySpec("clinical", CLINICAL_DIM, _clinical_prepare, _clinical_beat,
                      lambda: VectorEncoder(CLINICAL_DIM, 32)))
