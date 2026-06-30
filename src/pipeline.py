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
