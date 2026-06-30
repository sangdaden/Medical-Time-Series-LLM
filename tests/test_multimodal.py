import torch
from src import config
from src.models.rr_encoder import RREncoder
from src.models.multimodal import MultimodalClassifier


def test_rr_encoder_shape():
    enc = RREncoder()
    out = enc(torch.randn(4, config.RR_CONTEXT))
    assert out.shape == (4, 64)


def test_multimodal_forward_with_rr():
    model = MultimodalClassifier(use_rr=True)
    ecg = torch.randn(4, config.WINDOW)
    rr = torch.randn(4, config.RR_CONTEXT)
    logits = model(ecg, rr)
    assert logits.shape == (4, len(config.CLASSES))


def test_unimodal_forward_without_rr():
    model = MultimodalClassifier(use_rr=False)
    ecg = torch.randn(4, config.WINDOW)
    logits = model(ecg)                       # rr not required
    assert logits.shape == (4, len(config.CLASSES))
