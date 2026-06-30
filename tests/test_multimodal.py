import torch
from src import config
from src.models.modality_encoders import VectorEncoder
from src.models.multimodal import MultimodalModel, SHARED_DIM


def test_vector_encoder_shape():
    enc = VectorEncoder(config.RR_CONTEXT, 64)
    out = enc(torch.randn(4, config.RR_CONTEXT))
    assert out.shape == (4, 64)


def test_multimodal_forward_all_modalities():
    model = MultimodalModel(["ecg", "rr", "clinical"])
    feats = {
        "ecg": torch.randn(4, config.WINDOW),
        "rr": torch.randn(4, config.RR_CONTEXT),
        "clinical": torch.randn(4, 3),
    }
    logits = model(feats)
    assert logits.shape == (4, len(config.CLASSES))


def test_multimodal_single_modality():
    model = MultimodalModel(["ecg"])
    logits = model({"ecg": torch.randn(4, config.WINDOW)})
    assert logits.shape == (4, len(config.CLASSES))
    # one modality -> fused dim is a single shared block
    assert model.head[-1].in_features == SHARED_DIM
