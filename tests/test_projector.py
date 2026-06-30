import torch
from src.models.projector import Projector
from src import config


def test_projector_maps_to_llm_dim():
    proj = Projector()
    emb = torch.randn(4, config.HIDDEN_DIM)
    out = proj(emb)
    assert out.shape == (4, config.LLM_DIM)
