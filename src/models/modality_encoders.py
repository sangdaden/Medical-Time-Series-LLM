"""Per-modality encoders. Each exposes ``.out_dim`` so the framework can project it
into the shared multimodal space generically."""
import torch.nn as nn
from src import config
from src.models.temporal_encoder import TemporalTokenizer


class EcgEncoder(nn.Module):
    """1D-CNN tokenizer over a raw ECG beat window -> pooled embedding."""

    def __init__(self):
        super().__init__()
        self.backbone = TemporalTokenizer()
        self.out_dim = config.HIDDEN_DIM

    def forward(self, x):                 # x: B x WINDOW
        return self.backbone.pooled(x)


class VectorEncoder(nn.Module):
    """Small MLP for low-dimensional tabular/sequence modalities (RR trend, clinical)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)
