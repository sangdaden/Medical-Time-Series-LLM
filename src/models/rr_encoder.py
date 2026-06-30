"""Encoder for the RR-interval (heart-rate trend) modality."""
import torch.nn as nn
from src import config


class RREncoder(nn.Module):
    """Map a short RR-interval sequence to an embedding (wearable-like modality)."""

    def __init__(self, in_dim: int = config.RR_CONTEXT, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, rr):
        return self.net(rr)
