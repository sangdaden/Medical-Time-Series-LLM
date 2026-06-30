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
