"""1D-CNN temporal tokenizer + classifier head."""
import torch
import torch.nn as nn
from src import config


class TemporalTokenizer(nn.Module):
    """Encode a 1D ECG window into a sequence of temporal tokens (B x N x H)."""

    def __init__(self, hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)            # B x 1 x L
        feats = self.net(x)              # B x H x N'
        return feats.transpose(1, 2)     # B x N' x H

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward(x)
        return tokens.mean(dim=1)        # B x H


class ClassifierHead(nn.Module):
    """Linear classifier on pooled embedding."""

    def __init__(self, num_classes: int, hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled)
