"""Multimodal classifier: align ECG and RR-trend modalities into a shared space and fuse.

Setting ``use_rr=False`` runs the ECG-only baseline through the same code path, so the
two configurations are directly comparable (the point of the cross-modal experiment).
"""
import torch
import torch.nn as nn
from src import config
from src.models.temporal_encoder import TemporalTokenizer
from src.models.rr_encoder import RREncoder

SHARED_DIM = 128


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes: int = len(config.CLASSES), use_rr: bool = True,
                 shared_dim: int = SHARED_DIM):
        super().__init__()
        self.use_rr = use_rr
        self.ecg_backbone = TemporalTokenizer()
        self.proj_ecg = nn.Sequential(nn.Linear(config.HIDDEN_DIM, shared_dim), nn.ReLU())
        if use_rr:
            self.rr_encoder = RREncoder()
            self.proj_rr = nn.Sequential(nn.Linear(self.rr_encoder.out_dim, shared_dim), nn.ReLU())
            fused_dim = shared_dim * 2
        else:
            fused_dim = shared_dim
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, ecg, rr=None):
        e = self.proj_ecg(self.ecg_backbone.pooled(ecg))   # B x shared_dim
        if self.use_rr:
            r = self.proj_rr(self.rr_encoder(rr))           # B x shared_dim
            fused = torch.cat([e, r], dim=1)
        else:
            fused = e
        return self.head(fused)
