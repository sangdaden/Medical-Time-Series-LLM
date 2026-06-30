"""Generic multimodal model: encode each registered modality, align them in a shared
space, and fuse. Works for any list of modalities (1..N) — no per-modality code here.
"""
import torch
import torch.nn as nn
from src import config, framework
import src.modalities  # noqa: F401  (populates the registry)

SHARED_DIM = 128


class MultimodalModel(nn.Module):
    def __init__(self, modalities=None, num_classes: int = len(config.CLASSES),
                 shared_dim: int = SHARED_DIM):
        super().__init__()
        self.modalities = list(modalities if modalities is not None else config.MODALITIES)
        specs = framework.get_specs(self.modalities)
        self.encoders = nn.ModuleDict()
        self.projs = nn.ModuleDict()
        for s in specs:
            enc = s.make_encoder()
            self.encoders[s.name] = enc
            self.projs[s.name] = nn.Sequential(nn.Linear(enc.out_dim, shared_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(shared_dim * len(specs), num_classes),
        )

    def forward(self, feats: dict):
        """feats: {modality_name: tensor B×feat_dim}. Returns logits B×num_classes."""
        parts = [self.projs[name](self.encoders[name](feats[name])) for name in self.modalities]
        return self.head(torch.cat(parts, dim=1))
