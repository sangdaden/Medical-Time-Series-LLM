"""Modality-registry framework.

A *modality* = a per-beat feature extractor + an encoder that maps that feature to an
embedding. Register one and it plugs into both the data loader (`load_multimodal`) and
the model (`MultimodalModel`) without touching any core code — that is what makes this
an extensible framework rather than a fixed two-modality pipeline.

To add a modality:
    1. Write a `prepare(signal, rpeaks, symbols, header) -> ctx` function.
    2. Write a `beat_feature(ctx, j, r) -> np.ndarray[feat_dim]` function.
    3. Write/choose an encoder `nn.Module` exposing `.out_dim`.
    4. `register(ModalitySpec(name, feat_dim, prepare, beat_feature, make_encoder))`.
Then add its name to `config.MODALITIES`.
"""
from dataclasses import dataclass
from typing import Callable

REGISTRY: dict = {}


@dataclass
class ModalitySpec:
    name: str
    feat_dim: int                 # length of the per-beat feature vector
    prepare: Callable             # (signal, rpeaks, symbols, header) -> ctx
    beat_feature: Callable        # (ctx, j, r) -> np.ndarray[feat_dim]
    make_encoder: Callable        # () -> nn.Module with attribute .out_dim


def register(spec: ModalitySpec) -> None:
    REGISTRY[spec.name] = spec


def get_specs(names):
    """Return the registered specs for ``names`` (raises KeyError on unknown modality)."""
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise KeyError(f"Unknown modality(ies): {missing}. Registered: {list(REGISTRY)}")
    return [REGISTRY[n] for n in names]
