import torch
from src.models.temporal_encoder import TemporalTokenizer, ClassifierHead
from src import config


def test_tokenizer_output_shape():
    model = TemporalTokenizer()
    x = torch.randn(4, config.WINDOW)          # B x L
    tokens = model(x)
    assert tokens.shape[0] == 4
    assert tokens.shape[2] == config.HIDDEN_DIM  # B x N x H


def test_pooled_shape():
    model = TemporalTokenizer()
    x = torch.randn(4, config.WINDOW)
    pooled = model.pooled(x)
    assert pooled.shape == (4, config.HIDDEN_DIM)


def test_classifier_head_shape():
    head = ClassifierHead(num_classes=len(config.CLASSES))
    pooled = torch.randn(4, config.HIDDEN_DIM)
    logits = head(pooled)
    assert logits.shape == (4, len(config.CLASSES))
