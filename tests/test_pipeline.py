import numpy as np
from src import pipeline
from src import config


def test_analyze_returns_full_report(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # force fallback
    ecg = np.random.randn(config.WINDOW).astype("float32")
    result = pipeline.analyze(ecg, {"age": 65, "history": "Hypertension"})
    for key in ["label", "confidence", "descriptor", "llm_report", "faithfulness"]:
        assert key in result
    assert result["label"] in config.CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
