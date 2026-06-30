from src import features
from src import config


def test_build_descriptor_contains_class_and_confidence():
    d = features.build_descriptor(pred_idx=2, confidence=0.91, heart_rate=120)
    assert "Ventricular" in d or "V" in d
    assert "0.91" in d or "91" in d
    assert "120" in d


def test_faithfulness_flags_class_mention():
    descriptor = features.build_descriptor(pred_idx=2, confidence=0.9, heart_rate=120)
    report = {"risk": "High", "reasons": ["Ventricular ectopic beat detected", "Elevated heart rate"]}
    result = features.check_faithfulness(report, pred_idx=2, heart_rate=120)
    assert result["mentions_class"] is True
    assert result["faithful"] is True


def test_faithfulness_detects_unfaithful():
    report = {"risk": "Low", "reasons": ["Normal sinus rhythm"]}
    result = features.check_faithfulness(report, pred_idx=2, heart_rate=120)  # actually Ventricular
    assert result["mentions_class"] is False
    assert result["faithful"] is False
