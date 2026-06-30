from src.models import llm


def test_fallback_used_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = llm.LLMReasoner()                       # no key -> fallback mode
    assert client.mode == "fallback"
    report = client.explain("ECG analysis: Ventricular ectopic beat (V), confidence 0.9. HR 120 bpm.",
                             {"age": 65, "history": "Hypertension"})
    assert "risk" in report and "reasons" in report
    assert isinstance(report["reasons"], list) and len(report["reasons"]) >= 1
