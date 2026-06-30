from src.models import llm


def _clear_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_fallback_used_without_api_key(monkeypatch):
    _clear_keys(monkeypatch)
    client = llm.LLMReasoner()                       # no key -> fallback mode
    assert client.mode == "fallback"
    assert client.provider is None
    report = client.explain("ECG analysis: Ventricular ectopic beat (V), confidence 0.9. HR 120 bpm.",
                             {"age": 65, "history": "Hypertension"})
    assert "risk" in report and "reasons" in report
    assert isinstance(report["reasons"], list) and len(report["reasons"]) >= 1


def test_selects_openai_when_key_present(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = llm.LLMReasoner()
    assert client.provider == "openai"
    assert client.mode == "api"
    assert client.model == "gpt-4o-mini"


def test_llm_provider_env_overrides_to_anthropic(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = llm.LLMReasoner()
    assert client.provider == "anthropic"
    assert client.mode == "api"
