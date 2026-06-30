"""LLM reasoning client (OpenAI or Anthropic) with a deterministic template fallback."""
import json
import os
from src import config

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for a research prototype "
    "(not a medical device). Given an automated ECG analysis and patient context, "
    "assess cardiovascular risk and explain your reasoning step by step. "
    "Respond ONLY with JSON: "
    '{"risk": "Low|Moderate|High", "reasons": ["...", "..."], "confidence": 0.0}. '
    "Base every reason on the provided findings; do not invent measurements."
)


def _build_user_prompt(descriptor: str, patient_info: dict) -> str:
    return (
        f"Patient context: age={patient_info.get('age', 'unknown')}, "
        f"medical history={patient_info.get('history', 'none')}.\n\n"
        f"{descriptor}\n\n"
        "Return the JSON assessment now."
    )


def _select_provider(provider: str | None) -> str:
    """Resolve which provider to use: explicit arg > LLM_PROVIDER env > key presence."""
    chosen = (provider or os.environ.get("LLM_PROVIDER") or "").lower()
    if chosen in ("openai", "anthropic"):
        return chosen
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return ""


class LLMReasoner:
    """Generate risk reasoning via OpenAI or Anthropic, falling back to a rule-based template."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = None
        self.model = None
        self.client = None
        self.mode = "fallback"
        chosen = _select_provider(provider)
        try:
            if chosen == "openai" and os.environ.get("OPENAI_API_KEY"):
                import openai
                self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                self.provider = "openai"
                self.model = model or config.OPENAI_MODEL
                self.mode = "api"
            elif chosen == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                self.provider = "anthropic"
                self.model = model or config.CLAUDE_MODEL
                self.mode = "api"
        except Exception:
            self.client = None
            self.mode = "fallback"

    def explain(self, descriptor: str, patient_info: dict) -> dict:
        if self.mode == "api":
            try:
                if self.provider == "openai":
                    return self._explain_openai(descriptor, patient_info)
                return self._explain_anthropic(descriptor, patient_info)
            except Exception as e:
                report = self._explain_fallback(descriptor, patient_info)
                report["note"] = f"API error ({self.provider}), used fallback: {e}"
                return report
        return self._explain_fallback(descriptor, patient_info)

    def _explain_openai(self, descriptor: str, patient_info: dict) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(descriptor, patient_info)},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    def _explain_anthropic(self, descriptor: str, patient_info: dict) -> dict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(descriptor, patient_info)}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    def _explain_fallback(self, descriptor: str, patient_info: dict) -> dict:
        """Deterministic rule-based reasoning so the pipeline runs without an API key."""
        abnormal = any(k in descriptor for k in ["Ventricular", "Supraventricular"])
        history = str(patient_info.get("history", "")).lower()
        risky_history = any(h in history for h in ["hypertension", "cardiac", "infarct", "diabetes"])
        reasons = []
        if abnormal:
            reasons.append("Abnormal beat morphology detected in the ECG analysis.")
        else:
            reasons.append("ECG analysis indicates a predominantly normal beat type.")
        if "rhythm appears irregular" in descriptor:
            reasons.append("Rhythm appears irregular, which can signal arrhythmia.")
        if risky_history:
            reasons.append(f"Patient history ({patient_info.get('history')}) increases cardiovascular risk.")
        risk = "High" if (abnormal and risky_history) else "Moderate" if abnormal else "Low"
        return {"risk": risk, "reasons": reasons, "confidence": 0.6}
