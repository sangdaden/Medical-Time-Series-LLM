"""Claude API reasoning client with a deterministic template fallback."""
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


class LLMReasoner:
    def __init__(self, model: str = config.CLAUDE_MODEL):
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.mode = "api"
            except Exception:
                self.client = None
                self.mode = "fallback"
        else:
            self.client = None
            self.mode = "fallback"

    def explain(self, descriptor: str, patient_info: dict) -> dict:
        if self.mode == "api":
            try:
                return self._explain_api(descriptor, patient_info)
            except Exception as e:
                report = self._explain_fallback(descriptor, patient_info)
                report["note"] = f"API error, used fallback: {e}"
                return report
        return self._explain_fallback(descriptor, patient_info)

    def _explain_api(self, descriptor: str, patient_info: dict) -> dict:
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
