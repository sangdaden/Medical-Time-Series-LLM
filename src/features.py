"""Convert model predictions into a text descriptor and check LLM faithfulness."""
from src import config

CLASS_NAMES = {
    0: "Normal beat",
    1: "Supraventricular ectopic beat",
    2: "Ventricular ectopic beat",
}


def build_descriptor(pred_idx: int, confidence: float, heart_rate: float) -> str:
    """Human-readable summary of the model output for the LLM to reason over."""
    name = CLASS_NAMES[pred_idx]
    rhythm = "irregular" if pred_idx in (1, 2) else "regular"
    return (
        f"ECG analysis: predicted beat type = {name} ({config.CLASSES[pred_idx]}), "
        f"model confidence = {confidence:.2f}. "
        f"Estimated heart rate = {int(heart_rate)} bpm, rhythm appears {rhythm}."
    )


def check_faithfulness(report: dict, pred_idx: int, heart_rate: float) -> dict:
    """Rule-based check: does the LLM's reasoning reference the actual findings?"""
    text = " ".join(report.get("reasons", [])).lower()
    name = CLASS_NAMES[pred_idx].lower()
    keyword = name.split()[0]  # e.g. "ventricular"
    mentions_class = keyword in text or config.CLASSES[pred_idx].lower() in text
    mentions_hr = "heart rate" in text or "bpm" in text or str(int(heart_rate)) in text
    faithful = bool(mentions_class)
    return {
        "mentions_class": mentions_class,
        "mentions_heart_rate": mentions_hr,
        "faithful": faithful,
    }
