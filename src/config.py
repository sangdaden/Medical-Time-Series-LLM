"""Project-wide constants."""
import os

from dotenv import load_dotenv

# Load API keys / settings from a .env file at the repo root (if present), so
# OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_PROVIDER don't need re-exporting each
# session. Real environment variables still take precedence over .env values.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# Beat window: samples around the R-peak (MIT-BIH is 360 Hz). ~0.5s before, ~0.7s after.
PRE_SAMPLES = 180
POST_SAMPLES = 252
WINDOW = PRE_SAMPLES + POST_SAMPLES  # 432

HIDDEN_DIM = 768
LLM_DIM = 4096  # projection target (RQ2 design stub)

# AAMI 3-class task: Normal / Supraventricular / Ventricular.
# Fusion (F) and paced/unknown (Q) beats are extremely rare in MIT-BIH and absent
# from parts of the standard split, collapsing macro-F1 to chance; we exclude them
# (their symbols are unmapped here, so the loader drops those beats). This N/S/V
# formulation is the standard reporting task in the ECG arrhythmia literature.
CLASSES = ["N", "S", "V"]
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,        # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,                # Supraventricular
    "V": 2, "E": 2,                                # Ventricular
}

SAMPLE_RATE = 360

# Second modality (wearable-like): number of recent RR intervals per beat.
RR_CONTEXT = 8

# Modalities the framework uses (registered in src/modalities.py). Add a registered
# modality name here and the multimodal model/loader pick it up — no core changes.
# ecg = physiological signal, rr = wearable trend, clinical = patient records.
MODALITIES = ["ecg", "rr", "clinical"]

# Forecasting: from a window of N beats, predict an abnormal (S/V) beat within the
# next M beats.
FORECAST_WINDOW = 10
FORECAST_HORIZON = 5

# Inter-patient split (de Chazal DS1/DS2). No record appears in both splits, and
# both contain S and V beats so macro-F1 is well-defined over all three classes.
TRAIN_RECORDS = ["101", "106", "108", "109", "112", "114", "115", "116", "118",
                 "119", "122", "124", "201", "203", "205", "207", "208", "209",
                 "215", "220", "223", "230"]
TEST_RECORDS = ["100", "103", "105", "111", "113", "117", "121", "123", "200",
                "202", "210", "212", "213", "214", "219", "221", "222", "228",
                "231", "232", "233", "234"]

# LLM reasoning. Provider auto-detected at runtime: OpenAI if OPENAI_API_KEY is
# set, else Anthropic if ANTHROPIC_API_KEY is set, else a deterministic fallback.
# Override the choice with the LLM_PROVIDER env var ("openai" | "anthropic").
CLAUDE_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"
