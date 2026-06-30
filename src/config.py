"""Project-wide constants."""

# Beat window: samples around the R-peak (MIT-BIH is 360 Hz). ~0.5s before, ~0.7s after.
PRE_SAMPLES = 180
POST_SAMPLES = 252
WINDOW = PRE_SAMPLES + POST_SAMPLES  # 432

HIDDEN_DIM = 768
LLM_DIM = 4096  # projection target (RQ2 design stub)

# AAMI 5-class grouping. Maps MIT-BIH annotation symbols -> class index.
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,        # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,                # Supraventricular
    "V": 2, "E": 2,                                # Ventricular
    "F": 3,                                         # Fusion
    "/": 4, "f": 4, "Q": 4,                        # Unknown/paced
}

SAMPLE_RATE = 360

# Inter-patient split (records). Standard de Chazal-style split, trimmed for speed.
TRAIN_RECORDS = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119"]
TEST_RECORDS = ["100", "103", "105", "111", "113", "117", "121", "123"]

CLAUDE_MODEL = "claude-haiku-4-5"
