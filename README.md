# Time-Series LLM for ECG — Research Prototype

A prototype framework that adapts Large Language Models (LLMs) for **medical
time-series analysis and clinical reasoning**. It turns raw ECG beats into
structured temporal embeddings, classifies arrhythmia, and uses an LLM to produce
an **explainable** cardiovascular-risk assessment.

> ⚠️ **Research prototype, not a medical device.** Outputs are for demonstration
> and research only and must not be used for diagnosis or patient care.

This 4-day prototype focuses on a single modality (ECG, MIT-BIH). Wearable and
clinical-text modalities are addressed at the design level (see RQ2) as future work.

---

## Architecture

```
ECG beat (MIT-BIH)
   │
   └─► TemporalTokenizer (1D-CNN)  ──►  embedding (B×N×768)
              │                               │
              │                               └─► t-SNE / PCA  (RQ1 evidence)
              ├─► mean-pool ─► ClassifierHead ─► N/S/V label + confidence
              │                               │
              │                               └─► feature descriptor (class, confidence, HR, rhythm)
              │                                            │  + patient context (age, history)
              ▼                                            ▼
        Projector 768→4096                        Claude API (Haiku 4.5)
        (RQ2 design stub)                                  │
                                                           ▼
                                          risk + step-by-step reasoning  (RQ3)
                                                           │
                                                           ▼
                                          rule-based faithfulness check
```

- **Classification task:** AAMI 3-class **N / S / V** (Normal, Supraventricular,
  Ventricular ectopic beats), the standard reporting task in the ECG literature.
- **Split:** de Chazal **DS1/DS2** inter-patient split (no patient appears in both
  train and test).
- **LLM:** Anthropic Claude API (`claude-haiku-4-5`) with a deterministic
  template **fallback** so the pipeline runs with no API key.

---

## Setup

Requires Python 3.11+ (verified on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To enable real LLM reasoning (otherwise a deterministic template fallback is used),
provide a key either via the shell or a `.env` file at the repo root:

```bash
# Option A — shell env var (per session):
export OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY=sk-ant-...

# Option B — .env file (loaded automatically, gitignored):
cp .env.example .env                  # then edit .env and paste your key
```

The provider is auto-detected (OpenAI if `OPENAI_API_KEY` is set, else Anthropic).
Set `LLM_PROVIDER=openai|anthropic` to force one when both keys are present.

---

## How to run

```bash
# 1. Train the encoder + classifier (downloads MIT-BIH on first run, caches to ./mitdb)
python -m src.train
#    -> artifacts/metrics.json, artifacts/confusion_matrix.png, artifacts/checkpoint.pt

# 2. Generate the t-SNE embedding plot (RQ1 evidence)
python -m src.embed_viz
#    -> artifacts/tsne.png

# 3. Run the test suite
pytest -v          # 13 tests

# 4. Launch the interactive demo
streamlit run demo/app.py
```

---

## Results

Test set: full DS2 split (49,287 beats; N=44,230, S=1,837, V=3,220). Training used
class-balanced undersampling (2,000 / 943 / 2,000 beats for N/S/V).

| Metric | Value |
|---|---|
| Accuracy | 0.693 |
| Macro-F1 | 0.532 |
| Macro-AUROC | **0.907** |

- `artifacts/confusion_matrix.png` — per-class confusion on the test split.
- `artifacts/tsne.png` — t-SNE of the learned embeddings, coloured by class.

Accuracy is *lower* than a naive always-predict-N classifier (which would score
~0.90 by exploiting class imbalance) **by design**: balanced training forces the
model to actually detect the rare S and V beats, which is what makes Macro-F1 and
AUROC meaningful. AUROC 0.91 shows the embeddings are strongly class-discriminative.

---

## Research questions

**RQ1 — Can temporal tokenization help an LLM understand physiological signals?**
Yes, in the representational sense. The 1D-CNN tokenizer produces embeddings on
which a linear head reaches Macro-AUROC 0.91, and the t-SNE plot shows the
ventricular (V) and supraventricular (S) beats forming localized clusters
distinct from the normal (N) bulk. The temporal tokens therefore encode
clinically meaningful structure that a downstream reasoner can consume.

**RQ2 — How can multimodal medical data be aligned into a unified representation?**
Addressed at the design level. `src/models/projector.py` implements a projection
from the 768-d encoder space to a 4096-d LLM embedding space, wired and tested but
not trained end-to-end (closed-weights API LLMs cannot ingest soft embeddings).
The same projection design generalizes to wearable and text encoders — training a
true multimodal alignment is the primary future-work item.

**RQ3 — Can LLM reasoning improve interpretability of time-series prediction?**
Yes. Rather than emitting a bare label, the pipeline converts model outputs into a
text descriptor and has the LLM produce a risk level plus step-by-step reasons
grounded in the findings and patient history. A rule-based **faithfulness check**
verifies the reasoning references the actual predicted findings, guarding against
ungrounded explanations.

---

## Repository layout

```
src/
  config.py                 # classes, window, split, model name
  data/loader.py            # MIT-BIH download/cache, beat segmentation, AAMI map, balancing
  models/temporal_encoder.py# TemporalTokenizer (1D-CNN) + ClassifierHead
  models/projector.py       # 768->4096 projection (RQ2 stub)
  models/llm.py             # Claude client + template fallback
  features.py               # descriptor builder + faithfulness check
  train.py                  # training, metrics, confusion matrix, checkpoint
  embed_viz.py              # embedding extraction + t-SNE
  pipeline.py               # analyze(ecg, patient_info) end-to-end
notebooks/ECG_embedding.ipynb
demo/app.py                 # Streamlit demo
tests/                      # pytest suite
docs/superpowers/           # design spec + implementation plan
```

---

## Limitations & future work

- **Not a medical device.** No clinical validation; trained on a trimmed,
  balanced subset for prototype speed.
- **Single modality.** Only ECG is trained. Wearable signals and clinical notes
  are design-level only (RQ2).
- **Text-bridge, not embedding injection.** Because the LLM is a closed-weights
  API, temporal embeddings reach it as a text descriptor rather than as injected
  tokens. A local open-weights LLM with a trained soft-prompt adapter is the
  natural next step.
- **Heart-rate estimate is a placeholder.** A single ~1.2 s beat window does not
  contain RR intervals, so the demo's HR figure is a crude proxy, not a real
  measurement. It does not affect the classification or AUROC results.
- **Supraventricular (S) beats are the hardest class** (subtle morphology, fewest
  training samples) and drive most of the Macro-F1 gap.
- **Fusion (F) and paced/unknown (Q) beats are excluded** — they are too rare in
  MIT-BIH (and absent from parts of the standard split) to evaluate fairly, which
  is why this prototype reports the standard N/S/V task.
