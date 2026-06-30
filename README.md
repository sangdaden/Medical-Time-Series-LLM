# Time-Series LLM for ECG — Research Prototype

A prototype framework that adapts Large Language Models (LLMs) for **medical
time-series analysis and clinical reasoning**. It turns raw ECG beats into
structured temporal embeddings, classifies arrhythmia, and uses an LLM to produce
an **explainable** cardiovascular-risk assessment.

> ⚠️ **Research prototype, not a medical device.** Outputs are for demonstration
> and research only and must not be used for diagnosis or patient care.

This prototype works on MIT-BIH ECG and a second wearable-like modality (the
RR-interval / heart-rate trend derived from it), demonstrating multimodal alignment,
cross-modal gain, clinical reasoning, and short-horizon risk forecasting. An
independent clinical-text modality (e.g. MIMIC-IV) is scoped as future work.

---

## Architecture

```
MIT-BIH ECG
  │
  ├─ beat waveform (432) ─► TemporalTokenizer (1D-CNN) ─► emb 768 ─► proj ─┐
  │                                         └─► t-SNE (RQ1)                 │ shared
  ├─ RR / HR-trend (8)   ─► RREncoder ──────────────────► emb 64  ─► proj ─┤ 128-d ─► fuse ─► N/S/V + confidence   (RQ1, RQ2)
  │                                                                        ┘                        │
  │                                                                                                  ▼
  │                                                            feature descriptor  (+ age, history)
  │                                                                                                  │
  │                                                                                                  ▼
  │                                            LLM (OpenAI / Anthropic) ─► risk + step-by-step reasoning   (RQ3)
  │                                                                                                  │
  │                                                                                                  ▼
  │                                                                              rule-based faithfulness check
  │
  └─ RR stream ─► RiskForecaster (GRU) ─► abnormal (S/V) beat within next M beats?   (forecasting)
```

- **Classification task:** AAMI 3-class **N / S / V** (Normal, Supraventricular,
  Ventricular ectopic beats), the standard reporting task in the ECG literature.
- **Modalities:** ECG beat waveform + RR-interval (heart-rate) trend derived from it;
  each is projected into a shared 128-d space and fused (multimodal alignment).
- **Split:** de Chazal **DS1/DS2** inter-patient split (no patient appears in both
  train and test).
- **LLM:** OpenAI or Anthropic, auto-detected from the API key (default
  `gpt-4o-mini` / `claude-haiku-4-5`), with a deterministic template **fallback** so
  the pipeline runs with no key.

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

# 3. Multimodal experiment: ECG-only vs ECG+RR (cross-modal benefit)
python -m src.train_multimodal
#    -> artifacts/multimodal_metrics.json

# 4. Risk forecasting (predict an imminent abnormal beat)
python -m src.forecast
#    -> artifacts/forecast_metrics.json

# 5. Run the test suite
pytest -v          # 22 tests

# 6. Launch the interactive demo (two tabs: Live Analysis + Research Evidence)
streamlit run demo/app.py
```

The demo's **Research Evidence** tab reads the JSON/figures in `artifacts/`, so run
steps 1–4 first to populate it.

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

### Multimodal: cross-modal benefit (RQ2)

A second, wearable-like modality — the **RR-interval (heart-rate) trend** derived from
the ECG — is encoded and **aligned with the ECG embedding in a shared space**, then
fused (`src/train_multimodal.py`). Same backbone and data; only the RR modality differs:

| Configuration | Accuracy | Macro-F1 | Macro-AUROC |
|---|---|---|---|
| ECG only | 0.642 | 0.446 | 0.866 |
| **ECG + RR** | **0.910** | **0.737** | **0.952** |

Adding the RR modality lifts Macro-F1 by **+0.29**. The RR trend carries beat-timing
information (premature beats shorten RR) that a single waveform window lacks, so the
two modalities are genuinely complementary — direct evidence of cross-modal learning.

### Forecasting: imminent abnormal beat (`src/forecast.py`)

From a window of the **10 most recent RR intervals**, a GRU forecasts whether an
abnormal (S/V) beat occurs within the **next 5 beats** (inter-patient split):

| Metric | Value |
|---|---|
| AUROC | 0.847 |
| Average precision | 0.687 |
| Test positive rate | 0.281 |

The recent heart-rate rhythm is predictive of upcoming arrhythmic events — a first
step toward patient risk forecasting over time.

---

## Research questions

**RQ1 — Can temporal tokenization help an LLM understand physiological signals?**
Yes, in the representational sense. The 1D-CNN tokenizer produces embeddings on
which a linear head reaches Macro-AUROC 0.91, and the t-SNE plot shows the
ventricular (V) and supraventricular (S) beats forming localized clusters
distinct from the normal (N) bulk. The temporal tokens therefore encode
clinically meaningful structure that a downstream reasoner can consume.

**RQ2 — How can multimodal medical data be aligned into a unified representation?**
Demonstrated on two real modalities. The ECG waveform and the RR-interval (heart-rate)
trend are each encoded, **projected into a shared 128-d space, and fused**
(`src/models/multimodal.py`, `src/train_multimodal.py`). The aligned multimodal model
beats the ECG-only baseline by +0.29 Macro-F1, showing the shared representation
captures complementary cross-modal information. The same projection design extends to
further modalities (clinical text/notes) — that extension, and embedding injection into
an open-weights LLM, remain future work.

**RQ3 — Can LLM reasoning improve interpretability of time-series prediction?**
Yes. Rather than emitting a bare label, the pipeline converts model outputs into a
text descriptor and has the LLM produce a risk level plus step-by-step reasons
grounded in the findings and patient history. A rule-based **faithfulness check**
verifies the reasoning references the actual predicted findings, guarding against
ungrounded explanations.

---

## Coverage vs the original project description

| Claim in the project description | Status | Where |
|---|---|---|
| Adapt LLMs for medical time-series | ✅ Done | `pipeline.py` |
| Effective **temporal tokenization** | ✅ Done | `TemporalTokenizer`, AUROC 0.91 + t-SNE |
| **Multimodal** data | ✅ Two modalities (ECG + RR trend) | `train_multimodal.py` |
| **Multimodal alignment** | ✅ Shared-space projection + fusion | `models/multimodal.py` |
| Learn **cross-modal patterns** | ✅ +0.29 Macro-F1 from fusion | `multimodal_metrics.json` |
| **Interpretable predictions** | ✅ Done | LLM reasoning + faithfulness check |
| **Clinical reasoning** module | ✅ Done | `models/llm.py` |
| **Risk forecasting** | ✅ Short-horizon (AUROC 0.85) | `forecast.py` |
| Early disease detection | ⚠️ Proxy (arrhythmia detection) | classification + forecasting |
| **Clinical records / notes** modality | ❌ Future work | needs MIMIC-IV (credentialed) |
| Embedding injection into LLM | ❌ Future work | uses text-bridge instead |
| General-purpose smart-health agent | ⚠️ Foundation only | prototype groundwork |

This is an honest **proof-of-concept**: the core ideas (tokenization, multimodal
alignment, cross-modal gain, clinical reasoning, forecasting) are demonstrated on real
ECG-derived data; the remaining items (independent text modality, embedding injection,
full agent) are scoped as future work.

---

## Repository layout

```
src/
  config.py                 # classes, window, split, model name
  data/loader.py            # MIT-BIH download/cache, beat segmentation, AAMI map, balancing
  models/temporal_encoder.py# TemporalTokenizer (1D-CNN) + ClassifierHead
  models/rr_encoder.py      # RREncoder for the RR-interval (heart-rate) modality
  models/multimodal.py      # MultimodalClassifier: align + fuse ECG and RR (RQ2)
  models/projector.py       # 768->4096 projection (LLM-space design stub)
  models/llm.py             # OpenAI/Anthropic client + template fallback
  features.py               # descriptor builder + faithfulness check
  train.py                  # ECG classification: metrics, confusion matrix, checkpoint
  train_multimodal.py       # ECG-only vs ECG+RR cross-modal comparison
  forecast.py               # RR-trend GRU forecaster for imminent abnormal beats
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
- **Two modalities, both derived from ECG.** The multimodal experiment fuses the ECG
  waveform with the RR-interval (heart-rate) trend. A truly independent modality —
  clinical notes / structured records (e.g. MIMIC-IV) — is not yet included; that
  needs credentialed data access and is the next modality to add.
- **Text-bridge, not embedding injection.** Because the reasoning LLM is a
  closed-weights API, temporal embeddings reach it as a text descriptor rather than
  as injected tokens. A local open-weights LLM with a trained soft-prompt adapter is
  the natural next step.
- **Demo heart-rate figure is a placeholder.** The single-beat demo cannot derive a
  real heart rate (no RR interval within one beat); the training/forecasting code uses
  proper RR intervals from the beat stream. The demo HR does not affect any metric.
- **Forecasting is intra-record, short-horizon.** It predicts an abnormal beat within
  the next few beats from the RR stream, not long-term disease progression.
- **Supraventricular (S) beats are the hardest class** (subtle morphology, fewest
  training samples) and drive most of the Macro-F1 gap.
- **Fusion (F) and paced/unknown (Q) beats are excluded** — they are too rare in
  MIT-BIH (and absent from parts of the standard split) to evaluate fairly, which
  is why this prototype reports the standard N/S/V task.
