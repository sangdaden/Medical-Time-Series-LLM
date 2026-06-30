# Time-Series LLM for Multimodal Medical Signals — Research Prototype

A prototype framework that adapts Large Language Models (LLMs) for **multimodal
medical time-series analysis and clinical reasoning**. It turns medical signals into
structured temporal embeddings, **aligns and fuses multiple modalities**, classifies
arrhythmia, forecasts imminent risk, and uses an LLM to produce an **explainable**
cardiovascular-risk assessment.

It is built as an **extensible modality framework**: a modality (a per-beat feature
extractor + an encoder) is *registered* and automatically plugs into the data loader
and the multimodal model — adding one needs no change to core code. Three real
modalities, matching the project description, ship registered:

| Description's category | Modality here | Source |
|---|---|---|
| physiological signal | `ecg` — beat waveform | MIT-BIH signal |
| wearable sensor data | `rr` — RR-interval / heart-rate trend | derived from ECG |
| clinical records | `clinical` — age, sex, medication count | MIT-BIH header comments |

> ⚠️ **Research prototype, not a medical device.** Outputs are for demonstration
> and research only and must not be used for diagnosis or patient care.

This prototype works on MIT-BIH ECG and a second wearable-like modality (the
RR-interval / heart-rate trend derived from it), demonstrating multimodal alignment,
cross-modal gain, clinical reasoning, and short-horizon risk forecasting. An
independent clinical-text modality (e.g. MIMIC-IV) is scoped as future work.

---

## Architecture

```
MIT-BIH record
  │  (registered modalities — see src/modalities.py)
  ├─ ecg: beat waveform (432) ─► EcgEncoder (1D-CNN) ─► 768 ─► proj ─┐
  │                                       └─► t-SNE (RQ1)            │ shared
  ├─ rr: RR / HR-trend (8)     ─► VectorEncoder ─────────► 64  ─► proj ─┤ 128-d ─► fuse ─► N/S/V   (RQ1, RQ2)
  ├─ clinical: age/sex/meds (3)─► VectorEncoder ─────────► 32  ─► proj ─┘                  │
  │                                                                                         ▼
  │                                                       feature descriptor (+ patient context)
  │                                                                                         │
  │                                       LLM (OpenAI / Anthropic) ─► risk + step-by-step reasoning   (RQ3)
  │                                                                                         │
  │                                                                                         ▼
  │                                                                     rule-based faithfulness check
  │
  └─ RR stream ─► RiskForecaster (GRU) ─► abnormal (S/V) beat within next M beats?   (forecasting)
```

- **Classification task:** AAMI 3-class **N / S / V** (Normal, Supraventricular,
  Ventricular ectopic beats), the standard reporting task in the ECG literature.
- **Multimodal alignment:** each registered modality is encoded and projected into a
  shared 128-d space, then fused; the model is generic over any 1..N modalities.
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
pytest -v          # 27 tests

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

### Multimodal: per-modality contribution (RQ2)

The framework trains each cumulative modality combination on the same data
(`src/train_multimodal.py`), so we can *measure* what each modality contributes:

| Modalities | Macro-F1 | Macro-AUROC |
|---|---|---|
| `ecg` | 0.50 | 0.865 |
| **`ecg + rr`** | **0.67** | **0.946** |
| `ecg + rr + clinical` | 0.60 | 0.942 |

The **RR trend gives a large cross-modal gain (+0.17 Macro-F1)** — it carries
beat-timing information (premature beats shorten RR) that a single waveform window
lacks. The **clinical-records modality (age/sex/medications) does *not* improve
beat-level classification** — an honest negative result: those features are constant
within a patient and so cannot discriminate beats, though they remain valuable as
context for the LLM reasoning layer. Being able to quantify this per modality is
exactly what the framework is for. (Numbers vary slightly run-to-run.)

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
Demonstrated by an **extensible modality framework** (`src/framework.py`,
`src/modalities.py`). Three real modalities — physiological (ECG), wearable (RR trend),
and clinical records (header demographics/medications) — are each encoded and
**projected into a shared 128-d space, then fused** (`src/models/multimodal.py`). The
aligned ECG+RR model beats ECG-only by +0.17 Macro-F1, showing the shared
representation captures complementary cross-modal information. Adding a new modality
requires only registering it (no core changes). Embedding injection into an
open-weights LLM, and a free-text clinical-notes modality (e.g. MIMIC-IV), remain
future work.

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
| A **new framework** adapting LLMs | ✅ Extensible modality registry | `framework.py`, `modalities.py` |
| Adapt LLMs for medical time-series | ✅ Done | `pipeline.py` |
| Effective **temporal tokenization** | ✅ Done | `TemporalTokenizer`, AUROC 0.91 + t-SNE |
| **Multimodal** data (3 categories) | ✅ physiological + wearable + clinical | `modalities.py` |
| **Multimodal alignment** | ✅ Shared-space projection + fusion (N modalities) | `models/multimodal.py` |
| Learn **cross-modal patterns** | ✅ +0.17 Macro-F1 from ECG+RR fusion | `multimodal_metrics.json` |
| Physiological signals | ✅ ECG waveform | `modalities.py` (`ecg`) |
| Wearable sensor data | ✅ RR / heart-rate trend | `modalities.py` (`rr`) |
| Clinical records | ✅ demographics + medications (MIT-BIH headers) | `modalities.py` (`clinical`) |
| **Interpretable predictions** | ✅ Done | LLM reasoning + faithfulness check |
| **Clinical reasoning** module | ✅ Done | `models/llm.py` |
| **Risk forecasting** | ✅ Short-horizon (AUROC 0.85) | `forecast.py` |
| Early disease detection | ⚠️ Proxy (arrhythmia detection) | classification + forecasting |
| Free-text clinical **notes** modality | ❌ Future work | needs MIMIC-IV (credentialed) |
| Embedding injection into LLM | ❌ Future work | uses text-bridge instead |
| General-purpose smart-health agent | ⚠️ Foundation only | prototype groundwork |

This is an honest **proof-of-concept framework**: the core ideas (an extensible
modality framework, temporal tokenization, multimodal alignment across all three
description categories, cross-modal gain, clinical reasoning, forecasting) are
demonstrated on real data; the remaining items (free-text notes, embedding injection,
full agent) are scoped as future work.

### Extending the framework

To add a modality, register it in `src/modalities.py` and list its name in
`config.MODALITIES` — no other code changes:

```python
register(ModalitySpec(
    name="spo2", feat_dim=K,
    prepare=_spo2_prepare,          # (signal, rpeaks, symbols, header) -> ctx
    beat_feature=_spo2_beat,        # (ctx, j, r) -> np.ndarray[K]
    make_encoder=lambda: VectorEncoder(K, 32),
))
```

---

## Repository layout

```
src/
  config.py                 # classes, window, split, MODALITIES, model names
  framework.py              # ModalitySpec + registry (the extensible core)
  modalities.py             # ecg / rr / clinical modalities, registered
  data/loader.py            # MIT-BIH cache, segmentation, AAMI map, balancing, load_multimodal
  models/temporal_encoder.py# TemporalTokenizer (1D-CNN) + ClassifierHead
  models/modality_encoders.py# EcgEncoder + VectorEncoder (per-modality encoders)
  models/multimodal.py      # MultimodalModel: align + fuse any 1..N modalities (RQ2)
  models/projector.py       # 768->4096 projection (LLM-space design stub)
  models/llm.py             # OpenAI/Anthropic client + template fallback
  features.py               # descriptor builder + faithfulness check
  train.py                  # ECG classification: metrics, confusion matrix, checkpoint
  train_multimodal.py       # per-modality-combination comparison (ecg / +rr / +clinical)
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
