"""Streamlit demo: live ECG analysis + a research-evidence dashboard.

Tab 1 (Live Analysis): pick/upload an ECG beat + patient info -> AI risk report.
Tab 2 (Research Evidence): show classification, multimodal cross-modal gain,
forecasting metrics, and the embedding / confusion-matrix figures from artifacts/.
"""
import os
import sys

# Streamlit puts the script's own directory on sys.path, not the repo root,
# so make the project importable regardless of where streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src import config
from src.data import loader
from src.pipeline import analyze

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

st.set_page_config(page_title="Time-Series LLM for ECG", layout="centered")
st.title("Time-Series LLM — ECG Clinical Reasoning (Research Prototype)")
st.caption("Not a medical device. For research demonstration only.")


def _load_json(name):
    path = os.path.join(ARTIFACTS, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _artifact(name):
    path = os.path.join(ARTIFACTS, name)
    return path if os.path.exists(path) else None


@st.cache_data(show_spinner="Loading sample ECG beats...")
def load_samples():
    X, y = loader.load_split(["100"], max_beats_per_record=50)
    return X, y


with st.sidebar:
    st.header("Patient information")
    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    history = st.text_input("Medical history", value="Hypertension")
    source = st.radio("ECG source", ["Sample from MIT-BIH (record 100)", "Upload CSV (single beat)"])

tab_live, tab_evidence = st.tabs(["🔬 Live Analysis", "📊 Research Evidence"])

# ----------------------------------------------------------------------------- Live
with tab_live:
    ecg = None
    if source.startswith("Sample"):
        try:
            X, y = load_samples()
            idx = st.slider("Sample beat index", 0, len(X) - 1, 0)
            ecg = X[idx]
            st.write(f"True label (dataset): **{config.CLASSES[y[idx]]}**")
        except Exception as e:
            st.error(f"Could not load samples (network?): {e}")
    else:
        up = st.file_uploader("CSV with one column of ECG samples", type=["csv"])
        if up is not None:
            arr = np.loadtxt(up, delimiter=",").astype("float32").ravel()
            if len(arr) >= config.WINDOW:
                ecg = arr[:config.WINDOW]
            else:
                ecg = np.pad(arr, (0, config.WINDOW - len(arr)))

    if ecg is not None:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(ecg); ax.set_title("ECG beat window"); ax.set_xlabel("sample")
        st.pyplot(fig)

        if st.button("Analyze", type="primary"):
            with st.spinner("Running pipeline..."):
                result = analyze(ecg, {"age": age, "history": history})
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted beat", result["label"])
            c2.metric("Confidence", f"{result['confidence']:.2f}")
            c3.metric("Risk", result["llm_report"].get("risk", "?"))
            st.subheader("AI reasoning")
            for r in result["llm_report"].get("reasons", []):
                st.markdown(f"- {r}")
            st.caption(f"Descriptor: {result['descriptor']}")
            faith = result["faithfulness"]
            (st.success if faith["faithful"] else st.warning)(
                f"Faithfulness check: mentions findings = {faith['faithful']}"
            )

# ------------------------------------------------------------------------- Evidence
with tab_evidence:
    st.subheader("RQ1 — ECG classification (N/S/V)")
    m = _load_json("metrics.json")
    if m:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Macro-F1", f"{m['macro_f1']:.3f}")
        c3.metric("Macro-AUROC", f"{m['macro_auroc']:.3f}")
    else:
        st.info("Run `python -m src.train` to generate metrics.json")

    st.subheader("RQ2 — Multimodal: ECG-only vs ECG + RR-trend")
    mm = _load_json("multimodal_metrics.json")
    if mm:
        eo, er = mm["ecg_only"], mm["ecg_plus_rr"]
        labels = ["Accuracy", "Macro-F1", "Macro-AUROC"]
        keys = ["accuracy", "macro_f1", "macro_auroc"]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w / 2, [eo[k] for k in keys], w, label="ECG only")
        ax.bar(x + w / 2, [er[k] for k in keys], w, label="ECG + RR")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.legend()
        ax.set_title("Cross-modal gain")
        st.pyplot(fig)
        st.success(f"Adding the RR-trend modality lifts Macro-F1 by "
                   f"**+{mm['delta_macro_f1']:.2f}** (cross-modal benefit).")
    else:
        st.info("Run `python -m src.train_multimodal` to generate multimodal_metrics.json")

    st.subheader("Forecasting — imminent abnormal beat")
    fc = _load_json("forecast_metrics.json")
    if fc:
        c1, c2, c3 = st.columns(3)
        c1.metric("AUROC", f"{fc['auroc']:.3f}")
        c2.metric("Avg precision", f"{fc['average_precision']:.3f}")
        c3.metric("Positive rate", f"{fc['test_positive_rate']:.3f}")
        st.caption(f"From {fc['window']} recent RR intervals, predict an abnormal "
                   f"(S/V) beat within the next {fc['horizon']} beats.")
    else:
        st.info("Run `python -m src.forecast` to generate forecast_metrics.json")

    st.subheader("Embedding structure & confusion")
    tsne, cm = _artifact("tsne.png"), _artifact("confusion_matrix.png")
    cols = st.columns(2)
    if tsne:
        cols[0].image(tsne, caption="t-SNE of ECG embeddings (RQ1)")
    else:
        cols[0].info("Run `python -m src.embed_viz`")
    if cm:
        cols[1].image(cm, caption="Confusion matrix (N/S/V)")
    else:
        cols[1].info("Run `python -m src.train`")
