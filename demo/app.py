"""Streamlit demo: pick/upload an ECG beat + patient info -> AI risk report."""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src import config
from src.data import loader
from src.pipeline import analyze

st.set_page_config(page_title="Time-Series LLM for ECG", layout="centered")
st.title("Time-Series LLM — ECG Clinical Reasoning (Research Prototype)")
st.caption("Not a medical device. For research demonstration only.")


@st.cache_data(show_spinner="Loading sample ECG beats...")
def load_samples():
    X, y = loader.load_split(["100"], max_beats_per_record=50)
    return X, y


with st.sidebar:
    st.header("Patient information")
    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    history = st.text_input("Medical history", value="Hypertension")
    source = st.radio("ECG source", ["Sample from MIT-BIH (record 100)", "Upload CSV (single beat)"])

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
