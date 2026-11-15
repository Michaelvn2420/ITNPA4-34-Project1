"""Researchly Research Abstract Classifier - Streamlit app (ITNPA4-34 Project 1, Q2.3).

Run from this folder:  streamlit run app.py
model.joblib holds the full pipeline (cleaning -> TF-IDF -> logistic regression), so the app
passes the raw abstract straight in. text_utils.py must sit next to this file: the pipeline's
cleaning step refers to text_utils.clean_many, and loading fails without it.
"""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.joblib"
FRIENDLY = {"ai": "artificial intelligence", "classical": "classical research"}


@st.cache_resource          # load the model once, not on every button click
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Researchly Research Abstract Classifier")
st.write("Paste the abstract of a research paper below. The model predicts whether the paper "
         "uses artificial intelligence methods or classical research methods.")

abstract = st.text_area("Enter research abstract:", height=200)

if st.button("Classify Abstract"):
    if not abstract.strip():
        st.warning("Please enter an abstract first.")
    else:
        label = model.predict([abstract])[0]
        proba = model.predict_proba([abstract])[0]
        model_score = proba[list(model.classes_).index(label)]
        st.subheader("Prediction")
        st.success(f"The abstract is classified as: {FRIENDLY[label]}")
        st.metric("Model score", f"{model_score:.0%}")
        st.progress(float(model_score))
        st.caption("Score = the model's predicted probability for this class. It is not calibrated, "
           "so treat it as a ranking, not as a true chance of being correct.")
