## What this is
Streamlit app for Researchly: paste an abstract, and it says "artificial intelligence" or "classical research" plus a model score.

- **Model:** logistic regression on TF-IDF (1–2 word phrases), with the text cleaning built into the saved pipeline, so it takes raw text.

## Files
- `app.py` - the interface
- `model.joblib` - the trained pipeline (2.5 MB)
- `text_utils.py` - the cleaning functions
- `requirements.txt` - versions

`text_utils.py` must stay next to `model.joblib`.

joblib stores only a reference to its `clean_many` function, and loading fails with `ModuleNotFoundError: No module named 'text_utils'` without it.

## Requirements
- Python 3.11 (tested on 3.11.9)

Packages:
- `streamlit 1.67.0`
- `scikit-learn 1.9.1`
- `joblib 1.6.0`
- `nltk 3.10.4`
- `numpy 2.4.6`

`scikit-learn` is pinned to the version that saved the model, because other versions may not load it.

## Setup
```bat
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Run
From inside the `deployment` folder:

```bat
streamlit run app.py
```

Then open http://localhost:8501 if the browser doesn't open.

Press `Ctrl+C` in the terminal to stop it.

## First run
Needs internet once: it downloads the NLTK data (stopwords, WordNet, tokenizer). Not needed afterwards.

## Troubleshooting
**Error: `Resource stopwords not found` when the app starts.** The NLTK download was blocked (no internet, or the connection goes through a proxy). Either:
- set `NLTK_ALLOW_PROXIED_URLOPEN=1` and run `streamlit run app.py` again (in Windows cmd: `set NLTK_ALLOW_PROXIED_URLOPEN=1`), or
- download the data yourself:
  ```bat
  python -c "import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"
  ```

## How to read the result
The model score is the model's predicted probability for the class shown.

It is not calibrated: C=300 pushes scores to the extremes, so treat it as a ranking.

## Known limitations
- Very short inputs give the model little to work on (e.g. "the intersection between ai and cosmology" → classical, 87%).
- "AI" and "ML" are kept by the cleaning but are rare in the training data (4 and 6 abstracts), so they carry little weight.
- Trained on inflation-related abstracts (mostly economics); not tested on other fields.
- Final model test macro-F1 is 0.891.

## Test example
Paste this abstract into the app:

> Artificial intelligence (AI) has become a transformative technology in healthcare by enabling more accurate diagnosis, predictive analytics, and personalised treatment planning. Machine learning algorithms can analyse large volumes of medical data to identify patterns that assist clinicians in detecting diseases at earlier stages. In addition, AI-powered decision support systems improve clinical workflows by reducing manual effort and enhancing diagnostic consistency. Despite these benefits, challenges such as data privacy, model interpretability, and ethical considerations remain significant barriers to widespread adoption. Continued research is required to develop robust, transparent, and trustworthy AI systems that can be safely integrated into modern healthcare environments.

It should give a result of **98% artificial intelligence**.