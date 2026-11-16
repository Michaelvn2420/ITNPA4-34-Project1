"""
Generic text-denoising helpers for ITNPA4-34 Project 1.
IMPORTANT: the deployed model imports this file. Keep ONE copy in src/ and copy it
into deployment/ next to model.joblib, otherwise joblib.load() fails with
ModuleNotFoundError: No module named 'text_utils'.

Techniques are standard (NLTK book ch. 3; scikit-learn docs): HTML/entity removal,
Unicode normalisation, URL/e-mail removal, non-letter removal, lower-casing,
tokenisation, stop-word removal, lemmatisation.
"""
import html
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_data():
    """Download the NLTK resources this project needs (quietly, only if missing)."""
    for res, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.data.find(path + ".zip")       # some resources are stored zipped
            except LookupError:
                nltk.download(res, quiet=True)


ensure_nltk_data()
STOP = set(stopwords.words("english"))
_LEM = WordNetLemmatizer()
KEEP_SHORT = {"ai", "ml"}       # domain abbreviations kept even though they are shorter than min_len


# Structured-abstract headings and publisher boilerplate (describe the format, not the topic)
_HEADINGS = (r"Purpose|Design/methodology/approach|Methodology|Methods?|Findings|Results|"
             r"Originality/value|Research limitations/implications|Practical implications|"
             r"Conclusions?|Background|Objectives?|Abstract|Problem definition")


def strip_boilerplate(t):
    """Remove LaTeX, keyword/JEL lists and structured-abstract headings from one string."""
    t = re.sub(r"\$[^$]*\$", " ", t)                                  # inline LaTeX: $...$
    t = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", t)                  # LaTeX commands: \xi, \mathrm{S}
    t = re.sub(r"\(\s*JEL[^)]*\)", " ", t)                            # "(JEL C51, E31)"
    t = re.sub(r"(Key ?words?|JEL(\s+Classification)?(\s+(Codes?|Numbers?))?)\s*[:\-\u2013].*$",
               " ", t, flags=re.IGNORECASE | re.DOTALL)               # keyword / JEL lists at the end
    t = re.sub(rf"\b({_HEADINGS})\s*[:\-\u2013]\s*", " ", t)          # "Purpose:", "Findings -"
    t = re.sub(rf"^({_HEADINGS})\.?(?=[A-Z])", " ", t)                 # "BackgroundGame" glued at start
    t = re.sub(r"\b(Abstract|Purpose)\.\s+(?=[A-Z])", " ", t)          # "Abstract. The ...", "Purpose. The ..."
    return t


def clean_text(text, extra_stop=(), min_len=3):
    """Denoise one string and return the cleaned, lemmatised text as a single string."""
    t = html.unescape(str(text))                       # &amp; -> &
    t = unicodedata.normalize("NFKC", t)               # odd hyphens/quotes -> standard forms
    t = re.sub(r"[\u2010-\u2015\u2212]", "-", t)      # unicode dashes -> plain hyphen
    t = re.sub(r"<[^>]+>", " ", t)                     # leftover HTML/JATS tags
    t = strip_boilerplate(t)                           # structured headings, keywords, LaTeX
    t = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", t)    # URLs and e-mails
    t = re.sub(r"[^a-zA-Z\s]", " ", t).lower()         # replace (not delete) non-letters
    stop = STOP | set(extra_stop)
    words = [_LEM.lemmatize(w) for w in word_tokenize(t)]
#    return " ".join(w for w in words if w not in stop and len(w) >= min_len)
    return " ".join(w for w in words if w not in stop and (len(w) >= min_len or w in KEEP_SHORT))


def clean_many(texts):
    """List version for sklearn's FunctionTransformer (must be a module-level function)."""
    return [clean_text(t) for t in texts]
