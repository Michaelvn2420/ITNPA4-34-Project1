"""
Generic summarisation and topic-modelling helpers for ITNPA4-34 Project 1.
  freq_summary     - frequency-based extractive summariser (Luhn 1958 idea; NLTK book ex. 4.32)
  textrank_summary - graph-based extractive summariser (Mihalcea & Tarau 2004, TextRank):
                     TF-IDF sentence vectors -> cosine-similarity graph -> PageRank
  chunk_words      - split long documents into fixed-size pseudo-documents for LDA
  lda_sweep        - fit gensim LDA for a range of k and report c_v / u_mass coherence
NB: gensim.summarization no longer exists (removed in gensim 4.0), hence these functions.
"""
import re

import networkx as nx
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP = set(stopwords.words("english"))


def _sentences(text, min_words=8, max_words=60):
    """Sentence-split, drop very short/long fragments and exact duplicates (keep order)."""
    seen, out = set(), []
    for s in sent_tokenize(text):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if min_words <= len(s.split()) <= max_words and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


def freq_summary(text, n=5):
    """Return (summary_sentences, sentence_scores Series, word_freq Series)."""
    sents = _sentences(text)
    words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in STOP]
    freq = pd.Series(words).value_counts()
    wfreq = freq / freq.max()                                   # normalised word weights
    scores = pd.Series({i: sum(wfreq.get(w, 0) for w in word_tokenize(s.lower()))
                        / len(s.split()) for i, s in enumerate(sents)})
    top = sorted(scores.nlargest(n).index)                      # keep original order
    return [sents[i] for i in top], scores, freq


def textrank_summary(text, n=6):
    """Return (summary_sentences, pagerank Series, similarity matrix, sentences)."""
    sents = _sentences(text)
    X = TfidfVectorizer(stop_words="english").fit_transform(sents)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    pr = pd.Series(nx.pagerank(nx.from_numpy_array(S)))
    top = sorted(pr.nlargest(n).index)
    return [sents[i] for i in top], pr, S, sents


def chunk_words(text, size=150):
    """Split text into consecutive chunks of `size` words (pseudo-documents)."""
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def lda_sweep(corpus, dictionary, texts, k_values=range(2, 11), passes=10, seed=42):
    """Fit LDA for each k; return a DataFrame of coherence scores and the fitted models."""
    from gensim.models import LdaModel, CoherenceModel
    rows, models = [], {}
    for k in k_values:
        m = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=passes,
                     random_state=seed, alpha="auto", eta="auto")
        cv = CoherenceModel(model=m, texts=texts, dictionary=dictionary,
                            coherence="c_v", processes=1).get_coherence()
        um = CoherenceModel(model=m, corpus=corpus, dictionary=dictionary,
                            coherence="u_mass", processes=1).get_coherence()
        rows.append({"k": k, "c_v": cv, "u_mass": um})
        models[k] = m
    return pd.DataFrame(rows), models
