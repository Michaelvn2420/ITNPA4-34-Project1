# %% [markdown]
# # ITNPA4-34 - PROJECT 1 - EDUVOS ONLINE PART TIME - EDUV8426028
# **Michael van Niekerk** - Questions 1 and 2: text mining and classification

# %% Setup: imports, paths, helpers, etc.

import sys,time,warnings,re,sklearn,nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import make_scorer, recall_score, f1_score, balanced_accuracy_score, accuracy_score, classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay


# Repo root = the folder above notebooks/
try:
    BASE = Path(__file__).resolve().parents[1]
except NameError:
    BASE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
print("Project folder:", BASE)

# Make src/ importable (also downloads missing NLTK data on first import)
sys.path.insert(0, str(BASE / "src"))
from text_utils import clean_text, clean_many

# Output folders
FIG = BASE / "outputs" / "figures"
TAB = BASE / "outputs" / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Show plots only when running interactively (Spyder/Jupyter); always save them
try:
    get_ipython()
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False
    
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150)
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)
        
SEED = 26
np.random.seed(SEED)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 80)


print("Python", sys.version.split()[0], "| pandas", pd.__version__,"| scikit-learn", sklearn.__version__,"| nltk", nltk.__version__) 

# %% Load the data for QUESTION 1

df_raw = pd.read_json(BASE / "data" / "research_abstracts.json")
# Make a copy so we dont destroy the original
df = df_raw.copy()

print("Shape (rows, columns):", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())

print("\nClass counts:")
print(df["Label"].value_counts())

print("\nClass proportions:")
print(df["Label"].value_counts(normalize=True).round(3))

print(df.head().to_string(max_colwidth=60))

# %% Find Duplicates and Digital Object Identifiers (DOIs) for QUESTION 1

dup_abs = df["Abstract"].duplicated().sum()
print("Duplicated Abstracts:", dup_abs)

conflicts = (df[df["Abstract"].duplicated(keep=False)].groupby("Abstract")["Label"].nunique().gt(1).sum())
print("Duplicates with conflicting labels:", conflicts)

print("Duplicate Digital Object Identifiers, DOIs, (non-missing):", df["DOI"].dropna().duplicated().sum())
print("Missing DOIs:", df["DOI"].isna().sum())

print("\nThe duplicated rows:")
print(df[df["Abstract"].duplicated(keep=False)].sort_values("Abstract")[["DOI", "Label"]].head(10))

# %% Remove the duplicates for QUESTION 1

removed = df[df["Abstract"].duplicated(keep="first")]  # the rows being removed because they are duplicates
df_dedup = df.drop_duplicates(subset="Abstract", keep="first").reset_index(drop=True)

print(f"Rows: {len(df)} -> {len(df_dedup)} (removed {len(removed)})")
print("Duplicates remaining:", df_dedup["Abstract"].duplicated().sum())  # should be 0 now that all duplicates have been removed

print("\nRemoved rows (DOI, Label):")
print(removed[["DOI", "Label"]])

print("\nClass counts after removal:")
print(pd.concat([df["Label"].value_counts().rename("before"), df_dedup["Label"].value_counts().rename("after")], axis=1))

df = df_dedup  # now df is deduplicated, df_raw still has original

# %% Find the abstract lengths after duplicates have been removed

df["n_chars"] = df["Abstract"].str.len()
df["n_words_raw"] = df["Abstract"].str.split().str.len()

print(df[["n_chars", "n_words_raw"]].describe().round(1))

print("\nMedian words per abstract by class:")
print(df.groupby("Label")["n_words_raw"].median())

print("\nShortest 5:")
print(df.nsmallest(5, "n_chars")[["n_chars", "Label", "DOI"]])

print("\nLongest 5:")
print(df.nlargest(5, "n_chars")[["n_chars", "Label", "DOI"]])

# Read the extremes in full - this will help decide wether they are real abstracts
print("\nSHORTEST abstract:\n", df.loc[df["n_chars"].idxmin(), "Abstract"])
print("\nLONGEST abstract (first 1500 chars):\n", df.loc[df["n_chars"].idxmax(), "Abstract"][:1500])

# %% Charts: class balance and abstract length for QUESTION 1

fig, ax = plt.subplots(figsize=(5,4))
df["Label"].value_counts().plot(kind="bar", ax=ax, color=["tab:blue", "tab:orange"])
ax.set_title("Class balance after deduplication")
ax.set_xlabel("Class")
ax.set_ylabel("Number of abstracts")
fig.suptitle("")  # remove auto title
savefig(fig, "Q1_01_class_balance")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
upper = df["n_words_raw"].quantile(0.99)
for label, grp in df.groupby("Label"):
#    axes[0].hist(grp["n_words_raw"], bins=40, range=(0, upper), alpha=0.6, density=True, label=label)
    axes[0].hist(grp["n_words_raw"], bins=40, range=(0, upper), alpha=0.6, label=label, weights=np.ones(len(grp)) / len(grp) * 100)   # each class sums to 100%
axes[0].set_title("Abstract length distribution by class (up to 99th percentile)")
axes[0].set_xlabel("Words per abstract")
# axes[0].set_ylabel("Density")
axes[0].set_ylabel("% of abstracts in that class")
axes[0].legend()

df.boxplot(column="n_words_raw", by="Label", ax=axes[1])
axes[1].set_title("Abstract length by class")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Words per abstract")
fig.suptitle("")  # remove auto title
savefig(fig, "Q1_02_abstract_length_by_class")

# %% [markdown]
# **NOTES:** AI abstracts are shifted to the right: they peak at around 200 words, while classical abstracts peak at around 150 words. The two distributions overlap drastically which means length alone will not be able to seperate the classes, the words used must do this. The classical class has a much longer tail, with the extreme outliers shown in the box plot.

# %% Explore what themes are in the dataset abstracts? (unsupervised, labels not used)

tfidf_explore = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5)
X_explore = tfidf_explore.fit_transform(df["Abstract"])
terms = tfidf_explore.get_feature_names_out()

K = 8
nmf = NMF(n_components=K, random_state=SEED, max_iter=500)
W = nmf.fit_transform(X_explore)  # document-theme weights
df["theme"] = W.argmax(axis=1)  # each abstract's dominant theme

for k, comp in enumerate(nmf.components_):
    top = [terms[i] for i in comp.argsort()[::-1][:10]]
    print(f"Theme {k:2d} ({(df['theme'] == k).sum():3d} abstracts): {', '.join(top)}")

# %% Explore how the Themes relate to the Labels

theme_ct = pd.crosstab(df["theme"], df["Label"])
print(theme_ct)

# Name each theme from its top 3 words (generated, so it stays correct if K changes)
theme_names = {k: f"{k}: " + ", ".join(terms[i] for i in comp.argsort()[::-1][:3])
               for k, comp in enumerate(nmf.components_)}

theme_ct = pd.crosstab(df["theme"], df["Label"])
theme_ct.index = [theme_names[k] for k in theme_ct.index]
theme_ct["total"] = theme_ct.sum(axis=1)
print(theme_ct)

theme_share = (pd.crosstab(df["theme"], df["Label"], normalize="index")["ai"] * 100).sort_values()
theme_size = df["theme"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh([theme_names[k] for k in theme_share.index], theme_share.values)
overall = (df["Label"] == "ai").mean() * 100
ax.axvline(overall, linestyle="--", color="grey")
ax.text(overall + 0.5, -0.6, f"dataset average {overall:.0f}%", color="grey")
ax.bar_label(bars, labels=[f"{v:.0f}%  (n={theme_size[k]})" for k, v in theme_share.items()], padding=3)
ax.set_title("Share of AI-labelled abstracts in each theme")
ax.set_xlabel("% of abstracts in the theme labelled 'ai'")
ax.set_xlim(0, max(theme_share.max() * 1.25, overall * 1.5))
savefig(fig, "Q1_03_ai_share_by_theme")

# Read a few abstracts from each theme - this is where you form your judgement
for k in range(K):
    print(f"\n===== Theme {k} =====")
    for a in df.loc[df["theme"] == k, "Abstract"].head(3):
        print(" -", a[:200])

# %% Explore abstracts that fit no theme well (outlier scan)

df["best_theme_weight"] = W.max(axis=1)

# Printed list for reference
weak = df.nsmallest(15, "best_theme_weight")
for i, row in weak.iterrows():
    print(f"\n--- row {i} | {row['Label']} | weight {row['best_theme_weight']:.3f} | {row['DOI']}")
    print(row["Abstract"][:250])

# Chart for reference
weak_plot = df.nsmallest(15, "best_theme_weight").iloc[::-1]     # worst fit at the top
# labels = [f"row {i}: " + " ".join(a.split()[:6]) + "..." for i, a in weak_plot["Abstract"].items()]
# Label each bar with that abstract's 4 most distinctive words (highest TF-IDF within the abstract)
label_vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
L = label_vec.fit_transform(df["Abstract"])
vocab = label_vec.get_feature_names_out()

def key_words(i, n=4):
    row = L[i].toarray().ravel()
    return ", ".join(vocab[j] for j in row.argsort()[::-1][:n])

labels = [f"row {i}: {key_words(i)}" for i in weak_plot.index]

colours = ["tab:blue" if l == "ai" else "tab:orange" for l in weak_plot["Label"]]

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(labels, weak_plot["best_theme_weight"], color=colours)
median_fit = df["best_theme_weight"].median()
ax.axvline(median_fit, linestyle="--", color="grey")
ax.text(median_fit, -1.2, " typical abstract (median)", color="grey")
#ax.set_title("The 15 abstracts that fit no theme well (blue = ai, orange = classical)")
ax.set_title("The 15 abstracts that fit no theme well, labelled by their most distinctive words\n(blue = ai, orange = classical)")
ax.set_xlabel("How strongly the abstract belongs to its best theme (higher = clearer fit)")
savefig(fig, "Q1_04_weakest_fitting_abstracts")

# %% [markdown]
# **NOTES: Rules for excluding an abstract from the set being prepared for training. R1, when its more than one paper joined together in which case a single label wont be able to describe it. R2, label contradicted by content. each rule needs evidence, and we only exclude when two independent pieces of evidence agree.
# 
# %% R1 evidence: rows that may be several abstracts joined together
long_rows = df.index[df["n_words_raw"] > 700]  # Evidence 1: ~4x the median length, worth checking
r1 = pd.DataFrame({
    "label": df.loc[long_rows, "Label"],
    "words": df.loc[long_rows, "n_words_raw"],
    "repeated_closing_phrase": df.loc[long_rows, "Abstract"].str.count("insight for management"),  # Evidence 2: the same closing phrase repeated = one per joined abstract
    "author_bylines": df.loc[long_rows, "Abstract"].str.count( r"(?:[A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+, )+[A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+ [A-Z]"),  # Evidence 3: author bylines (Name Surname, Name Surname ...) inside the text
})
print(r1)
df["r1_joined"] = df.index.isin(r1.index[(r1["repeated_closing_phrase"] > 1) | (r1["author_bylines"] > 1)])
print("\nRows meeting R1:", df.index[df["r1_joined"]].tolist())





# %% R2 evidence: AI-labelled abstracts whose content contradicts the label
PHYSICS = (r"inflaton|tachyon|inflationary model|slow-roll|primordial|large-scale structure|f\(R\)")
ML = (r"machine learning|neural|artificial intelligence|deep learning|random forest|LSTM|support vector|gradient boosting|learning algorithm|ensemble|classifier|XGBoost|genetic algorithm")

df["physics_subject"] = (df["Abstract"].str.contains(PHYSICS, case=False, regex=True) | df["DOI"].fillna("").str.contains(r"hep-|astro-ph|gr-qc", regex=True))  # Evidence 1: physics subject (vocabulary or arXiv physics category in the DOI)
df["ml_terms"] = df["Abstract"].str.count(f"(?i){ML}")  # Evidence 2: number of AI/ML method terms

print("Physics-subject abstracts:", df["physics_subject"].sum())
for i in df.index[df["physics_subject"]]:
    print(f"\n--- row {i} | {df.loc[i, 'Label']} | AI/ML terms: {df.loc[i, 'ml_terms']} | {df.loc[i, 'DOI']}")
    print(df.loc[i, "Abstract"][:250])

df["r2_contradicted"] = (df["Label"] == "ai") & df["physics_subject"] & (df["ml_terms"] == 0)  # R2 needs BOTH: labelled ai + physics subject + zero AI/ML content
print("\nRows meeting R2:", df.index[df["r2_contradicted"]].tolist())

# Do the physics terms appear anywhere else? (they should not, if they are specific to physics)
term_check = pd.DataFrame(
    [(t, df["Abstract"].str.contains(t, case=False, regex=True).sum(),
      df.loc[~df["physics_subject"], "Abstract"].str.contains(t, case=False, regex=True).sum())
     for t in PHYSICS.split("|")],
    columns=["term", "abstracts containing it", "of which NOT flagged physics"])
print(term_check.to_string(index=False))

# %% How were the physics papers identified? Evidence per row
phys_rows = df.index[df["physics_subject"]]
evidence = pd.DataFrame({
    "label": df.loc[phys_rows, "Label"],
    "DOI": df.loc[phys_rows, "DOI"],
    # arXiv DOIs can be checked on arxiv.org, where each paper's subject category is listed
    "arxiv_page": df.loc[phys_rows, "DOI"].str.extract(r"arxiv\.(.+)$", flags=re.I)[0]
                    .radd("https://arxiv.org/abs/"),
    # which physics terms each abstract actually contains
    "physics_terms_found": df.loc[phys_rows, "Abstract"].apply(
        lambda a: ", ".join(sorted({m.lower() for m in re.findall(PHYSICS, a, flags=re.I)}))),
})
print(evidence.to_string())

# %% Evidence chart: does each group contain AI/ML content?
group = np.select(
    [df["physics_subject"] & (df["Label"] == "ai"),
     df["physics_subject"] & (df["Label"] == "classical"),
     df["Label"] == "ai"],
    ["physics, labelled ai", "physics, labelled classical", "other ai-labelled"],
    default="other classical-labelled")
share = df.groupby(group)["ml_terms"].apply(lambda s: (s > 0).mean() * 100)
counts = pd.Series(group).value_counts()
print(pd.DataFrame({"% with AI/ML terms": share.round(1), "n": counts}))

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(share.index, share.values)
ax.bar_label(bars, labels=[f"{v:.0f}%  (n={counts[k]})" for k, v in share.items()])
ax.set_title("Share of abstracts mentioning any AI/ML method, by group")
ax.set_ylabel("% of abstracts in the group")
savefig(fig, "Q1_05_ai_content_by_group")

# %% Decision table: every candidate, its evidence, the rule, the decision
cands = df.index[df["r1_joined"] | df["physics_subject"] | df.index.isin(r1.index)]
decisions = pd.DataFrame({
    "label": df.loc[cands, "Label"],
    "words": df.loc[cands, "n_words_raw"],
    "physics_subject": df.loc[cands, "physics_subject"],
    "ml_terms": df.loc[cands, "ml_terms"],
    "rule_met": np.select([df.loc[cands, "r1_joined"], df.loc[cands, "r2_contradicted"]],
                          ["R1", "R2"], default="none"),
})
decisions["decision"] = np.where(decisions["rule_met"] == "none", "keep", "exclude")
print(decisions.to_string())
decisions.to_csv(TAB / "Q1_exclusion_decisions.csv")

exclude = decisions.index[decisions["decision"] == "exclude"]
df_model = df.drop(index=exclude).reset_index(drop=True)      # df itself stays complete
print(f"\nModelling rows: {len(df)} -> {len(df_model)} (excluded {len(exclude)})")
print(df_model["Label"].value_counts())

# %% Q1.1 Text denoising: apply clean_text and show what it removed
NOISE_PATTERNS = {
    "LaTeX":                r"\$[^$]*\$|\\[a-zA-Z]+",
    "keyword / JEL lists":  r"(?i)key ?words?\s*[:\-]|JEL",
    "section headings":     r"\b(?:Purpose|Findings|Background|Abstract)\s*[:\-.]",
    "HTML tags/entities":   r"<[^>]+>|&[a-zA-Z]+;",
    "URLs / e-mails":       r"http\S+|www\.\S+|\S+@\S+",
    "unicode dashes":       r"[\u2010-\u2015\u2212]",
    "digits":               r"\d",
    "punctuation":          r"[^\w\s]",
}
def noise_inventory(series):
    return pd.Series({name: series.str.contains(p, regex=True).sum()
                      for name, p in NOISE_PATTERNS.items()})

t0 = time.perf_counter()
df_model["clean"] = df_model["Abstract"].apply(clean_text)
print(f"Cleaned {len(df_model)} abstracts in {time.perf_counter() - t0:.1f} s")
print("Empty after cleaning:", (df_model["clean"].str.len() == 0).sum())

noise = pd.DataFrame({"before": noise_inventory(df_model["Abstract"]),
                      "after": noise_inventory(df_model["clean"])})
print(noise)
noise.to_csv(TAB / "Q1_noise_before_after.csv")

fig, ax = plt.subplots(figsize=(10, 4))
noise.plot(kind="barh", ax=ax)
ax.set_title("Abstracts containing each type of noise, before and after cleaning")
ax.set_xlabel("Number of abstracts")
savefig(fig, "Q1_06_noise_before_after")

# Size reduction
df_model["n_words_clean"] = df_model["clean"].str.split().str.len()
print("\nAverage words per abstract: raw",
      round(df_model["n_words_raw"].mean(), 1), "-> clean", round(df_model["n_words_clean"].mean(), 1))

# Three examples, raw vs clean (one structured, one with LaTeX-free physics removed, one ordinary)
for i in [0, df_model["n_words_raw"].idxmax(), 4]:
    print(f"\n--- row {i} ({df_model.loc[i, 'Label']}) ---")
    print("RAW  :", df_model.loc[i, "Abstract"][:300])
    print("CLEAN:", df_model.loc[i, "clean"][:300])
    
# %% Q1.2 (A) Which columns are relevant? DOI as a candidate feature
df_model["doi_prefix"] = df_model["DOI"].str.extract(r"(10\.\d+)")[0].fillna("missing")
prefix_ct = pd.crosstab(df_model["doi_prefix"], df_model["Label"])
prefix_ct["total"] = prefix_ct.sum(axis=1)
prefix_ct["% ai"] = (prefix_ct["ai"] / prefix_ct["total"] * 100).round(0)
top_prefix = prefix_ct.sort_values("total", ascending=False).head(8)
print(top_prefix)

fig, ax = plt.subplots(figsize=(9, 4))
top_prefix["% ai"].plot(kind="bar", ax=ax)
ax.axhline((df_model["Label"] == "ai").mean() * 100, linestyle="--", color="grey")
ax.set_title("Share of AI-labelled abstracts by DOI prefix (publisher) - a leakage risk")
ax.set_xlabel("DOI prefix (10.1109 = IEEE, 10.48550 = arXiv, 10.2139 = SSRN ...)")
ax.set_ylabel("% labelled ai")
savefig(fig, "Q1_07_doi_prefix_leakage")

# %% Q1.2 (A2) Evidence: how far does the DOI prefix alone get? (no abstract text)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(df_model[["doi_prefix"]], df_model["Label"], test_size=0.2,
                                              stratify=df_model["Label"], random_state=SEED)
doi_only = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                     ("m", LogisticRegression(class_weight="balanced", max_iter=1000))])
doi_only.fit(Xd_tr, yd_tr)
pd_pred = doi_only.predict(Xd_te)
print("DOI-only test macro-F1:", round(f1_score(yd_te, pd_pred, average="macro"), 3),
      "| bal. acc:", round(balanced_accuracy_score(yd_te, pd_pred), 3),
      "| AI recall:", round(recall_score(yd_te, pd_pred, pos_label="ai"), 3),
      "| accuracy:", round(accuracy_score(yd_te, pd_pred), 3))
print("Test abstracts whose DOI prefix never appears in training:",
      (~Xd_te["doi_prefix"].isin(Xd_tr["doi_prefix"])).sum(), "of", len(Xd_te))

# %% [markdown]
# **NOTES:** it was found that by using DOI a model can reach reasonable accuracy without even looking at the abstract, this is detrimental to the training of the models since it gives them the soltion without showing them how to get there. this would be the same as using the publisher alone to predict a label. if we go ahead and use DOI the model would fail when given papers that come from publishers that are not in this set or are rare in this set.
# 
# %% Q1.2 (B) Feature modelling: compare text representations, rank informative terms
y_all = (df_model["Label"] == "ai").astype(int)

# 1. How big and sparse is each representation?
reps = {
    "BoW unigrams":              CountVectorizer(),
    "TF-IDF unigrams":           TfidfVectorizer(),
    "TF-IDF uni+bigrams":        TfidfVectorizer(ngram_range=(1, 2)),
    "TF-IDF uni+bigrams, min_df=2": TfidfVectorizer(ngram_range=(1, 2), min_df=2),
}
rows = []
for name, vec in reps.items():
    X = vec.fit_transform(df_model["clean"])
    rows.append((name, X.shape[1], round(100 * X.nnz / (X.shape[0] * X.shape[1]), 3)))
rep_table = pd.DataFrame(rows, columns=["representation", "features", "% of matrix non-zero"])
print(rep_table)
rep_table.to_csv(TAB / "Q1_representations.csv", index=False)

# %% [markdown]
# **NOTES:** It was found that for my first attempt at the chi square i gave the wrong input. Chi square measured observed (O), i.e. how much of the term actually appears in that class. and Expected (E), i.e. how much would appear in each class if the term had nothing to do with the label. the score is (O-E)^2/E which means a small gap between observed and expected means the term is less linked to the class. the score is then converted into a p value using the chi-square distribution. when i ran it the first time i used the TF-IDF which is essentially observed totals in units that don't work well with the chi square. because TF-IDF values are small by definition, only about 40 terms passed when i used the chi-square method since it relies on actual counts, not weights scaled per abstract. when i multiplied the TF-IDF values by 10 i got more than 2.3k significant terms and scores exactly 10x higher, so the result of the chi square depended on the units. to rectify this i ran chi square again using raw counts, which gave me more than 2.8k significant terms.

# %% Q1.2 (B2) First attempt: chi-square on TF-IDF weights (kept to show the working)

# 2. Which terms separate the classes? Chi-square test per term
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X = vec.fit_transform(df_model["clean"])
terms_all = vec.get_feature_names_out()
chi_scores, p_values = chi2(X, y_all)
mean_ai = np.asarray(X[y_all.values == 1].mean(axis=0)).ravel()
mean_cl = np.asarray(X[y_all.values == 0].mean(axis=0)).ravel()
chi_df = pd.DataFrame({"term": terms_all, "chi2": chi_scores, "p_value": p_values,
                       "favours": np.where(mean_ai > mean_cl, "ai", "classical")}
                      ).sort_values("chi2", ascending=False)
print("Terms significant at p < 0.05:", (chi_df["p_value"] < 0.05).sum(), "of", len(chi_df))

# NEW - Scaling test: a valid test statistic should not change when the input is rescaled
chi_x10, p_x10 = chi2(X * 10, y_all)
print("Significant terms - TF-IDF:", (p_values < 0.05).sum(),
      "| TF-IDF x 10:", (p_x10 < 0.05).sum(),
      "| largest score ratio:", round(chi_x10.max() / chi_scores.max(), 2))

chi_df.head(40).to_csv(TAB / "Q1_08a_chi2_tfidf_first_attempt.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, cls, colour in zip(axes, ["ai", "classical"], ["tab:blue", "tab:orange"]):
    top = chi_df[chi_df["favours"] == cls].head(15).iloc[::-1]
    ax.barh(top["term"], top["chi2"], color=colour)
    ax.set_title(f"Top 15 '{cls}' terms - TF-IDF input (first attempt)")
    ax.set_xlabel("Chi-square score on TF-IDF weights (scale-dependent)")
savefig(fig, "Q1_08a_chi2_tfidf_first_attempt")

# %% Q1.2 (C) Chi-square on word COUNTS (chi2 is defined for frequencies, not TF-IDF weights)
count_vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
X_counts = count_vec.fit_transform(df_model["clean"])
terms_all = count_vec.get_feature_names_out()
chi_scores, p_values = chi2(X_counts, y_all)

# Which class does each term lean towards? Compare average count per abstract in each class
mean_ai = np.asarray(X_counts[y_all.values == 1].mean(axis=0)).ravel()
mean_cl = np.asarray(X_counts[y_all.values == 0].mean(axis=0)).ravel()
chi_df = pd.DataFrame({"term": terms_all, "chi2": chi_scores, "p_value": p_values,
                       "favours": np.where(mean_ai > mean_cl, "ai", "classical")}
                      ).sort_values("chi2", ascending=False)

# Show why counts were used: how many terms are significant under each input
_, p_tfidf = chi2(X, y_all)  # CHANGED - X is the TF-IDF matrix from the first-attempt cell
print("Terms significant at p < 0.05 - TF-IDF input:", (p_tfidf < 0.05).sum(),
      "| count input:", (p_values < 0.05).sum(), "of", len(chi_df))
print(chi_df.head(20).to_string(index=False))
chi_df.head(40).to_csv(TAB / "Q1_08b_chi2_counts.csv", index=False)  # CHANGED

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, cls, colour in zip(axes, ["ai", "classical"], ["tab:blue", "tab:orange"]):
    top = chi_df[chi_df["favours"] == cls].head(20).iloc[::-1]
    ax.barh(top["term"], top["chi2"], color=colour)
    ax.set_title(f"Top 20 terms pointing to '{cls}'")
    ax.set_xlabel("Chi-square score on word counts (higher = stronger link)")
savefig(fig, "Q1_08b_chi2_counts")  # CHANGED

# %% Q1.3 (A) Train/test split - once, stratified, before any model sees the data

X_text = df_model["clean"]
y = df_model["Label"]          # keep the string labels; "ai" is the positive class

X_tr, X_te, y_tr, y_te = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=SEED)

print("Train:", len(X_tr), "| Test:", len(X_te))
print(pd.DataFrame({"train %": (y_tr.value_counts(normalize=True) * 100).round(1),
                    "test %":  (y_te.value_counts(normalize=True) * 100).round(1)}))

# %% Q1.3 (B) Cross-validation folds and metrics - reused for every model in Q2
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = {
    "accuracy":  "accuracy",
    "f1_macro":  "f1_macro",
    "bal_acc":   "balanced_accuracy",
    "ai_recall": make_scorer(recall_score, pos_label="ai"),
}

# %% Q1.3 (C) Two baselines: no-skill floor and a simple learned model
# The vectoriser sits INSIDE the pipeline, so each CV fold builds its vocabulary from its own training part only
baselines = {
    "Dummy (most frequent)": Pipeline([("v", CountVectorizer()),
                                       ("m", DummyClassifier(strategy="most_frequent"))]),
    "NB + BoW":              Pipeline([("v", CountVectorizer()),
                                       ("m", MultinomialNB())]),
}

rows = []
for name, pipe in baselines.items():
    cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, return_train_score=True)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = pipe.predict(X_te)
    rows.append({
        "model": name,
        "CV train macro-F1": cvres["train_f1_macro"].mean(),
        "CV val macro-F1":   cvres["test_f1_macro"].mean(),
        "CV val std":        cvres["test_f1_macro"].std(),
        "test accuracy":     accuracy_score(y_te, pred),
        "test macro-F1":     f1_score(y_te, pred, average="macro"),
        "test bal. acc":     balanced_accuracy_score(y_te, pred),
        "test AI recall":    recall_score(y_te, pred, pos_label="ai"),
        "fit time (s)":      fit_s,
    })

baseline_results = pd.DataFrame(rows).set_index("model").round(3)
print(baseline_results.T)
baseline_results.to_csv(TAB / "Q1_baseline_results.csv")

# %% Q1.3 (D) Test-set report for the learned baseline
nb = baselines["NB + BoW"]
nb_pred = nb.predict(X_te)
print(classification_report(y_te, nb_pred, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(y_te, nb_pred, labels=["ai", "classical"], ax=axes[0], colorbar=False)
axes[0].set_title("NB + BoW: confusion matrix (test set)")
ConfusionMatrixDisplay.from_predictions(y_te, baselines["Dummy (most frequent)"].predict(X_te),
                                        labels=["ai", "classical"], ax=axes[1], colorbar=False)
axes[1].set_title("Dummy: confusion matrix (test set)")
savefig(fig, "Q1_09_baseline_confusion_matrices")

fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_estimator(nb, X_te, y_te, pos_label="ai", ax=ax, name="NB + BoW")
ax.axhline((y_te == "ai").mean(), linestyle="--", color="grey", label="no-skill (share of ai)")
ax.set_title("Precision-recall curve for the 'ai' class (test set)")
ax.legend()
savefig(fig, "Q1_10_baseline_precision_recall")

# %% Q1.3 (E) Learning curve: does the model improve with more training data?
sizes, train_sc, val_sc = learning_curve(nb, X_tr, y_tr, cv=cv, scoring="f1_macro",
                                         train_sizes=np.linspace(0.1, 1.0, 8))

fig, ax = plt.subplots(figsize=(7, 4.5))
for scores, label in [(train_sc, "training"), (val_sc, "cross-validation")]:
    m, s = scores.mean(axis=1), scores.std(axis=1)
    ax.plot(sizes, m, marker="o", label=label)
    ax.fill_between(sizes, m - s, m + s, alpha=0.2)
ax.set_title("Learning curve: NB + BoW")
ax.set_xlabel("Number of training abstracts")
ax.set_ylabel("Macro-F1")
ax.legend()
savefig(fig, "Q1_11_learning_curve_nb")

print(pd.DataFrame({"train size": sizes, "train F1": train_sc.mean(axis=1).round(3),
                    "val F1": val_sc.mean(axis=1).round(3)}).to_string(index=False))

# %% Q1.3 (F) Sensitivity check: do the R1/R2 exclusions change the result?
df_all = df.copy()  # deduplicated, but NOT excluded
df_all["clean"] = df_all["Abstract"].apply(clean_text)
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(df_all["clean"], df_all["Label"], test_size=0.2,
                                              stratify=df_all["Label"], random_state=SEED)
nb_all = Pipeline([("v", CountVectorizer()), ("m", MultinomialNB())])
cv_all = cross_validate(nb_all, Xa_tr, ya_tr, cv=cv, scoring=scoring)

sens = pd.DataFrame({
    "rows": [len(df_model), len(df_all)],
    "CV val macro-F1": [baseline_results.loc["NB + BoW", "CV val macro-F1"], cv_all["test_f1_macro"].mean()],
    "CV val AI recall": [cross_validate(nb, X_tr, y_tr, cv=cv, scoring=scoring)["test_ai_recall"].mean(),
                         cv_all["test_ai_recall"].mean()],
}, index=["with exclusions", "without exclusions"]).round(3)
print(sens)
sens.to_csv(TAB / "Q1_exclusion_sensitivity.csv")