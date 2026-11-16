# %% [markdown]
# # ITNPA4-34 - PROJECT 1 - EDUVOS ONLINE PART TIME - EDUV8426028
# **Michael van Niekerk** - Questions 3 and 4: summarisation, topic modelling

# %% Setup: imports, paths, helpers
import re, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    BASE = Path(__file__).resolve().parents[1]
except NameError:
    BASE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
print("Project folder:", BASE)

FIG = BASE / "outputs" / "figures"
TAB = BASE / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

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

# %% Q3.1 Build the dataframe from research_corpus.zip
ZIP = BASE / "data" / "research_corpus.zip"

def article_number(name):
    # "research_corpus/research_10.txt" -> 10 (numeric, so 10 sorts after 9)
    return int(re.search(r"research_(\d+)\.txt$", name).group(1))

rows = []
with zipfile.ZipFile(ZIP) as z:
    names = sorted([n for n in z.namelist() if n.endswith(".txt")], key=article_number)
    for n in names:
        text = z.read(n).decode("utf-8")   # cp1252 fails on research_3; the files are UTF-8
        rows.append({"article_id": article_number(n), "file": Path(n).name, "content": text})

df_raw = pd.DataFrame(rows)
print(df_raw[["article_id", "file"]])

# %% Q3.1 Describe each article (rough counts BEFORE any cleaning)
def rough_sentences(t):
    # rough split: full stop / ? / ! followed by white space and a capital or opening quote
    return len(re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c"])', t.strip()))

df_raw["n_chars"] = df_raw["content"].str.len()
df_raw["n_words"] = df_raw["content"].str.split().str.len()
df_raw["n_sentences"] = df_raw["content"].apply(rough_sentences)
df_raw["n_lines"] = df_raw["content"].apply(lambda t: len(t.splitlines()))
df_raw["n_crlf"] = df_raw["content"].str.count("\r\n")
df_raw["n_non_ascii"] = df_raw["content"].apply(lambda t: sum(ord(c) > 127 for c in t))
df_raw["chars_per_line"] = (df_raw["n_chars"] / df_raw["n_lines"]).round(1)
df_raw["first_line"] = df_raw["content"].apply(lambda t: t.splitlines()[0][:60])

summary_cols = ["article_id", "n_chars", "n_words", "n_sentences", "n_lines", "chars_per_line", "n_crlf", "n_non_ascii", "first_line"]
print(df_raw[summary_cols].to_string(index=False))
df_raw[summary_cols].to_csv(TAB / "Q3_01_corpus_overview.csv", index=False)

# %% Q3.1 Chart: size and layout of the five articles
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(df_raw["article_id"].astype(str), df_raw["n_words"], color="tab:blue")
ax[0].set_title("Words per article (raw)")
ax[0].set_xlabel("article_id")
ax[0].set_ylabel("words")
ax[1].bar(df_raw["article_id"].astype(str), df_raw["chars_per_line"], color="tab:orange")
ax[1].set_title("Average characters per line (raw)")
ax[1].set_xlabel("article_id")
ax[1].set_ylabel("characters per line")
savefig(fig, "Q3_01_corpus_overview")

# %% Q3.1 Look at how each article opens (abstract? title? boilerplate?)
for _, r in df_raw.iterrows():
    print("=" * 20, "article", r["article_id"], "- first 6 lines (cut at 110 characters)")
    for line in r["content"].splitlines()[:6]:
        print("   ", line[:110])
    has_abs = bool(re.search(r"^Abstract\s*$", r["content"], re.M))
    has_kw = bool(re.search(r"^Keywords?\b", r["content"], re.M | re.I))
    print("   'Abstract' heading line:", has_abs, "| 'Keywords' line:", has_kw)
    
# %% Q3.2 Step 0: work on a COPY and normalise line endings (original df_raw is never touched)
df_clean = df_raw[["article_id", "file", "content"]].copy()
df_clean["text"] = df_clean["content"].str.replace("\r\n", "\n", regex=False)
print("CRLF before:", int(df_raw["n_crlf"].sum()), "| after:", int(df_clean["text"].str.count("\r").sum()))

# %% Q3.2 Step 1: noise audit - how often does each suspected noise pattern occur in each article?
AUDIT = {
    "footnote digits, LOOSE  ([.,;:)] then 1-2 digits)":  r"(?<=[a-z\)])[.,;:\)]\s?\d{1,2}(?:,\s?\d{1,2})*(?=\s|$)",
    "footnote digits, STRICT (.1 / ).2 / .3, 4)":          r"(?<=[a-z\)])[.\)]\d{1,2}(?:,\s?\d{1,2})*(?=\s|$)",
    "citation in brackets  (Smith, 2004; Lee 1999)":       r"\([^()]*?\b(?:1[89]|20)\d{2}[a-z]?[^()]*?\)",
    "citation in text  Smith (2004)":                      r"\b[A-Z][A-Za-z\-]+(?: (?:&|and) [A-Z][A-Za-z\-]+| et al\.)? \((?:1[89]|20)\d{2}[a-z]?\)",
    "line-end hyphen (word-<newline>word)":                r"[A-Za-z]-\n[a-z]",
    "URL":                                                 r"https?://\S+|www\.\S+",
    "e-mail":                                              r"\b[\w.\-]+@[\w.\-]+\.\w+\b",
    "section number at line start (41.2.1)":               r"(?m)^\d+(?:\.\d+)+\.?\s",
#    "running head / page number (1100 • NAME)":            r"(?m)^(?:\d{3,4} \u2022 .*|.* \u2022 \d{3,4}|\d{4})$",
    "running head / page number (1100 • NAME)":            r"(?m)^(?:\d{3,4} \u2022 .*|.* \u2022 \d{3,4}|\d{4}|[A-Z]{3,9} \d{4} \d{1,3}|\d{1,3} EDUCATIONAL RESEARCHER|Journal of Social Change \d+|[A-Z][a-z]+ \d{3}|\d{3} Qualitative Research 15\(2\)|[A-Z][A-Za-z]+(?:,? (?:& )?[A-Z][A-Za-z]+)*, \d{4})$",
    "figure / table caption":                              r"(?m)^(?:Figure|Fig\.|TABLE|Table)\s*[\d.]+",
    "(c) / Copyright":                                     r"\u00a9|Copyright|All rights reserved",
    "blank lines":                                         r"(?m)^\s*$",
    "bullet / arrow characters":                           r"[\u2022\u21d2\u2192]",
    "control / invisible characters (\\x01, BOM, ...)":    r"[\x00-\x09\x0b-\x1f\x7f-\x9f\u00a0\u200b\u00ad\ufeff]",
    "curly quotes and dashes":                             r"[\u201c\u201d\u2018\u2019\u2013\u2014]",
}
audit = pd.DataFrame({name: [len(re.findall(p, t)) for t in df_clean["text"]] for name, p in AUDIT.items()},
                     index=df_clean["article_id"]).T
audit.columns = [f"article_{i}" for i in audit.columns]
print(audit.to_string())
audit.to_csv(TAB / "Q3_02a_noise_audit.csv")

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(np.log1p(audit.values), cmap="Blues", aspect="auto")
ax.set_xticks(range(audit.shape[1])); ax.set_xticklabels(audit.columns, rotation=30, ha="right")
ax.set_yticks(range(audit.shape[0])); ax.set_yticklabels(audit.index, fontsize=8)
for (r, c), v in np.ndenumerate(audit.values):
    ax.text(c, r, v, ha="center", va="center", fontsize=8)
ax.set_title("Suspected noise per article (raw counts; colour = log scale)")
savefig(fig, "Q3_02a_noise_audit")

# %% Q3.2 Step 2: do the LOOSE footnote hits include false positives?  (print each hit in context)
for aid, t in zip(df_clean["article_id"], df_clean["text"]):
    loose  = {m.start() for m in re.finditer(AUDIT["footnote digits, LOOSE  ([.,;:)] then 1-2 digits)"], t)}
    strict = {m.start() for m in re.finditer(AUDIT["footnote digits, STRICT (.1 / ).2 / .3, 4)"], t)}
    for pos in sorted(loose - strict):
        print(f"article {aid}  LOOSE-only hit:", repr(t[max(0, pos - 40):pos + 30]))

# %% Q3.2 Step 3: is any article a copy of itself?  (exact-duplicate test: first half vs second half)
for aid, t in zip(df_clean["article_id"], df_clean["text"]):
    flat = re.sub(r"\s+", " ", t).strip()
    half = len(flat) // 2
    same = flat[:half].strip() == flat[half:].strip()
    print(f"article {aid}: first half == second half (ignoring white space): {same}")

dup = df_clean.loc[df_clean["article_id"] == 1, "text"].iloc[0]
starts = [m.start() for m in re.finditer("The purposes of this article", dup)]
print("article 1: opening sentence found at character positions", starts, "| file length", len(dup))
copy1, copy2 = dup[starts[0]:starts[1]], dup[starts[1]:]
norm = lambda s: re.sub(r"\s+", " ", s).strip()
print("copy 1 == copy 2 after white-space normalisation:", norm(copy1) == norm(copy2))
print("words per copy:", len(norm(copy1).split()), "and", len(norm(copy2).split()))

# %% Q3.2 measuring tool: the same three numbers after every rule (words, rough sentences, short fragments)
norm = lambda s: re.sub(r"\s+", " ", s).strip()
funnel = []

def metrics(t):
    flat = norm(t)
    sents = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c"])', flat) if s]
    return len(flat.split()), len(sents), sum(len(s.split()) < 5 for s in sents)

def snap(step, texts):
    for aid, t in zip(df_clean["article_id"], texts):
        w, s, f = metrics(t)
        funnel.append({"step": step, "article_id": aid, "words": w, "sentences": s, "fragments_lt5": f})

snap("0 raw file", df_raw["content"])

# %% Q3.2 Rule 1: remove an exact second copy of the article (found in the audit: article 1)
norm = lambda s: re.sub(r"\s+", " ", s).strip()

def drop_exact_copy(t):
    # If the opening 60 characters come back later AND everything from there on equals everything before it
    # (ignoring white space), keep only the first copy. Any other article is returned unchanged.
    second = t.find(t[:60], 1)
    if second == -1:
        return t
    first_copy, rest = t[:second], t[second:]
    return first_copy if norm(first_copy) == norm(rest) else t

df_clean["words_raw"] = df_clean["text"].str.split().str.len()
df_clean["text"] = df_clean["text"].apply(drop_exact_copy)
df_clean["words_rule1"] = df_clean["text"].str.split().str.len()
print(df_clean[["article_id", "words_raw", "words_rule1"]].to_string(index=False))
snap("1 after duplicate removal", df_clean["text"])
print("Articles changed by Rule 1:", df_clean.loc[df_clean["words_raw"] != df_clean["words_rule1"], "article_id"].tolist())

# test: after Rule 1, no long sentence appears twice in any article
for aid, t in zip(df_clean["article_id"], df_clean["text"]):
    sents = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c"])', norm(t)) if len(s.split()) >= 8]
    print(f"article {aid}: long sentences {len(sents)}, repeated {len(sents) - len(set(sents))}")
    
# %% Q3.2 Rule 2: separate front matter, abstract, keywords/boilerplate and body  (boundaries checked by eye below)
# abstract_from: text that starts the abstract (if it ends with a new line it is a heading and is skipped)
# abstract_to  : text that ends the abstract (not included)
# body_from    : text where the body starts
SPLIT = {
    1: dict(abstract_from="The purposes of this article", abstract_to="For more than a century", body_from="For more than a century"),
    2: dict(abstract_from="Abstract\n", abstract_to="Keywords\n", body_from="The main purpose of this article"),
    3: dict(abstract_from=None, abstract_to=None, body_from="41.1 INTRODUCTION"),
    4: dict(abstract_from="Abstract\n", abstract_to="Similar content being viewed by others", body_from="Introduction\n"),
    5: dict(abstract_from="One approach to promote social change", abstract_to="Keywords:", body_from="Introduction\n"),
}

def split_article(t, s):
    b0 = t.index(s["body_from"])
    if s["abstract_from"] is None:
        return t[:b0], "", "", t[b0:]
    a0 = t.index(s["abstract_from"])
    front = t[:a0]
    if s["abstract_from"].endswith("\n"):
        a0 += len(s["abstract_from"])
    a1 = t.index(s["abstract_to"], a0)
    return front, t[a0:a1], t[a1:b0], t[b0:]

parts = [split_article(t, SPLIT[aid]) for aid, t in zip(df_clean["article_id"], df_clean["text"])]
df_clean["front"], df_clean["abstract"], df_clean["between"], df_clean["body"] = zip(*parts)

# check 1: the pieces put back together equal the article exactly (nothing lost, nothing added)
for aid, t, f, a, b, bo in zip(df_clean["article_id"], df_clean["text"], df_clean["front"], df_clean["abstract"], df_clean["between"], df_clean["body"]):
    heading = SPLIT[aid]["abstract_from"] if (SPLIT[aid]["abstract_from"] or "").endswith("\n") else ""
    print(f"article {aid}: pieces rebuild the article exactly: {f + heading + a + b + bo == t}")

# check 2: look at every boundary
for _, r in df_clean.iterrows():
    print("=" * 18, "article", r["article_id"])
    print("  FRONT    :", repr(norm(r["front"])[:140]))
    print("  ABSTRACT :", repr(norm(r["abstract"])[:90]), "...", repr(norm(r["abstract"])[-70:]))
    print("  BETWEEN  :", repr(norm(r["between"])[:200]))
    print("  BODY     :", repr(norm(r["body"])[:90]), "...", repr(norm(r["body"])[-60:]))

snap("2 after front matter and abstract removed", df_clean["body"])

# table: reference abstracts and the word budget for the bake-off (written down before any summarising)
df_clean["abstract_words"] = df_clean["abstract"].apply(lambda s: len(s.split()))
budget = df_clean[["article_id", "abstract_words"]].copy()
budget["body_words"] = df_clean["body"].str.split().str.len()
print(budget.to_string(index=False))
budget.to_csv(TAB / "Q3_02b_abstract_word_budget.csv", index=False)    

# %% Q3.2 Rule 3: page furniture (running heads, journal metadata, table and figure blocks)
# Single-line patterns come from the audit and the context printouts; block rules remove a caption plus what follows it.
FURNITURE_LINES = {
    "contact block":        r"^(?:Corresponding author:|Email: \S+@\S+|.*, School of .*USA\.)$",
    "journal code line":    r"^\d+[A-Z]{3}\d+\.\d+/\d+.*$",
    "lone year / page no.": r"^\d{4}$",
    "lone 'Article'":       r"^Article$",
    "running head (journal)": r"^\d{3} Qualitative Research 15\(2\)$",
    "running head (book 1)":  r"^\d{3,4} \u2022 [A-Z ,]+$",
    "running head (book 2)":  r"^\d+\. [A-Z ]+ \u2022 \d{3,4}$",
    "running head (author, year)": r"^[A-Z][A-Za-z]+(?:,? (?:& )?[A-Z][A-Za-z]+)*, \d{4}$",
    "running head (journal name)": r"^Journal of Social Change \d+$",
    "running head (author, page)": r"^[A-Z][a-z]+ \d{3}$",
    "running head (month year page)": r"^[A-Z]{3,9} \d{4} \d{1,3}$",
    "running head (page journal)":    r"^\d{1,3} EDUCATIONAL RESEARCHER$",
    "figure caption line":    r"^(?:Figure|Fig\.) ?\d+.*$",
}
FURNITURE_BLOCKS = {
    # name: (first line of the block, stop line, is the stop line removed too?)
    # The two tables in article 3 are found by hand (their cells look like ordinary short lines), so each has its own stop line.
    "table 41.1": (r"^TABLE 41\.1\.", r"^generated by applied research", False),
    "table 41.2": (r"^TABLE 41\.2\.", r"\u21d2", True),
    "figure (Fig. N to 'Full size image')": (r"^Fig\. \d+$", r"^Full size image$", True),
}

def remove_furniture(text):
    lines, keep, removed = text.split("\n"), [], []
    i = 0
    while i < len(lines):
        l = lines[i]
        blk = next((name for name, (start, _, _) in FURNITURE_BLOCKS.items() if re.match(start, l)), None)
        hit = next((name for name, p in FURNITURE_LINES.items() if re.match(p, l)), None)
        if blk:
            _, stop, inclusive = FURNITURE_BLOCKS[blk]
            j = i + 1
            while j < len(lines) and not re.search(stop, lines[j]):
                j += 1
            end = j + 1 if inclusive else j
            removed.append((blk, lines[i:end]))
            i = end
        elif hit:
            removed.append((hit, [l])); i += 1
        else:
            keep.append(l); i += 1
    return "\n".join(keep), removed

df_clean["words_rule2"] = df_clean["body"].str.split().str.len()
res = [remove_furniture(b) for b in df_clean["body"]]
df_clean["body"] = [r[0] for r in res]
df_clean["removed_furniture"] = [r[1] for r in res]
df_clean["words_rule3"] = df_clean["body"].str.split().str.len()

# evidence: print every removed line / block (cut at 100 characters)
for aid, rem in zip(df_clean["article_id"], df_clean["removed_furniture"]):
    print("=" * 14, "article", aid, "-", len(rem), "removals")
    for name, ls in rem:
        print(f"  [{name}] {len(ls)} line(s)")
        for l in ls[:8]:
            print("       ", repr(l[:100]))

furn = df_clean[["article_id", "words_rule2", "words_rule3"]].assign(words_removed=lambda d: d["words_rule2"] - d["words_rule3"])
snap("3 after page furniture", df_clean["body"])
print(furn.to_string(index=False))
furn.to_csv(TAB / "Q3_02c_furniture_removed.csv", index=False)

# test: re-run the audit patterns on the new body text - they should now be zero
for name in ["e-mail", "running head / page number (1100 \u2022 NAME)", "figure / table caption", "(c) / Copyright"]:
    print(f"{name:45}", [len(re.findall(AUDIT[name], t)) for t in df_clean["body"]])
    
# %% Q3.2 Rule 4: invisible characters and bullet markers (found by the audit row: \x01 in article 3, BOM in article 4)
INVISIBLE = r"[\x00-\x09\x0b-\x1f\x7f-\x9f\u200b\u00ad\ufeff]"
print("article 3, first bullet line before:", repr(next(l for l in df_clean.loc[df_clean["article_id"] == 3, "body"].iloc[0].split("\n") if "\x01" in l)))
df_clean["body"] = df_clean["body"].apply(lambda t: re.sub(INVISIBLE, "", t.replace("\x01 ", "")))
print("left after the rule:", [len(re.findall(INVISIBLE, t)) for t in df_clean["body"]])
snap("4 after invisible characters", df_clean["body"])

# %% Q3.2 Rule 5: footnote digits glued to the end of a sentence (strict pattern, see the loose-pattern false positives above)
FOOTNOTE = r"(?<=[a-z\)])([.\)])\d{1,2}(?:,\s?\d{1,2})*(?=\s|$)"
found = [(aid, m.group(0)) for aid, t in zip(df_clean["article_id"], df_clean["body"]) for m in re.finditer(FOOTNOTE, t, flags=re.M)]
print("footnote markers found:", found)
df_clean["body"] = df_clean["body"].apply(lambda t: re.sub(FOOTNOTE, r"\1", t, flags=re.M))
print("left after the rule:", sum(len(re.findall(FOOTNOTE, t, flags=re.M)) for t in df_clean["body"]))
snap("5 after footnotes", df_clean["body"])

# %% Q3.2 Rule 6: headings (they have no full stop, so they would glue onto the next sentence)
SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs"}
SECTION = r"^\d+(?:\.\d+)+\.?\s+"
END = ".!?:;,)\u201d\"\u2019'"

def clean_line(l):
    return re.sub(SECTION, "", l.strip())

def prev_ok(prev):
    return prev is None or prev.strip() == "" or prev.strip()[-1] in ".!?\u201d\")"

def starts_new(s):
    return s is not None and s.strip() != "" and (s.strip()[0].isupper() or s.strip()[0].isdigit() or s.strip()[0] in "\u201c\"")

def one_line_heading(line, prev, nxt):
    s = clean_line(line)
    if not s or s[-1] in "." and not s.isupper():
        return False
    if s.isupper() and len(s.split()) <= 8:                      # ALL CAPS line: "RESEARCH", "41.1 INTRODUCTION"
        return True
    if len(s.split()) > 12 or s[-1] in END or not s[0].isupper():
        return False
    big = [w for w in s.split() if w.lower() not in SMALL]
    titleish = sum(w[0].isupper() for w in big) / max(1, len(big)) >= 0.7
    return (titleish or len(s.split()) <= 4) and prev_ok(prev) and starts_new(nxt)

def two_line_heading(a, b, c, prev):
    sa, sb = clean_line(a), clean_line(b)
    return (sa and sb and sa[0].isupper() and len(sa.split()) <= 10 and sa[-1] not in END
            and len(sb.split()) <= 7 and sb[-1] not in END and prev_ok(prev) and starts_new(c))

def drop_headings(text):
    lines, keep, dropped, i = text.split("\n"), [], [], 0
    while i < len(lines):
        prev = lines[i - 1] if i > 0 else None
        l = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        nxt2 = lines[i + 2] if i + 2 < len(lines) else None
        if nxt is not None and nxt2 is not None and not one_line_heading(l, prev, nxt) and two_line_heading(l, nxt, nxt2, prev):
            dropped.append((l.strip() + " / " + nxt.strip())); i += 2; continue
        if one_line_heading(l, prev, nxt):
            dropped.append(l.strip()); i += 1; continue
        m = re.match(SECTION + r"([^.]{3,80}\.)\s+(?=[A-Z])", l)        # numbered heading that runs on into the sentence
        if m:
            dropped.append(m.group(0).strip())
            l = l[m.end():]
        keep.append(l); i += 1
    return "\n".join(keep), dropped

res = [drop_headings(b) for b in df_clean["body"]]
df_clean["body"] = [r[0] for r in res]
for aid, (_, d) in zip(df_clean["article_id"], res):
    print(f"article {aid}: {len(d)} headings removed")
    for h in d:
        print("     ", repr(h[:100]))
snap("6 after headings", df_clean["body"])

# %% Q3.2 Rule 7: join the hard line breaks (blank line = new paragraph, single line break = same sentence)
def join_lines(t):
    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    return "\n\n".join(norm(p) for p in paras)

df_clean["body"] = df_clean["body"].apply(join_lines)
df_clean["n_paragraphs"] = df_clean["body"].apply(lambda t: len(t.split("\n\n")))
print(df_clean[["article_id", "n_paragraphs"]].to_string(index=False))
print("single line breaks left inside a paragraph:", sum(len(re.findall(r"(?<!\n)\n(?!\n)", t)) for t in df_clean["body"]))
snap("7 after joining lines", df_clean["body"])

# %% Q3.2 Rule 8: in-text citations and page references  (they are not content and they break sentence splitting)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
YEAR = r"(?:1[89]|20)\d{2}[a-z]?"
# (a) a bracket that holds Name + year, with no bracket inside:  (Smith, 2004; Lee & Kim, 1999)  (Dunya et al., 2011)
PAREN_CITE = rf"\((?=[^()]*\b(?!(?:{MONTHS})\b)[A-Z][A-Za-z’'\-]+(?: et al\.)?,? {YEAR}\b)[^()]*\)"
# (b) the year after a name in running text:  Denzin (1970, 1978) -> Denzin   /  Jackson (1996, p. 6) -> Jackson
NARR_CITE = rf"\s*\({YEAR}(?:[,;] ?(?:{YEAR}|pp?\. ?\d+(?:[-–]\d+)?))*\)"
# (c) a bracket that is only a page reference:  (p. 237)
PAGE_REF = r"\s*\(pp?\. ?\d+(?:[-–]\d+)?\)"

def cite_hits(t):
    return {"bracket citation": re.findall(PAREN_CITE, t), "year after a name": re.findall(NARR_CITE, t), "page reference": re.findall(PAGE_REF, t)}

before_hits = [cite_hits(t) for t in df_clean["body"]]
for aid, h in zip(df_clean["article_id"], before_hits):
    print(f"article {aid}:", {k: len(v) for k, v in h.items()})

# review 1: every DIFFERENT bracket the rule will remove (cut at 90 characters) - read them for false positives
allp = sorted({m.strip() for h in before_hits for m in h["bracket citation"]})
print(len(allp), "different bracket citations; the shortest 8 and the longest 8:")
for m in allp[:8] + allp[-8:]:
    print("    ", repr(m[:90]))
print("year-after-a-name examples:", sorted({m.strip() for h in before_hits for m in h["year after a name"]})[:12])
print("page references:", sorted({m.strip() for h in before_hits for m in h["page reference"]}))

LOWER_OK = {"and", "et", "al", "see", "cf", "also", "in", "eds", "ed", "de", "van", "von", "der", "e.g", "e.g.", "i.e.", "p", "pp"}

def only_citation(m):
    # keep a bracket that is really a sentence with a citation inside it: more than 3 ordinary lower-case words
    words = [w.strip(",;.") for w in m.group(0)[1:-1].split() if w[:1].islower()]
    return "" if sum(w not in LOWER_OK for w in words) <= 3 else m.group(0)

def drop_citations(t):
    t = re.sub(PAREN_CITE, only_citation, t)
    t = re.sub(NARR_CITE, "", t)
    t = re.sub(PAGE_REF, "", t)
    t = re.sub(r"\(\s*\)", "", t)                    # empty brackets left behind
#    t = re.sub(r"[ \t]+([.,;:!?])", r"\1", t)         # space before punctuation
    t = re.sub(r"[ \t]+([.,;:!?])(?!\d)", r"\1", t)    # space before punctuation (not before a decimal such as "alpha = .77")
    return re.sub(r"[ \t]{2,}", " ", t)

df_clean["body"] = df_clean["body"].apply(drop_citations)

# review 2: brackets that still hold a year (should be only things that are NOT citations)
for aid, t in zip(df_clean["article_id"], df_clean["body"]):
    left = re.findall(rf"\([^()]*\b{YEAR}\b[^()]*\)", t)
    print(f"article {aid}: brackets with a year left: {len(left)}", [m[:60] for m in left[:5]])
snap("8 after citations", df_clean["body"])
print("decimals that kept their space:", [len(re.findall(r"[A-Za-z=] \.\d", t)) for t in df_clean["body"]])

# %% Q3.2 Rule 9: curly quotes, dashes and other non-ASCII characters -> plain forms
import unicodedata, collections
#PLAIN = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": " - ", "−": "-", " ": " ", "…": "..."}
PLAIN = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": ", ", "\u2212": "-", "\u00a0": " ", "\u2026": "..."}
before = collections.Counter(ch for t in df_clean["body"] for ch in t if ord(ch) > 127)
print("non-ASCII characters before:", sum(before.values()), {unicodedata.name(c): n for c, n in before.most_common(12)})

def plain(t):
    for a, b in PLAIN.items():
        t = t.replace(a, b)
    t = re.sub(r"[ \t]+,", ",", t)                    # no space before the comma that replaced a dash
    return re.sub(r"[ \t]{2,}", " ", t)

df_clean["body"] = df_clean["body"].apply(plain)
after = collections.Counter(ch for t in df_clean["body"] for ch in t if ord(ch) > 127)
print("non-ASCII characters after :", sum(after.values()), {unicodedata.name(c): n for c, n in after.most_common(12)})
snap("9 after quotes and dashes", df_clean["body"])

# %% Q3.2 Cleaning funnel (final): words, sentences and short fragments after every rule
fun = pd.DataFrame(funnel)
for col in ["words", "sentences", "fragments_lt5"]:
    print(f"\n{col}")
    print(fun.pivot(index="step", columns="article_id", values=col).to_string())
fun.to_csv(TAB / "Q3_02d_cleaning_funnel.csv", index=False)

# the cleaned summarisation text, one row per article (this is the input for Q3.3 and Q3.4)
df_sum = df_clean[["article_id", "file", "abstract", "body"]].copy()
df_sum["abstract"] = df_sum["abstract"].apply(lambda s: plain(norm(s)))
df_sum.to_csv(TAB / "Q3_02e_cleaned_for_summarisation.csv", index=False)
print(df_sum.assign(abstract=df_sum["abstract"].str.slice(0, 50), body=df_sum["body"].str.slice(0, 50)).to_string(index=False))

# %% Q3.2 Evidence chart 1: the 20 most frequent words before and after denoising
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def top_words(texts, n=20):
    c = collections.Counter(w for t in texts for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in ENGLISH_STOP_WORDS)
    return c.most_common(n)

tw_before, tw_after = top_words(df_raw["content"]), top_words(df_clean["body"])
gone = {w for w, _ in tw_before} - {w for w, _ in tw_after}          # in the top 20 before, not any more
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for a, data, title in [(ax[0], tw_before, "Top 20 words: raw files"), (ax[1], tw_after, "Top 20 words: after denoising")]:
    words, counts = zip(*data)
    a.barh(range(len(words))[::-1], counts, color=["tab:red" if w in gone else "tab:blue" for w in words])
    a.set_yticks(range(len(words))[::-1]); a.set_yticklabels(words)
    a.set_title(title); a.set_xlabel("count (all five articles)")
savefig(fig, "Q3_02b_top_words_before_after")
print("in the top 20 before, gone after:", sorted(gone))
pd.DataFrame({"rank": range(1, 21), "word_before": [w for w, _ in tw_before], "count_before": [c for _, c in tw_before],
              "word_after": [w for w, _ in tw_after], "count_after": [c for _, c in tw_after]}).to_csv(TAB / "Q3_02f_top_words.csv", index=False)

# %% Q3.2 Evidence chart 2: sentence lengths before and after denoising
def sent_lengths(t):
    return [len(s.split()) for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', norm(t)) if s]

len_before = [n for t in df_raw["content"] for n in sent_lengths(plain(t))]
len_after = [n for t in df_clean["body"] for n in sent_lengths(t)]
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = range(0, 100, 4)
ax.hist(len_before, bins=bins, alpha=0.55, label=f"raw files ({len(len_before)} sentences)", color="tab:orange")
ax.hist(len_after, bins=bins, alpha=0.55, label=f"after denoising ({len(len_after)} sentences)", color="tab:blue")
ax.set_xlabel("words per sentence"); ax.set_ylabel("sentences"); ax.legend(); ax.set_title("Sentence lengths before and after denoising")
savefig(fig, "Q3_02c_sentence_lengths")
summ = pd.DataFrame({"raw": pd.Series(len_before).describe(), "clean": pd.Series(len_after).describe()}).round(1)
print(summ.to_string())
summ.to_csv(TAB / "Q3_02g_sentence_length_summary.csv")

# %% Q3.2 Evidence chart 3: words per article after every rule
fig, ax = plt.subplots(figsize=(10, 5))
piv = fun.pivot(index="step", columns="article_id", values="words")
for aid in piv.columns:
    ax.plot(range(len(piv)), piv[aid], marker="o", label=f"article {aid}")
ax.set_xticks(range(len(piv))); ax.set_xticklabels([s.split(" ", 1)[0] for s in piv.index])
ax.set_xlabel("after rule number (0 = raw file; see table Q3_02d)"); ax.set_ylabel("words"); ax.legend(); ax.set_title("Words per article through the cleaning funnel")
savefig(fig, "Q3_02d_funnel_words")

# %% Q3.2 Evidence chart 4: the noise audit again, on the cleaned text (same patterns as Step 1)
# "before" = the Step 1 audit table (raw files); "after" = the same patterns run on the final cleaned bodies
audit_after = pd.DataFrame({name: [len(re.findall(p, t)) for t in df_clean["body"]] for name, p in AUDIT.items()},
                           index=df_clean["article_id"]).T
audit_after.columns = audit.columns
cmp = pd.DataFrame({"before": audit.sum(axis=1), "after": audit_after.sum(axis=1)})
cmp["after_by_article"] = audit_after.apply(lambda r: list(r), axis=1)
print(cmp.to_string())
cmp.to_csv(TAB / "Q3_02h_noise_before_after.csv")

# every hit that is still there, in context (read these: each one must be explained, not just counted)
for name in cmp.index[cmp["after"] > 0]:
    if name == "blank lines":
        continue                                         # kept on purpose: paragraph breaks in article 4
    for aid, t in zip(df_clean["article_id"], df_clean["body"]):
        for m in re.finditer(AUDIT[name], t):
            print(f"article {aid} | {name[:28]:28} |", repr(t[max(0, m.start() - 40):m.end() + 20]))

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(cmp))
ax.barh(y + 0.2, cmp["before"].clip(lower=0.5), height=0.4, color="tab:orange", label="raw files")
ax.barh(y - 0.2, cmp["after"].clip(lower=0.5), height=0.4, color="tab:blue", label="after denoising")
ax.set_xscale("log")
ax.set_yticks(y); ax.set_yticklabels(cmp.index, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("hits in all five articles (log scale; 0 is drawn as 0.5)")
ax.set_title("Noise audit: raw files vs cleaned text"); ax.legend()
savefig(fig, "Q3_02e_noise_before_after")

# %% Q3.2 Rule 10: token-level text for topic modelling (Q4) - chunks of about 150 words
import sys
sys.path.insert(0, str(BASE / "src"))
from text_utils import clean_text            # the same denoiser as Q1/Q2: lower case, letters only (digits go), stop words, lemmas

CHUNK_WORDS, MIN_LAST = 150, 75               # fixed before looking at any topic output

def make_chunks(text):
    # whole sentences are added until the chunk reaches CHUNK_WORDS; a short last piece joins the chunk before it
    sents = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', norm(text)) if s]
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= CHUNK_WORDS:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        if chunks and sum(len(x.split()) for x in cur) < MIN_LAST:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks

rows = []
for aid, body in zip(df_clean["article_id"], df_clean["body"]):
    for k, c in enumerate(make_chunks(body), start=1):
        toks = clean_text(c).split()
        rows.append({"article_id": aid, "chunk_id": k, "n_words": len(c.split()), "n_tokens": len(toks), "text": c, "tokens": " ".join(toks)})
df_chunks = pd.DataFrame(rows)

chunk_sum = df_chunks.groupby("article_id").agg(chunks=("chunk_id", "count"), words_min=("n_words", "min"), words_mean=("n_words", "mean"),
                                                words_max=("n_words", "max"), tokens_min=("n_tokens", "min"), tokens_mean=("n_tokens", "mean")).round(1)
print(chunk_sum.to_string())
print("total chunks:", len(df_chunks), "| chunks with fewer than 20 tokens:", int((df_chunks["n_tokens"] < 20).sum()))
print("digits left in any token:", int(df_chunks["tokens"].str.contains(r"\d").sum()))
print("example chunk 1, article 4 - text :", df_chunks.loc[(df_chunks.article_id == 4) & (df_chunks.chunk_id == 1), "text"].iloc[0][:200])
print("example chunk 1, article 4 - tokens:", df_chunks.loc[(df_chunks.article_id == 4) & (df_chunks.chunk_id == 1), "tokens"].iloc[0][:200])

# preview of the Q4 filter: in what share of the chunks does each token appear?  (the filter itself is applied in Q4 with gensim)
doc_freq = collections.Counter(w for t in df_chunks["tokens"] for w in set(t.split()))
share = pd.Series(doc_freq).div(len(df_chunks)).sort_values(ascending=False)
print("tokens in more than 40% of all chunks:", {w: round(v, 2) for w, v in share[share > 0.4].items()})
print("tokens in fewer than 3 chunks:", int((pd.Series(doc_freq) < 3).sum()), "of", len(doc_freq))
df_chunks.to_csv(TAB / "Q3_02i_lda_chunks.csv", index=False)
chunk_sum.to_csv(TAB / "Q3_02j_lda_chunk_summary.csv")

# %% Q3.3 Article 1: frequency-based summary (score every sentence by how frequent its words are)
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

ART = 1
BUDGET = int(budget.loc[budget["article_id"] == ART, "abstract_words"].iloc[0])     # 168 words = article 1's abstract length, fixed before running
MIN_W, MAX_W = 10, 60                                                             # only sentences of 10 to 60 words may be chosen
body1 = df_sum.loc[df_sum["article_id"] == ART, "body"].iloc[0]
sents1 = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', norm(body1)) if s]
print("article", ART, "| sentences:", len(sents1), "| word budget:", BUDGET)

def freq_scores(sents):
    # a word's score = its count / the count of the most frequent word; a sentence's score = the mean score of its words
    toks = [clean_text(s).split() for s in sents]
    freq = collections.Counter(w for t in toks for w in t)
    top = max(freq.values())
    wscore = {w: c / top for w, c in freq.items()}
    scores = [sum(wscore[w] for w in t) / len(t) if t else 0.0 for t in toks]
    return toks, freq, scores

def pick(sents, scores, budget, min_w, max_w):
    # best score first (ties: earlier sentence first); a sentence is skipped if it is outside the length gate or does not fit the budget
    order = sorted(range(len(sents)), key=lambda i: (-scores[i], i))
    chosen, used = [], 0
    for i in order:
        n = len(sents[i].split())
        if min_w <= n <= max_w and used + n <= budget:
            chosen.append(i)
            used += n
    return sorted(chosen), used            # sorted = back in reading order

# test 1: a made-up text where the answer is known (the two "alpha" sentences must win; 13 + 15 = 28 words fit a 30-word budget)
toy = ["Alpha values describe how alpha reliability works in many different testing situations today.",
       "Alpha reliability depends on item correlations and on how many items a scale really has.",
       "Completely unrelated gardens bloom beautifully during spring across northern hills every year.",
       "Another unrelated sentence mentions rivers mountains oceans forests deserts valleys plains."]
_, _, toy_scores = freq_scores(toy)
print("test 1 - toy scores:", [round(x, 2) for x in toy_scores], "| chosen:", pick(toy, toy_scores, 30, MIN_W, MAX_W), "(expected ([0, 1], 28))")

toks1, freq1, scores1 = freq_scores(sents1)
chosen1, used1 = pick(sents1, scores1, BUDGET, MIN_W, MAX_W)
summary1 = " ".join(sents1[i] for i in chosen1)

# test 2-4: budget respected, every chosen sentence is an untouched sentence of the article, same answer twice
print("test 2 - summary words:", len(summary1.split()), "<=", BUDGET, ":", len(summary1.split()) <= BUDGET)
print("test 3 - every chosen sentence is in the article text:", all(s in norm(body1) for s in (sents1[i] for i in chosen1)))
print("test 4 - same answer on a second run:", pick(sents1, freq_scores(sents1)[2], BUDGET, MIN_W, MAX_W) == (chosen1, used1))

print("\nmost frequent words (count):", freq1.most_common(12))
print("\nFREQUENCY-BASED SUMMARY OF ARTICLE 1 (%d sentences, %d words)" % (len(chosen1), len(summary1.split())))
for i in chosen1:
    print(f"  [sentence {i + 1:>2}, score {scores1[i]:.2f}, {len(sents1[i].split())} words] {sents1[i]}")

table1 = pd.DataFrame({"position": range(1, len(sents1) + 1), "words": [len(s.split()) for s in sents1], "score": np.round(scores1, 3),
                       "in_length_gate": [MIN_W <= len(s.split()) <= MAX_W for s in sents1], "chosen": [i in chosen1 for i in range(len(sents1))], "sentence": sents1})
table1["rank"] = table1["score"].rank(ascending=False, method="first").astype(int)
table1.to_csv(TAB / "Q3_03a_article1_sentence_scores.csv", index=False)
(TAB / "Q3_03b_article1_frequency_summary.txt").write_text(summary1, encoding="utf-8")
print("\nsentences outside the length gate:", int((~table1["in_length_gate"]).sum()), "| best-scoring sentence outside the gate:",
      table1.loc[~table1["in_length_gate"]].sort_values("rank").head(1)[["position", "words", "rank"]].to_dict("records"))

# %% Q3.3 Evidence: raw frequency vs TF-IDF (the lesson's weighting), word cloud, and the score of every sentence
vec = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}")
tfidf = vec.fit_transform(df_sum["body"])                                    # one row per article (the five cleaned bodies)
terms = np.array(vec.get_feature_names_out())
row = tfidf[df_sum["article_id"].tolist().index(ART)].toarray().ravel()
top_tfidf = row.argsort()[::-1][:15]
top_freq = freq1.most_common(15)
print("top 15 by TF-IDF  :", list(terms[top_tfidf]))
print("top 15 by frequency:", [w for w, _ in top_freq])
print("in both lists:", sorted(set(terms[top_tfidf]) & {w for w, _ in top_freq}))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].barh([w for w, _ in top_freq][::-1], [c for _, c in top_freq][::-1], color="tab:blue")
ax[0].set_title("Article 1: 15 most frequent words"); ax[0].set_xlabel("count")
ax[1].barh(list(terms[top_tfidf])[::-1], row[top_tfidf][::-1], color="tab:green")
ax[1].set_title("Article 1: top 15 TF-IDF terms (against the other four)"); ax[1].set_xlabel("TF-IDF weight")
savefig(fig, "Q3_03a_frequency_vs_tfidf")

cloud = WordCloud(width=1000, height=500, background_color="white", random_state=SEED).generate_from_frequencies(dict(freq1))
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(cloud, interpolation="bilinear"); ax.axis("off"); ax.set_title("Article 1: word cloud (word frequency)")
savefig(fig, "Q3_03b_wordcloud")

fig, ax = plt.subplots(figsize=(12, 4))
colour = ["tab:red" if c else ("lightgrey" if not g else "tab:blue") for c, g in zip(table1["chosen"], table1["in_length_gate"])]
ax.bar(table1["position"], table1["score"], color=colour)
ax.set_xlabel("sentence position in article 1"); ax.set_ylabel("score (mean word frequency score)")
ax.set_title("Sentence scores: red = chosen, blue = candidate not chosen, grey = outside the 10-60 word gate")
savefig(fig, "Q3_03c_sentence_scores")