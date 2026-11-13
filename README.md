# ITNPA4-34 Project 1 – Text Mining, Classification, Summarisation and Topic Modelling

**Author:** Michael van Niekerk (EDUV8426028) · Eduvos, ITNPA4-34

This project is built around the following:
* Python programming
* Text 
  - preprocessing
  - modeling
  - summarisation and analysis
  - classification
* analytics
* basic deployment
 
## Status

| Question | Content | Status |
|---|---|---|
| Q1 | Abstract data: load, clean, features, baseline | Done |
| Q2 | Four classifiers, enhancement, web app | In progress |
| Q3 | Corpus summarisation | Not started |
| Q4 | LDA topics, pyLDAvis, this repository | Not started |

## Repository structure

    ITNPA4-34-Project1/
    ├── data/
    │   ├── research_abstracts.json     # Q1–Q2: labelled abstracts (ai / classical)
    │   └── research_corpus.zip         # Q3–Q4: article corpus
    ├── notebooks/
    │   └── Q1_Q2_abstract_classification.py   # Spyder script with # %% cells
    ├── src/
    │   ├── text_utils.py               # text cleaning (also used by the deployed model)
    │   └── summarise_topics.py         # summarisation and LDA helpers
    ├── outputs/
    │   ├── figures/                    # every chart, saved as PNG
    │   └── tables/                     # every results table, saved as CSV
    ├── deployment/                     # Streamlit app (Q2.3, to come)
    ├── docs/                           # report
    ├── requirements.txt
    └── README.md

## How to run

Requires Python 3.11.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python notebooks/Q1_Q2_abstract_classification.py

Or open the script in Spyder and run the cells in order. Paths are relative to
the repository, so it runs from any location. The first run downloads the NLTK
data it needs (stop words, WordNet, tokenisers). Figures and tables are written
to `outputs/`.

## Results so far

**Data:** 1,138 abstracts → 1,125 after removing 13 duplicates → 1,119 after
excluding 6 rows under rules R1/R2 (see notebook). Classes: 889 classical / 230 ai (≈ 79 / 21).

**Baselines (Q1.3)** – stratified 80/20 split (895 / 224), 5-fold stratified CV:

| Model | CV train macro-F1 | CV val macro-F1 | Test macro-F1 | Test bal. acc. | Test AI recall |
|---|---|---|---|---|---|
| Dummy (most frequent) | 0.443 | 0.443 | 0.443 | 0.500 | 0.000 |
| Multinomial NB + BoW | 0.958 | 0.843 | 0.852 | 0.858 | 0.783 |

What the Dummy shows about accuracy:
The Dummy tends to ignore the abstract and always predicts "classical" which is the most common class.
Even with the above to be true, the Dummy still scores about 79.5% on accuracy, this means that the Dummy accuracy scores are just the share of the dataset that the majority class holds.
The Dummy finds non of the 46 AI papers, which means its AI recall is nonexistent. AI was the class the classifier exists to find, and this means the Dummy classifier is next to useless.
The bottom line, when the classes are split 79/21, even a model that learned nothing gets nearly 80% accuracy purely by choosing the majority class every time. Accuracy rewards ignoring the minority class, in this case AI.
The NB result is 90.2% accuracy, while this may look like a small improvement over the Dummys 79.5% accuracy, its actually the proof that the model has started learning and is actively recalling AI as a class now. its not merely relying on choosing the majority class every time. this is a major positive result and indicates and AI recall of close to 78%.

The metrics that treat both classes equally show the real difference:

Measure			Dummy	NB
------------------------------------------
Accuracy		0.795	0.902
Macro-F1		0.443	0.852
Balanced accuracy	0.500	0.858
AI recall		0.000	0.783

What the train-validation gap means
NB scores 0.958 macro-F1 on the abstracts it was trained on and 0.843 on the cross-validation folds it didn't see which give a gap of about 0.115.
This means the model performs better on data it has seen before than on new data. This indicates that part of what it learned is specific to the training abstracts and does not carry over. 
This would indicate either overfitting or memorisation.
When a word appears in only a handful of training abstracts, all from one class, it gets a strong probability for that class. On new abstracts, those words either wont appear or point the wrong way to a different class than it should be pointing to.

In this particular case, the validation is till high at 0.843, which indicates that while the model overfits a little it still generalises well. A gap with a low validation score would become a problem.
The test macro-F1 comes in at 0.852 which is within one standard deviation (0.22) of the CV estimate 0.843 which indicates that cross-validation gave a trustworthy prediction of performance on new data and the 224 test abstracts confirmed it.
The learning curve shows that the gap shrinks with more data, trining F1 falls from 1 to 0.958 and validation F1 rises from 0.759 to 0.843 which narrows the gap from about 0.24 to 0.115 and validation is still creeping up to full size.
This shows that with more data it would help, but since the curve is flattening dont expect too much gains.

The Dummys train and validations scores are identical (o.443), it never developed a gap because it never learned anything. A gap is what happenes when a model starts to learn, the question is, will what the model learned be carried over to new data? In this case, for NB it mostly does.

## Key decisions
* Why DOI was not used as a feature:
The DOI records where a paper was published, not what it says. 
The deployed app only receives abstract text, so DOI wont even be available there.
- A test model was included in the scripts. It shows the following:
-- a model using only the DOI prefix reached 0.679 test macro-F1 (Dummy: 0.443) without reading any text. So the publisher carries real signal, which makes it a leakage risk.
-- Its accuracy (0.763) was below the Dummy’s (0.795), so accuracy hides that signal.
-- 24 of 224 test abstracts come from publishers never seen in training, and 4 rows have no DOI at all. 
-- A DOI-based model would fail on new or rare publishers.

* Why rows were excluded (R1/R2), and the sensitivity check result
- A row was excluded only when two independent pieces of evidence agree.
-- Rule 1: row 1122 is 13 abstracts joined together (2,075 words, the closing phrase repeated 13 times, 11 author bylines), so one label can’t describe it.
-- Rule 2: five “ai”-labelled rows are cosmology papers about cosmic inflation, with physics vocabulary, arXiv physics categories and zero AI/ML terms. The label is contradicted by the content.
-- Why one piece of evidence isn’t enough: 42% of genuine AI abstracts also contain no ML terms, and the classical-labelled physics row (513) was kept because its label isn’t contradicted.
-- Sensitivity check: 0.843 CV macro-F1 with exclusions vs 0.829 without, and AI recall 0.712 vs 0.692. That’s a small difference, so the exclusions don’t inflate the results.

* Why chi-square was run on counts instead of TF-IDF
- Chi-square compares observed and expected frequencies, so it needs counts. TF-IDF values are weights scaled per abstract.
-- Scaling test: multiplying TF-IDF by 10 changed the significant terms from 42 to 2,345, and every score rose exactly 10×. The result depended on the units, not the data.
-- On counts, 2,844 terms are significant. The scores are treated as a ranking, because words within one abstract aren’t independent.

* Why macro-F1 / balanced accuracy instead of accuracy
- With a 79/21 class split, the Dummy reaches 0.795 accuracy by always predicting “classical”, while finding 0 of 46 AI papers.
- Macro-F1 and balanced accuracy weight both classes equally, so ignoring the AI class costs half the score: the Dummy drops to 0.443 and 0.500.
- Second piece of evidence: accuracy ranks the DOI-only model below the Dummy, even though it finds 61% of AI papers.

## Deployment

<To come in Q2.3: how to run the Streamlit app, plus the public link if deployed.>

## Data sources

- `research_abstracts.json` and `research_corpus.zip`: provided with the ITNPA4-34 Project 1 brief.

## References

<Harvard style, as in your report. For Q1 at least:>
- Bird, S., Klein, E. and Loper, E. (2009) *Natural Language Processing with Python*. O'Reilly.
- Manning, C.D., Raghavan, P. and Schütze, H. (2008) *Introduction to Information Retrieval*. Cambridge University Press.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

## AI use

<One line pointing to the AI Disclosure Appendix in the report, in your own words.>